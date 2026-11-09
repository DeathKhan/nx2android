#!/usr/bin/env python3
"""Real MRT insertion, selector and runtime ABI checks; no game/device needed."""
from pathlib import Path
import random
import subprocess
import unittest
import cca_mrt_probe as mrt
import spv_intermediate as spv
from test_cca_lighting_modes import execute
from test_spv_runtime_contract import RuntimeContractTests

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dist/cca-intermediate-probes/runtime-origin/original.spv"
OUT = ROOT / "dist/cca-intermediate-probes/mrt-v1/cca3264814bd7187.spv"


class MrtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SOURCE.read_bytes()
        cls.output, cls.meta = mrt.instrument(cls.base)
        cls.h, cls.ins = spv.decode(cls.output)
        cls.proof = cls.meta["proof"]
        RuntimeContractTests.setUpClass()
        cls.contract = RuntimeContractTests()

    def test_original_instruction_identity_and_capture_order(self):
        groups = [self.proof[k] for k in ("annotations", "declarations", "initialization", "before_return")]
        groups += list(self.proof["captures"].values())
        added = [(op, a) for group in groups for op, a in group]
        keep, removed = [], []
        for op, a in self.ins:
            if (op, a) in added:
                removed.append((op, a))
            else:
                if op == 15:
                    a = a[:-1]  # sole appended Output for this SPIR-V1.3 module
                keep.append((op, a))
        self.assertEqual(len(removed), len(added))
        header = list(self.h)
        header[3] = self.proof["source_bound"]
        self.assertEqual(spv.encode(header, keep), self.base)
        self.assertEqual(OUT.read_bytes(), self.output)
        # Added stores target only fresh private vars and output1, never output0.
        self.assertTrue(all(a[0] >= header[3] for op, a in added if op == 62))
        for target, capture in self.proof["captures"].items():
            at = next(i for i, (op, a) in enumerate(self.ins)
                      if op in (12, 61, 133, 87) and a[1] == target)
            self.assertEqual(self.ins[at + 1:at + 1 + len(capture)], capture)
        with self.assertRaises(ValueError):
            mrt.instrument(self.base[:-4])

    def test_actual_selectors_preserve_output0_and_raw_bits(self):
        rand = random.Random(0xCCA396)
        for bits in (0, 0x80000000, 0x7F800000, 0x7FA12345, 0xFFC12345, *[rand.getrandbits(32) for _ in range(16)]):
            original = (bits, bits ^ 1, bits ^ 2, bits ^ 3)
            sample = (bits ^ 4, bits ^ 5, bits ^ 6, bits ^ 7)
            for mode in (0, *range(1, 13), 16, 0xFFFFFFFF):
                for taken in (False, True):
                    env, memory = {}, {13: [0] * 7 + [mode], 91: original}
                    execute([(op, a) for op, a in self.ins if op in (43, 44, 59)], env, memory)
                    execute(self.proof["initialization"], env, memory)
                    for target, sequence in self.proof["captures"].items():
                        env[target] = sample if target in mrt.SAMPLES.values() else bits
                        if taken:
                            execute(sequence, env, memory)
                    execute(self.proof["before_return"], env, memory)
                    expected = original
                    if mode in mrt.SAMPLES:
                        expected = sample if taken else (0, 0, 0, 0)
                    elif mode == 2:
                        expected = (bits,) * 4 if taken else (0, 0, 0, 0)
                    elif mode in (11, 12):
                        expected = ((bits,) * 3 if taken else (0, 0, 0)) + (0x3F800000,)
                    self.assertEqual(memory[91], original)
                    self.assertEqual(memory[self.proof["output_id"]], expected)

    def test_layer_coordinate_variant_preserves_original_and_other_modes(self):
        output, meta = mrt.instrument(self.base, "layer-coordinates")
        h, ins = spv.decode(output)
        proof = meta["proof"]
        groups = [proof[k] for k in ("annotations", "declarations", "initialization", "before_return")]
        groups += list(proof["captures"].values())
        added = [(op, a) for group in groups for op, a in group]
        keep = [(op, a[:-1] if op == 15 else a) for op, a in ins if (op, a) not in added]
        h[3] = proof["source_bound"]
        self.assertEqual(spv.encode(h, keep), self.base)
        self.assertEqual(meta["scalar_ids"][2], mrt.LAYER_COORDS)
        for target in mrt.LAYER_COORDS:
            pos = next(i for i, (op, a) in enumerate(ins) if op == 111 and a[1] == target)
            self.assertEqual(ins[pos + 1:pos + 2], proof["captures"][target])
        expected_layers = (0x3F800000, 0x3F800000, 0x40000000, 0x40000000)
        for mode in (0, *range(1, 13), 16):
            results = []
            for data, m in ((self.output, self.meta), (output, meta)):
                _, code = spv.decode(data)
                env, memory = {}, {13: [0] * 7 + [mode], 91: (1, 2, 3, 4)}
                execute([(op, a) for op, a in code if op in (43, 44, 59)], env, memory)
                execute(m["proof"]["initialization"], env, memory)
                for target, sequence in m["proof"]["captures"].items():
                    env[target] = ((5, 6, 7, 8) if target in mrt.SAMPLES.values() else
                                   expected_layers[mrt.LAYER_COORDS.index(target)] if target in mrt.LAYER_COORDS else 9)
                    execute(sequence, env, memory)
                execute(m["proof"]["before_return"], env, memory)
                self.assertEqual(memory[91], (1, 2, 3, 4))
                results.append(memory[m["proof"]["output_id"]])
            self.assertEqual(results[1], expected_layers if mode == 2 else results[0])
        target = ROOT / "dist/cca-intermediate-probes/layer-coords/cca3264814bd7187.spv"
        self.assertEqual(target.read_bytes(), output)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(target)], check=True)
        self.contract.check(self.base, output, True)
        with self.assertRaises(ValueError):
            mrt.instrument(self.base, "unrecognized")

    def test_actual_layer_constant_transport_has_no_relaxed_precision(self):
        _, ins = spv.decode(self.base)
        # SPIR-V decoration0 is RelaxedPrecision; neither direct nor member annotations use it.
        self.assertFalse(any(op == 71 and a[1] == 0 or op == 72 and a[2] == 0 for op, a in ins))
        self.assertIn((43, [4, 1300, 1]), ins)
        self.assertIn((43, [4, 2236, 2]), ins)
        self.assertIn((59, [99, 229, 7]), ins)  # Function float register for layer1
        self.assertIn((59, [99, 519, 7]), ins)  # Function float register for layer2
        for target, source, register in ((1349, 1348, 229), (1383, 1382, 229),
                                         (2308, 2307, 519), (2347, 2346, 519)):
            self.assertIn((111, [4, target, source]), ins)
            cast = next(a for op, a in ins if op == 124 and a[1] == source)
            self.assertEqual(cast[0], 6)
            self.assertIn((61, [4, cast[2], register]), ins)

    def test_vulkan_and_exact_mrt_contract_rejections(self):
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(OUT)], check=True)
        self.contract.check(self.base, self.output, True)
        oid = self.proof["output_id"]
        mutations = []
        for kind in ("location2", "missing_location", "missing_interface", "duplicate_interface",
                     "wrong_output_type", "extra_output", "new_resource", "duplicate_location"):
            h, ins = spv.decode(self.output)
            if kind == "location2":
                ins = [(op, [oid, 30, 2] if op == 71 and a == [oid, 30, 1] else a) for op, a in ins]
            elif kind == "missing_location":
                ins.remove((71, [oid, 30, 1]))
            elif kind == "missing_interface":
                ins = [(op, a[:-1] if op == 15 else a) for op, a in ins]
            elif kind == "duplicate_interface":
                ins = [(op, a + [oid] if op == 15 else a) for op, a in ins]
            elif kind == "wrong_output_type":
                ins = [(op, [3131, oid, 3] if op == 59 and a[1] == oid else a) for op, a in ins]
            elif kind == "duplicate_location":
                at = next(i for i, x in enumerate(ins) if x == (71, [oid, 30, 1]))
                ins.insert(at, (71, [oid, 30, 1]))
            else:
                at = next(i for i, (op, _) in enumerate(ins) if op == 54)
                ins.insert(at, (59, [89 if kind == "extra_output" else 11, h[3], 3 if kind == "extra_output" else 2]))
                h[3] += 1
            mutations.append((kind, spv.encode(h, ins)))
        for kind, data in mutations:
            with self.subTest(kind=kind):
                self.contract.check(self.base, data, False)


if __name__ == "__main__":
    unittest.main()
