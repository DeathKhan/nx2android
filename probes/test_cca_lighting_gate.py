#!/usr/bin/env python3
"""Actual runtime-module diagnostic gate checks; no production shader fixtures."""
from pathlib import Path
import random
import subprocess
import unittest
import cca_lighting_gate as gate
import spv_intermediate as spv

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dist/cca-intermediate-probes/runtime-origin/original.spv"
OUT = ROOT / "dist/cca-intermediate-probes/live-lighting/cca3264814bd7187.spv"


def evaluate_control(instructions, constants, mask, values):
    """Execute the emitted integer/load/extract/select operations with raw bits."""
    env = dict(constants)
    env.update(values)
    env[13] = [0] * 7 + [mask]
    for op, a in instructions:
        if op == 61:
            env[a[1]] = env[a[2]]
        elif op == 81:
            v = env[a[2]]
            for index in a[3:]:
                v = v[index]
            env[a[1]] = v
        elif op == 199:
            env[a[1]] = env[a[2]] & env[a[3]]
        elif op == 171:
            env[a[1]] = env[a[2]] != env[a[3]]
        elif op == 169:
            env[a[1]] = env[a[3]] if env[a[2]] else env[a[4]]
        else:
            raise AssertionError((op, a))
    return env


class LightingGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_bytes()
        cls.output, cls.metadata = gate.instrument(cls.source)
        cls.original_header, cls.original = spv.decode(cls.source)
        cls.header, cls.rewritten = spv.decode(cls.output)

    def test_original_instruction_sequence_is_exactly_recoverable(self):
        """No original math, memory access, control flow or declaration changes."""
        meta = self.metadata
        prefix = [(op, a) for op, a in meta["prefix"]]
        raw = {g["raw_result"]: result for result, g in meta["gates"].items()}
        recovered = []
        prefix_count = select_count = 0
        for op, original_args in self.rewritten:
            a = list(original_args)
            if (op, a) in prefix:
                prefix_count += 1
                continue
            if op == 169 and a[1] in meta["gates"]:
                g = meta["gates"][a[1]]
                self.assertEqual(a, [4, a[1], g["condition"], g["replacement"], g["raw_result"]])
                select_count += 1
                continue
            if op in (12, 124) and a[1] in raw:
                a[1] = raw[a[1]]
            recovered.append((op, a))
        self.assertEqual(prefix_count, 10)
        self.assertEqual(select_count, 30)
        self.assertEqual(spv.encode(self.original_header, recovered), self.source)
        self.assertEqual(self.original[:next(i for i, (op, _) in enumerate(self.original) if op == 54)],
                         self.rewritten[:next(i for i, (op, _) in enumerate(self.rewritten) if op == 54)])

    def test_all_masks_preserve_or_select_exact_raw_bits(self):
        constants = {a[1]: a[2] for op, a in self.original if op == 43 and len(a) == 3}
        selects = {a[1]: (op, a) for op, a in self.rewritten if op == 169 and a[1] in self.metadata["gates"]}
        random_bits = random.Random(0xCCA326)
        payloads = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001,
                    0x7FA12345, 0xFFC12345, 1, 0x007FFFFF, 0x3F800000]
        payloads += [random_bits.getrandbits(32) for _ in range(128)]
        for mask in list(range(32)) + [0x80000000, 0x8000000F, 0xFFFFFFFF]:
            for result, g in self.metadata["gates"].items():
                for bits in payloads:
                    replacement = constants.get(g["replacement"], bits ^ 0x5A5A5A5A)
                    values = {g["raw_result"]: bits, g["replacement"]: replacement}
                    env = evaluate_control(self.metadata["prefix"] + [selects[result]], constants, mask, values)
                    self.assertEqual(env[result], replacement if mask & g["mask"] else bits)
        # Negative control: a reversed select must break the baseline guarantee.
        result, g = next(iter(self.metadata["gates"].items()))
        op, a = selects[result]
        wrong = a[:3] + [a[4], a[3]]
        env = evaluate_control(self.metadata["prefix"] + [(op, wrong)], constants, 0,
                               {g["raw_result"]: 0x7FC01234, g["replacement"]: 0})
        self.assertNotEqual(env[result], 0x7FC01234)

    def test_provenance_scope_and_real_runtime_acceptance(self):
        groups = {mask: [g for g in self.metadata["gates"].values() if g["mask"] == mask]
                  for mask in (1, 2, 4, 8)}
        self.assertEqual([len(groups[m]) for m in groups], [12, 12, 3, 3])
        self.assertEqual([g["byte_offset"] for g in groups[1]],
                         [48, 52, 56, 64, 68, 72, 80, 84, 88, 96, 100, 104])
        for g in groups[2]:
            self.assertEqual(g["original"][0], 12)
            self.assertEqual(g["original"][1][3], 50)  # GLSL Fma
            self.assertEqual(g["replacement"], g["original"][1][6])
        self.assertIn((72, [10, 7, 35, 1376]), self.original)
        self.assertEqual(OUT.read_bytes(), self.output)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(OUT)], check=True)
        subprocess.run([str(ROOT / "dist/tests/spv-intermediate/runtime-contract"),
                        str(SOURCE), str(OUT), "1"], check=True)
        bad = bytearray(self.source)
        bad[-1] ^= 1
        with self.assertRaises(ValueError):
            gate.instrument(bytes(bad))


if __name__ == "__main__":
    unittest.main()
