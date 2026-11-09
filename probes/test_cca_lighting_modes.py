#!/usr/bin/env python3
"""Execute the actual v2 capture/selector instructions and preserve the v1 stream."""
from pathlib import Path
import random
import subprocess
import unittest
import cca_lighting_gate as gate
import cca_lighting_modes as modes
import spv_intermediate as spv

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dist/cca-intermediate-probes/runtime-origin/original.spv"
OUT = ROOT / "dist/cca-intermediate-probes/live-lighting-v2/cca3264814bd7187.spv"


def execute(ins, env, memory):
    def read(pointer):
        root, path = pointer
        v = memory[root]
        for index in path:
            v = v[index]
        return v
    for op, a in ins:
        if op == 43:
            env[a[1]] = a[2]
        elif op == 44 or op == 80:
            env[a[1]] = tuple(env[x] for x in a[2:])
        elif op == 59:
            env[a[1]] = (a[1], ())
        elif op == 61:
            env[a[1]] = read(env[a[2]])
        elif op == 62:
            root, path = env[a[0]]
            if path:
                raise AssertionError("the probe stores whole private/output variables only")
            memory[root] = env[a[1]]
        elif op == 65:
            root, path = env[a[2]]
            env[a[1]] = (root, path + tuple(env[x] for x in a[3:]))
        elif op == 81:
            v = env[a[2]]
            for index in a[3:]:
                v = v[index]
            env[a[1]] = v
        elif op == 194:
            env[a[1]] = (env[a[2]] & 0xFFFFFFFF) >> env[a[3]]
        elif op == 199:
            env[a[1]] = env[a[2]] & env[a[3]]
        elif op in (170, 171):
            env[a[1]] = (env[a[2]] == env[a[3]]) if op == 170 else (env[a[2]] != env[a[3]])
        elif op == 169:
            c, t, f = (env[x] for x in a[2:])
            env[a[1]] = tuple(tv if cv else fv for cv, tv, fv in zip(c, t, f)) if isinstance(c, tuple) else (t if c else f)
        elif op in (23, 32):
            pass
        else:
            raise AssertionError((op, a))


class LiveModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_bytes()
        cls.gated, _ = gate.instrument(cls.source)
        cls.output, cls.meta = modes.instrument(cls.source)
        cls.h, cls.ins = spv.decode(cls.output)
        cls.proof = cls.meta["v2_proof"]

    def test_exact_v1_instruction_recovery_and_immutable_artifact(self):
        groups = [self.proof[k] for k in ("declarations", "initialization", "selector", "before_return")]
        groups += list(self.proof["captures"].values())
        added = [(op, a) for group in groups for op, a in group]
        keep = []
        removed = []
        for instruction in self.ins:
            if instruction in added:
                removed.append(instruction)
            else:
                keep.append(instruction)
        self.assertEqual(len(removed), len(added))
        header = list(self.h)
        header[3] = self.proof["gated_bound"]
        self.assertEqual(spv.encode(header, keep), self.gated)
        self.assertEqual((ROOT / "dist/cca-intermediate-probes/live-lighting/cca3264814bd7187.spv").read_bytes(), self.gated)
        self.assertEqual(OUT.read_bytes(), self.output)
        # Captures follow the exact source operation, before any low-bit final gate.
        for target, sequence in self.proof["captures"].items():
            at = next(i for i, (op, a) in enumerate(self.ins) if op in (12, 87) and a[1] == target)
            self.assertEqual(self.ins[at + 1:at + 1 + len(sequence)], sequence)

    def run_modes(self, mask, payload, sample_branch=True):
        env, memory = {}, {13: [0] * 7 + [mask], 80: payload["uv"], 91: payload["original"]}
        declarations = [(op, a) for op, a in self.ins if op in (43, 44, 59)]
        execute(declarations, env, memory)
        execute(self.proof["initialization"], env, memory)
        execute(self.meta["prefix"], env, memory)
        execute(self.proof["selector"], env, memory)
        env[1484], env[1521] = payload["rgb"], payload["scalar"]
        for result, value in zip(gate.DIRECT + gate.INDIRECT, payload["direct"] + payload["indirect"]):
            env[self.meta["gates"][result]["raw_result"]] = value
        for target, capture in self.proof["captures"].items():
            if sample_branch or target not in (1484, 1521):
                execute(capture, env, memory)
        execute(self.proof["before_return"], env, memory)
        return memory[91]

    def test_actual_selector_modes_branch_zeros_and_raw_payloads(self):
        random_bits = random.Random(0xCCA250)
        cases = [0, 0x80000000, 0x7F800000, 0x7FA12345, 0xFFC12345]
        cases += [random_bits.getrandbits(32) for _ in range(64)]
        one = 0x3F800000
        for bits in cases:
            payload = {"original": (bits, bits ^ 0x1234, bits ^ 0x5678, bits ^ 0x9ABC),
                       "rgb": (bits ^ 1, bits ^ 2, bits ^ 3, bits ^ 4),
                       "scalar": (bits ^ 5, bits ^ 6, bits ^ 7, bits ^ 8),
                       "uv": (bits ^ 9, bits ^ 10, bits ^ 11, bits ^ 12),
                       "direct": (bits ^ 13, bits ^ 14, bits ^ 15),
                       "indirect": (bits ^ 16, bits ^ 17, bits ^ 18)}
            for sample_branch in (False, True):
                expected = {1: (0x3F600000, 0x3E000000, 0x3F600000, one),
                            2: payload["rgb"] if sample_branch else (0, 0, 0, 0),
                            3: (payload["scalar"][0],) * 3 + (one,) if sample_branch else (0, 0, 0, 0),
                            4: payload["uv"][:2] + (0, one),
                            5: payload["direct"] + (one,), 6: payload["indirect"] + (one,)}
                for mode in range(16):
                    for low in (0, 1, 15, 16):
                        # Low-bit arithmetic was tested in v1. This interpreter
                        # injects the already-computed original/gated outputs.
                        mask = mode << 8 | low
                        self.assertEqual(self.run_modes(mask, payload, sample_branch),
                                         expected.get(mode, payload["original"]))
                self.assertEqual(self.run_modes(0x80000010, payload, sample_branch), payload["original"])

    def test_real_vulkan_validation_and_runtime_contract(self):
        subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(OUT)], check=True)
        subprocess.run([str(ROOT / "dist/tests/spv-intermediate/runtime-contract"), str(SOURCE), str(OUT), "1"], check=True)


if __name__ == "__main__":
    unittest.main()
