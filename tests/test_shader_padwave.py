# shader_padwave.py: a 32-lane compute kernel padded to 64 lanes for a 64-lane-wave GPU (the Odin 3's Adreno 830 driver fails Ocarina of Time's NDVE movie kernels at
# LocalSize 32). A tiny hand-written kernel (assembled with spirv-as when it is on the PATH), and Ocarina of Time's e6be82d835fbe4d3 when its SPIR-V is at hand
# (DM_PADWAVE_SAMPLE=<dir with <key>.spv>): the output validates, its LocalSize is 64xNx1, every store to Workgroup / storage / BDA / image memory and every atomic
# sits in a block entered only under the lane < 32 predicate (mode branch) or behind an OpSelect of its slot / coordinate (mode select), the barrier and shuffle counts are
# unchanged, and a padded module pads no further.
import os, shutil, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'nxnative/dm/android/tools'))
import shader_padwave as pw   # noqa: E402

SPIRV_AS, SPIRV_VAL = shutil.which('spirv-as'), shutil.which('spirv-val')
SAMPLE = Path(os.environ.get('DM_PADWAVE_SAMPLE', '')) / 'e6be82d835fbe4d3.spv'

# 32x2x1: a lane's slot of a shared array written, a barrier, the neighbour's slot read with a shuffle and stored to a storage buffer, an atomic counter bumped, an image write
TINY = '''
               OpCapability Shader
               OpCapability GroupNonUniformShuffle
               OpCapability StorageImageWriteWithoutFormat
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %lid %sgid
               OpExecutionMode %main LocalSize 32 2 1
               OpDecorate %lid BuiltIn LocalInvocationId
               OpDecorate %sgid BuiltIn SubgroupLocalInvocationId
               OpDecorate %rarr ArrayStride 4
               OpDecorate %sb BufferBlock
               OpMemberDecorate %sb 0 Offset 0
               OpDecorate %ssbo DescriptorSet 0
               OpDecorate %ssbo Binding 0
               OpDecorate %img DescriptorSet 0
               OpDecorate %img Binding 1
       %void = OpTypeVoid
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
     %v3uint = OpTypeVector %uint 3
     %v4float = OpTypeVector %float 4
     %v2int = OpTypeVector %uint 2
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
    %uint_31 = OpConstant %uint 31
    %uint_32 = OpConstant %uint 32
   %uint_264 = OpConstant %uint 264
   %uint_n32 = OpConstant %uint 4294967264
    %float_1 = OpConstant %float 1
      %fvec = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
  %arr32 = OpTypeArray %uint %uint_32
%_ptr_Workgroup_arr = OpTypePointer Workgroup %arr32
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
     %shared = OpVariable %_ptr_Workgroup_arr Workgroup
       %rarr = OpTypeRuntimeArray %uint
         %sb = OpTypeStruct %rarr
%_ptr_Uniform_sb = OpTypePointer Uniform %sb
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
       %ssbo = OpVariable %_ptr_Uniform_sb Uniform
      %image = OpTypeImage %float 2D 0 0 0 2 Unknown
%_ptr_UniformConstant_image = OpTypePointer UniformConstant %image
        %img = OpVariable %_ptr_UniformConstant_image UniformConstant
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%_ptr_Input_uint = OpTypePointer Input %uint
        %lid = OpVariable %_ptr_Input_v3uint Input
       %sgid = OpVariable %_ptr_Input_uint Input
     %fnty = OpTypeFunction %void
     %helper_ty = OpTypeFunction %void %uint
%_ptr_Function_uint = OpTypePointer Function %uint
       %main = OpFunction %void None %fnty
      %entry = OpLabel
        %tmp = OpVariable %_ptr_Function_uint Function
       %lidv = OpLoad %v3uint %lid
          %x = OpCompositeExtract %uint %lidv 0
          %y = OpCompositeExtract %uint %lidv 1
               OpStore %tmp %x
       %slot = OpAccessChain %_ptr_Workgroup_uint %shared %x
               OpStore %slot %y
               OpControlBarrier %uint_3 %uint_3 %uint_264
      %xp = OpAccessChain %_ptr_Input_uint %lid %uint_0
      %xl = OpLoad %uint %xp
      %nb = OpBitwiseXor %uint %xl %uint_1
     %nslot = OpAccessChain %_ptr_Workgroup_uint %shared %nb
       %nval = OpLoad %uint %nslot
       %sg = OpLoad %uint %sgid
     %base = OpBitwiseAnd %uint %sg %uint_n32
     %lane = OpBitwiseAnd %uint %sg %uint_31
     %lx = OpBitwiseXor %uint %lane %uint_2
     %idx = OpBitwiseOr %uint %lx %base
       %shf = OpGroupNonUniformShuffle %uint %uint_3 %nval %idx
      %cond = OpUGreaterThan %bool %shf %uint_2
               OpSelectionMerge %join None
               OpBranchConditional %cond %then %join
       %then = OpLabel
       %oslot = OpAccessChain %_ptr_Uniform_uint %ssbo %uint_0 %x
               OpStore %oslot %shf
               OpBranch %join
       %join = OpLabel
       %pv = OpPhi %uint %uint_1 %then %uint_0 %entry
       %cslot = OpAccessChain %_ptr_Uniform_uint %ssbo %uint_0 %uint_0
        %old = OpAtomicIAdd %uint %cslot %uint_1 %uint_0 %pv
       %sum = OpIAdd %uint %old %x
       %oslot2 = OpAccessChain %_ptr_Uniform_uint %ssbo %uint_0 %uint_1
               OpStore %oslot2 %sum
       %imv = OpLoad %image %img
       %co = OpCompositeConstruct %v2int %x %y
               OpImageWrite %imv %co %fvec
       %call = OpFunctionCall %void %helper %x
               OpReturn
               OpFunctionEnd
     %helper = OpFunction %void None %helper_ty
       %harg = OpFunctionParameter %uint
     %hentry = OpLabel
       %hs = OpAccessChain %_ptr_Workgroup_uint %shared %harg
               OpStore %hs %harg
               OpReturn
               OpFunctionEnd
'''


def assemble(text):
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'a.spvasm').write_text(text, encoding='utf-8')
        subprocess.run([SPIRV_AS, '--target-env', 'vulkan1.1', Path(d, 'a.spvasm'), '-o', Path(d, 'a.spv')], check=True)
        return Path(d, 'a.spv').read_bytes()


def validate(spv):
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'a.spv').write_bytes(spv)
        r = subprocess.run([SPIRV_VAL, '--target-env', 'vulkan1.1', Path(d, 'a.spv')], capture_output=True, text=True)
        return r.returncode == 0, r.stderr


def predicated_blocks(spv):
    """{label: True} of blocks entered only as the 'then' of an OpBranchConditional on a lane < 32 predicate (OpULessThan of LocalInvocationId.x and 32), per function"""
    m = pw._Mod(spv); lid = m.builtin[pw.BI_LOCAL_ID]; preds = set()
    for x in m.ins:
        if x.op == pw.OP_ULESS:   # OpULessThan %bool %x %c32 with %x = OpCompositeExtract 0 of OpLoad lid
            xv = m.by_id.get(x.a[2])
            if xv is not None and xv.op == pw.OP_EXTRACT and xv.a[3] == 0 and m.by_id[xv.a[2]].op == pw.OP_LOAD and m.by_id[xv.a[2]].a[2] == lid and m.const_value(x.a[3]) == 32: preds.add(x.a[1])
    then = {x.a[1] for x in m.ins if x.op == pw.OP_BRANCH_COND and x.a[0] in preds}   # the true target
    # each block: its label, and whether every side effect in it sits in a predicated block
    bad, label = [], None
    for x in m.ins[m.first_fn:]:
        if x.op == pw.OP_LABEL: label = x.a[0]
        elif x.op in (pw.OP_STORE, pw.OP_COPY_MEMORY):
            if m.storage_class(x.a[0]) in pw.SE_KIND and label not in then: bad.append((x.op, label))
        elif x.op == pw.OP_IMAGE_WRITE or x.op in pw.ATOMICS:
            if label not in then: bad.append((x.op, label))
    return preds, then, bad


def local_size(spv):
    return pw.info(spv)['local']


class PadWave(unittest.TestCase):
    @unittest.skipUnless(SPIRV_AS and SPIRV_VAL, 'needs spirv-as and spirv-val')
    def test_tiny(self):
        spv = assemble(TINY)
        self.assertTrue(validate(spv)[0]); self.assertTrue(pw.wants_pad(spv)); self.assertEqual(local_size(spv), [32, 2, 1])
        out, st = pw.pad(spv)
        ok, err = validate(out); self.assertTrue(ok, err)
        self.assertEqual(local_size(out), [64, 2, 1]); self.assertEqual(st['local'], [64, 2, 1])
        self.assertEqual(st['stores'], {'workgroup': 2, 'storage': 2}); self.assertEqual(st['atomics'], 1); self.assertEqual(st['image_writes'], 1); self.assertEqual(st['functions'], 2)
        self.assertEqual(st['lid_loads'], 2)   # the vec3 load and the .x load, both masked
        self.assertEqual(st['barriers'], 1); self.assertEqual(pw.info(out)['barriers'], 1); self.assertEqual(pw.info(out)['shuffles'], 1)
        preds, then, bad = predicated_blocks(out)
        self.assertEqual(len(preds), 2, 'one predicate per function with side effects'); self.assertEqual(bad, [])
        self.assertFalse(pw.wants_pad(out), 'a padded module is not padded again')
        with self.assertRaises(ValueError): pw.pad(out)
        # the shuffle-xor variant: this kernel's index is (lane ^ 2) | base
        xo, n, total = pw.shuffle_xor(spv); self.assertEqual((n, total), (1, 1)); self.assertTrue(validate(xo)[0])
        self.assertEqual(pw.info(xo)['shuffles'], 1); self.assertIn(pw.OP_SHUFFLE_XOR, {x.op for x in pw.parse(xo)[1]})

    @unittest.skipUnless(SPIRV_AS and SPIRV_VAL, 'needs spirv-as and spirv-val')
    def test_tiny_select(self):   # branchless padding: the Workgroup stores and the image write take an OpSelect, only the storage stores and the atomic a branch
        spv = assemble(TINY); out, st = pw.pad(spv, mode='select')
        ok, err = validate(out); self.assertTrue(ok, err); self.assertEqual(local_size(out), [64, 2, 1])
        self.assertEqual(st['stores'], {'workgroup': 2, 'storage': 2}); self.assertEqual(st['atomics'], 1); self.assertEqual(st['image_writes'], 1); self.assertEqual(st['selects'], 3)
        def blocks(b): return sum(1 for x in pw.parse(b)[1] if x.op == pw.OP_LABEL)
        self.assertEqual(blocks(out) - blocks(spv), 2 * st['selections']); self.assertEqual(st['selections'], 3)   # the ssbo store, the atomic, the second ssbo store (an OpIAdd between the last two)
        ins = pw.parse(out)[1]
        arr = [x for x in ins if x.op == 28]; self.assertEqual(len(arr), 1); self.assertEqual(pw._Mod(out).const_value(arr[0].a[2]), 33, 'the shared array grew by one spare slot')
        for x in ins:   # every Workgroup store's access chain indexes through a select on the predicate
            if x.op == pw.OP_STORE and pw._Mod(out).storage_class(x.a[0]) == pw.SC_WORKGROUP:
                ch = next(y for y in ins if y.result == x.a[0]); sel = next(y for y in ins if y.result == ch.a[3]); self.assertEqual(sel.op, pw.OP_SELECT)
        self.assertEqual(pw.info(out)['barriers'], 1); self.assertEqual(pw.info(out)['shuffles'], 1)

    @unittest.skipUnless(SPIRV_AS and SPIRV_VAL, 'needs spirv-as and spirv-val')
    def test_bails(self):
        other = TINY.replace('LocalSize 32 2 1', 'LocalSize 16 2 1'); spv = assemble(other)
        self.assertFalse(pw.wants_pad(spv))
        with self.assertRaises(ValueError): pw.pad(spv)
        # LocalInvocationId read through a run-time index
        spv = assemble(TINY.replace('%xp = OpAccessChain %_ptr_Input_uint %lid %uint_0', '%xp = OpAccessChain %_ptr_Input_uint %lid %y'))
        with self.assertRaises(ValueError): pw.pad(spv)

    @unittest.skipUnless(SPIRV_VAL and SAMPLE.is_file(), 'needs spirv-val and DM_PADWAVE_SAMPLE/e6be82d835fbe4d3.spv')
    def test_ocarina_e6(self):
        spv = SAMPLE.read_bytes(); before = pw.info(spv)
        self.assertEqual(before['local'], [32, 1, 1]); self.assertEqual(before['barriers'], 362); self.assertEqual(before['shuffles'], 864); self.assertEqual(before['ballot_votes'], [])
        out, st = pw.pad(spv)
        ok, err = validate(out); self.assertTrue(ok, err)
        after = pw.info(out)
        self.assertEqual(after['local'], [64, 1, 1]); self.assertEqual(after['barriers'], 362); self.assertEqual(after['shuffles'], 864)
        self.assertEqual(st['stores'], {'workgroup': 276, 'bda': 1}); self.assertEqual(st['atomics'], 1); self.assertEqual(st['functions'], 3)
        preds, then, bad = predicated_blocks(out)
        self.assertEqual(len(preds), 3); self.assertEqual(bad, [])
        # bindings as before: the same OpDecorate Binding / DescriptorSet rows
        def binds(b): return sorted(tuple(x.a) for x in pw.parse(b)[1] if x.op == pw.OP_DECORATE and len(x.a) == 3 and x.a[1] in (33, 34))
        self.assertEqual(binds(spv), binds(out))
        xo, n, total = pw.shuffle_xor(spv); self.assertEqual(total, 864); self.assertEqual(n, 0, 'its shuffle indices are 2 * v | base, no lane ^ constant')
        out2, st2 = pw.pad(spv, mode='select'); ok, err = validate(out2); self.assertTrue(ok, err)
        self.assertEqual(st2['selects'], 276); self.assertEqual(st2['selections'], 2); self.assertEqual(pw.info(out2)['barriers'], 362); self.assertEqual(pw.info(out2)['shuffles'], 864)


if __name__ == '__main__':
    unittest.main()
