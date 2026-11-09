#!/usr/bin/env python3
"""Actual-transformer tests; artifacts stay under project dist/tests."""
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import unittest

os.environ["PYTHONDONTWRITEBYTECODE"]="1"
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("probe",HERE/"spv_intermediate.py")
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
OUT=HERE.parent/"dist/tests/spv-intermediate"
OUT.mkdir(parents=True,exist_ok=True)

PREFIX="""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main" %2 %3
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 Location 0
OpDecorate %3 Location 13
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeFloat 32
%7 = OpTypeVector %6 2
%8 = OpTypeVector %6 3
%9 = OpTypeVector %6 4
%10 = OpTypePointer Output %9
%11 = OpTypePointer Input %9
%12 = OpTypeBool
%13 = OpTypeInt 32 0
%14 = OpConstant %6 0
%15 = OpConstant %6 1
%16 = OpConstant %13 0
%17 = OpTypePointer Input %6
%18 = OpConstantComposite %9 %15 %15 %15 %15
%19 = OpTypeFunction %6
%2 = OpVariable %10 Output
%3 = OpVariable %11 Input
"""
BRANCH="""%1 = OpFunction %4 None %5
%20 = OpLabel
%21 = OpAccessChain %17 %3 %16
%22 = OpLoad %6 %21
%23 = OpFOrdGreaterThan %12 %22 %14
OpSelectionMerge %30 None
OpBranchConditional %23 %25 %26
%25 = OpLabel
%27 = OpFAdd %6 %15 %15
OpBranch %30
%26 = OpLabel
OpBranch %30
%30 = OpLabel
%31 = OpPhi %6 %27 %25 %14 %26
%32 = OpPhi %6 %15 %25 %14 %26
%33 = OpCompositeConstruct %7 %31 %32
%34 = OpCompositeConstruct %8 %31 %32 %15
%35 = OpCompositeConstruct %9 %31 %32 %15 %15
OpStore %2 %18
OpReturn
OpFunctionEnd
"""

def assemble(name,text,version="spv1.3"):
    asm=OUT/(name+".spvasm");spv=OUT/(name+".spv")
    asm.write_text(text)
    subprocess.run(["spirv-as","--target-env",version,"--preserve-numeric-ids",str(asm),"-o",str(spv)],check=True,capture_output=True)
    return spv.read_bytes()

def valid(name,data):
    p=OUT/(name+".spv");p.write_bytes(data)
    # Vulkan1.2 accepts SPIR-V1.4 as well as1.3.
    result=probe.validate(p,"vulkan1.2")
    if result["exit_code"]:raise AssertionError(result["message"])

def run(data,input_value):
    """Evaluate the exercised SPIR-V operations, including branch/Phi/memory."""
    _,ins=probe.decode(data);values={};memory={3:input_value};blocks={};funcs={};fn=None;block=None
    for op,a in ins:
        if op==43:
            values[a[1]]=struct.unpack("<f",struct.pack("<I",a[2]))[0] if a[0]==6 else a[2]
        elif op==44:values[a[1]]=tuple(values[x] for x in a[2:])
        elif op==59:values[a[1]]=(a[1],());memory.setdefault(a[1],None)
        elif op==54:fn=a[1];funcs[fn]=[]
        elif op==248:block=a[0];blocks[block]=[];funcs[fn].append(block)
        elif op==56:fn=None;block=None
        elif fn is not None and block is not None:blocks[block].append((op,a))
    def read(ptr):
        root,path=ptr;v=memory[root]
        for i in path:v=v[i]
        return v
    def write(ptr,v):
        root,path=ptr
        if path:raise AssertionError("fixture interpreter only whole-variable writes")
        memory[root]=v
    def execute(fid):
        block=funcs[fid][0];prev=None
        while True:
            seq=blocks[block];phis={}
            for op,a in seq:
                if op!=245:break
                choices=dict(zip(a[3::2],a[2::2]));phis[a[1]]=values[choices[prev]]
            values.update(phis)
            for op,a in seq:
                if op in (245,247,8,317):continue
                if op==59:values[a[1]]=(a[1],());memory[a[1]]=None
                elif op==65:
                    root,path=values[a[2]];values[a[1]]=(root,path+tuple(values[i] for i in a[3:]))
                elif op==61:values[a[1]]=read(values[a[2]])
                elif op==62:write(values[a[0]],values[a[1]])
                elif op==80:values[a[1]]=tuple(values[i] for i in a[2:])
                elif op==81:
                    v=values[a[2]]
                    for i in a[3:]:v=v[i]
                    values[a[1]]=v
                elif op==129:values[a[1]]=values[a[2]]+values[a[3]]
                elif op==186:values[a[1]]=values[a[2]]>values[a[3]]
                elif op==57:values[a[1]]=execute(a[2])
                elif op in (249,250):
                    nxt=a[0] if op==249 else (a[1] if values[a[0]] else a[2])
                    prev,block=block,nxt;break
                elif op==253:return None
                elif op==254:return values[a[0]]
                elif op==252:return "discard"
                else:raise AssertionError((op,a))
            else:raise AssertionError("unterminated fixture")
    ret=execute(1)
    return ret,memory.get(2)

class ProbeTests(unittest.TestCase):
    def test_branch_values_phi_and_vector_padding(self):
        src=assemble("branch",PREFIX+BRANCH);valid("branch-baseline",src)
        for target,n in [(27,1),(31,1),(33,2),(34,3),(35,4)]:
            out,m=probe.instrument(src,target=target);valid("branch-"+str(target),out)
            self.assertEqual(m["main_returns_overridden"],1)
            for x in (-1.,1.):
                _,got=run(out,(x,0.,0.,1.))
                if target==27 and x<0:want=(0.,0.,0.,0.)
                else:
                    vals=([2. if x>0 else 0.] if n==1 else
                          [2. if x>0 else 0.,1. if x>0 else 0.,1.,1.][:n])
                    want=tuple(vals+[0.]*(3-len(vals))+[1.]) if n<4 else tuple(vals)
                self.assertEqual(got,want,(target,x))
        # Original instructions are an exact subsequence (IDs/order unchanged).
        original=probe.decode(src)[1];rewritten=probe.decode(out)[1];j=0
        for inst in rewritten:
            if j<len(original) and inst==original[j]:j+=1
        self.assertEqual(j,len(original))

    def test_input_selects_only_requested_components_and_interfaces(self):
        src=assemble("input",PREFIX+BRANCH)
        for components in ("x","xy","xyz","xyzw"):
            out,m=probe.instrument(src,input_location=13,components=components)
            valid("input-"+components,out)
            v=(.25,.5,.75,1.)
            _,got=run(out,v);n=len(components)
            self.assertEqual(got,v[:n]+(0.,)*(3-n)+(1.,) if n<4 else v)
            self.assertEqual(probe.module_info(probe.decode(src)[1])[4],
                             probe.module_info(probe.decode(out)[1])[4])
            # No whole-vector load from the input: Z/W aren't touched for XY.
            self.assertFalse(any(op==61 and a[2]==3 for op,a in probe.decode(out)[1]))
        src14=assemble("input14",PREFIX+BRANCH,"spv1.4")
        out,m=probe.instrument(src14,input_location=13)
        valid("input14-probe",out)
        self.assertIn(m["private_debug_id"],probe.module_info(probe.decode(out)[1])[4][0][3])

    def test_helper_result_multiple_returns_and_discard(self):
        body=BRANCH[:BRANCH.index("OpSelectionMerge")]+"""OpSelectionMerge %30 None
OpBranchConditional %23 %25 %26
%25 = OpLabel
%27 = OpFunctionCall %6 %40
OpStore %2 %18
OpReturn
%26 = OpLabel
OpStore %2 %18
OpReturn
%30 = OpLabel
OpUnreachable
OpFunctionEnd
%40 = OpFunction %6 None %19
%41 = OpLabel
%42 = OpFAdd %6 %15 %15
OpReturnValue %42
OpFunctionEnd
"""
        src=assemble("returns",PREFIX+body);valid("returns-base",src)
        out,m=probe.instrument(src,target=42);valid("returns-probe",out)
        self.assertEqual(m["main_returns_overridden"],2)
        self.assertEqual(run(out,(1.,0.,0.,1.))[1],(2.,0.,0.,1.))
        self.assertEqual(run(out,(-1.,0.,0.,1.))[1],(0.,0.,0.,0.))
        discard=body.replace("%26 = OpLabel\nOpStore %2 %18\nOpReturn","%26 = OpLabel\nOpKill")
        src=assemble("discard",PREFIX+discard)
        out,_=probe.instrument(src,target=42);valid("discard-probe",out)
        self.assertEqual(run(out,(-1.,0.,0.,1.))[0],"discard")
        self.assertEqual(sum(op==252 for op,a in probe.decode(out)[1]),1)

    def test_rejections_and_validation_comparison(self):
        src=assemble("reject",PREFIX+BRANCH)
        for args in ({},{"target":9999},{"target":2},{"target":23},{"target":27,"input_location":13},
                     {"input_location":12},{"input_location":13,"components":"q"}):
            with self.assertRaises(ValueError,msg=str(args)):probe.instrument(src,**args)
        self.assertEqual(probe.encode(*probe.decode(src)),src)

if __name__=="__main__":unittest.main()

