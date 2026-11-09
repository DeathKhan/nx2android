#!/usr/bin/env python3
"""Actual offline observer tests; artifacts remain in project dist/tests."""
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import unittest
import flat_carrier_observer as observer
import spv_intermediate as spv

OUT = Path(__file__).resolve().parents[1] / "dist/tests/flat-carrier-observer"
OUT.mkdir(parents=True, exist_ok=True)

VERTEX = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %vertex %position %out0
OpDecorate %vertex BuiltIn VertexIndex
OpDecorate %position BuiltIn Position
OpDecorate %out0 Location 0
OpDecorate %words ArrayStride 4
OpDecorate %block BufferBlock
OpMemberDecorate %block 0 Offset 0
OpDecorate %buffer DescriptorSet 1
OpDecorate %buffer Binding 0
%void = OpTypeVoid
%u = OpTypeInt 32 0
%i = OpTypeInt 32 1
%f = OpTypeFloat 32
%v4 = OpTypeVector %f 4
%fn = OpTypeFunction %void
%words = OpTypeRuntimeArray %u
%block = OpTypeStruct %words
%pb = OpTypePointer Uniform %block
%pu = OpTypePointer Uniform %u
%pi = OpTypePointer Input %i
%po = OpTypePointer Output %v4
%buffer = OpVariable %pb Uniform
%vertex = OpVariable %pi Input
%position = OpVariable %po Output
%out0 = OpVariable %po Output
%z = OpConstant %u 0
%word4 = OpConstant %u 4
%stride = OpConstant %u 21
%fz = OpConstant %f 0
%fo = OpConstant %f 1
%pos = OpConstantComposite %v4 %fz %fz %fz %fo
%main = OpFunction %void None %fn
%label = OpLabel
%vi = OpLoad %i %vertex
%vu = OpBitcast %u %vi
%base = OpIMul %u %vu %stride
%index = OpIAdd %u %base %word4
%pointer = OpAccessChain %pu %buffer %z %index
%raw = OpLoad %u %pointer
%bits = OpBitcast %f %raw
%result = OpCompositeConstruct %v4 %bits %fz %fz %fo
OpStore %out0 %result
OpStore %position %pos
OpReturn
OpFunctionEnd
"""

FRAGMENT = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input0 %input1 %output0
OpExecutionMode %main OriginUpperLeft
OpDecorate %input0 Location 0
OpDecorate %input0 Flat
OpDecorate %input1 Location 1
OpDecorate %output0 Location 0
%void = OpTypeVoid
%u = OpTypeInt 32 0
%i = OpTypeInt 32 1
%f = OpTypeFloat 32
%v4 = OpTypeVector %f 4
%fn = OpTypeFunction %void
%pi4 = OpTypePointer Input %v4
%pif = OpTypePointer Input %f
%pof = OpTypePointer Output %v4
%plf = OpTypePointer Function %f
%input0 = OpVariable %pi4 Input
%input1 = OpVariable %pi4 Input
%output0 = OpVariable %pof Output
%z = OpConstant %u 0
%fz = OpConstant %f 0
%fo = OpConstant %f 1
%main = OpFunction %void None %fn
%label = OpLabel
%local = OpVariable %plf Function
%pointer = OpAccessChain %pif %input0 %z
%immediate = OpLoad %f %pointer
OpStore %local %immediate
%first = OpLoad %f %local
%recovered = OpBitcast %i %first
%second = OpLoad %f %local
%again = OpBitcast %i %second
%asfloat = OpConvertSToF %f %recovered
%uv = OpLoad %v4 %input1
%x = OpCompositeExtract %f %uv 0
%derivative = OpDPdx %f %x
%color = OpCompositeConstruct %v4 %asfloat %derivative %fz %fo
OpStore %output0 %color
OpReturn
OpFunctionEnd
"""

def sha(data):
    return hashlib.sha256(data).hexdigest()

def assemble(name, text, version="spv1.3"):
    asm, path = OUT/(name+".spvasm"), OUT/(name+".spv")
    asm.write_text(text)
    subprocess.run(["spirv-as", "--target-env", version, str(asm), "-o", str(path)],
                   check=True, capture_output=True)
    return path.read_bytes()

def validate(name, data, environment="vulkan1.1"):
    path = OUT/(name+".spv")
    path.write_bytes(data)
    subprocess.run(["spirv-val", "--target-env", environment, str(path)], check=True, capture_output=True)

def execute(data, original_input, sideband, local_override=None):
    """Raw32 interpreter for the exercised straight-line original+inserted code."""
    header, instructions = spv.decode(data)
    values, memory, locations = {}, {}, {}
    uint_types, float_types = set(), set()
    main = next(a[1] for op,a in instructions if op == 15)
    for op,a in instructions:
        if op == 21 and a[1:] == [32,0]:uint_types.add(a[0])
        if op == 22 and a[1:] == [32]:float_types.add(a[0])
        if op == 71 and a[1] == 30:locations[a[0]]=a[2]
        if op == 43:values[a[1]]=a[2]
        if op == 44:values[a[1]]=tuple(values[v] for v in a[2:])
        if op == 59:
            values[a[1]]=(a[1],())
            memory[a[1]]=None
            if a[2] == 1:
                loc=locations[a[1]]
                memory[a[1]]=sideband if loc==12 else (original_input,0,0,0x3f800000) if loc==0 else (0,0,0,0)
    function=None
    for op,a in instructions:
        if op==54:function=a[1];continue
        if op==56:function=None;continue
        if function!=main:continue
        if op==59:values[a[1]]=(a[1],());memory[a[1]]=None
        elif op==65:
            root,path=values[a[2]];values[a[1]]=(root,path+tuple(values[v] for v in a[3:]))
        elif op==61:
            root,path=values[a[2]];value=memory[root]
            for index in path:value=value[index]
            values[a[1]]=value
        elif op==62:
            root,path=values[a[0]]
            if path:raise AssertionError("fixture uses only scalar/whole stores")
            value=values[a[1]]
            variable=next(v for o,v in instructions if o==59 and v[1]==root)
            if variable[2]==7 and local_override is not None:value=local_override
            memory[root]=value
        elif op==124:values[a[1]]=values[a[2]]
        elif op==80:values[a[1]]=tuple(values[v] for v in a[2:])
        elif op==81:values[a[1]]=values[a[2]][a[3]]
        elif op==111:
            v=values[a[2]];v=v-(1<<32) if v&(1<<31) else v
            values[a[1]]=struct.unpack("<I",struct.pack("<f",float(v)))[0]
        elif op==207:values[a[1]]=0  # uniform fixture: existing OpDPdx
        elif op in (248,253,8,317):pass
        else:raise AssertionError("unimplemented fixture opcode"+str(op))
    return {locations[a[1]]:memory[a[1]] for op,a in instructions if op==59 and a[2]==3}


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vertex=assemble("vertex",VERTEX)
        cls.fragment=assemble("fragment",FRAGMENT)
        cls.v,cls.f,cls.meta=observer.instrument_pair(cls.vertex,cls.fragment,sha(cls.vertex),sha(cls.fragment))

    def test_validator_and_exact_restoration(self):
        for name,source,output in (("vertex",self.vertex,self.v),("fragment",self.fragment,self.f)):
            validate(name+"-observer",output)
            self.assertEqual(observer.restore(output,self.meta[name]["proof"]),source)

    def test_payloads_distinguish_interface_and_local_loss(self):
        cases=[0,1,2,0xffff,0x10000,0x800000,0x7fc12345,0xffffffff,0x3f800000]
        for payload in cases:
            for immediate,local in [(payload,None),(0,None),(payload,0)]:
                original=execute(self.fragment,immediate,payload,local)
                observed=execute(self.f,immediate,payload,local)
                self.assertEqual(observed[0],original[0])
                self.assertEqual(observed[1],(payload,immediate,immediate if local is None else local,observer.MARKER))

    def test_no_original_output_write_or_derivative_motion(self):
        for stage,data in (("vertex",self.v),("fragment",self.f)):
            _,ins=spv.decode(data);proof=self.meta[stage]["proof"]
            added=[ins[i] for i in proof["inserted_instruction_indices"]]
            self.assertFalse(any(op==62 and a[0]==self.meta[stage]["original_output_id"] for op,a in added))
            self.assertTrue(all(op in (21,23,32,43,44,59,61,62,71,80,124) for op,a in added))
            self.assertFalse(any(op in (87,88,207,208,209,111,112,129,133) for op,a in added))
        _,old=spv.decode(self.fragment);_,new=spv.decode(self.f)
        self.assertEqual([x for x in old if 207<=x[0]<=215],[x for x in new if 207<=x[0]<=215])

    def test_hash_and_shape_rejections(self):
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
            observer.instrument_pair(self.vertex,self.fragment,"0"*64,sha(self.fragment))
        for name,text,which in [
            ("notflat",FRAGMENT.replace("OpDecorate %input0 Flat",""),"fragment"),
            ("occupied12",FRAGMENT.replace("Location 1","Location 12"),"fragment"),
            ("stride22",VERTEX.replace("OpConstant %u 21","OpConstant %u 22"),"vertex"),
            ("word3",VERTEX.replace("OpConstant %u 4","OpConstant %u 3"),"vertex"),
            ("extra_local_write",FRAGMENT.replace("OpStore %local %immediate","OpStore %local %immediate\nOpStore %local %fz"),"fragment"),
            ("numeric_use",FRAGMENT.replace("%again = OpBitcast %i %second","%again = OpConvertFToS %i %second"),"fragment")]:
            data=assemble(name,text)
            v=data if which=="vertex" else self.vertex
            f=data if which=="fragment" else self.fragment
            with self.assertRaises(ValueError,msg=name):
                observer.instrument_pair(v,f,sha(v),sha(f))

    def test_extinst_literal_is_not_input_id(self):
        text=FRAGMENT.replace('%input0','%81').replace('OpMemoryModel Logical GLSL450',
            '%ext = OpExtInstImport "GLSL.std.450"\nOpMemoryModel Logical GLSL450')
        text=text.replace('%uv = OpLoad',
            '%nc = OpExtInst %f %ext NClamp %asfloat %fz %fo\n%uv = OpLoad')
        asm=OUT/'literal-collision.spvasm';path=OUT/'literal-collision.spv'
        asm.write_text(text)
        subprocess.run(['spirv-as','--preserve-numeric-ids','--target-env','spv1.3',str(asm),'-o',str(path)],check=True)
        data=path.read_bytes()
        _,observed,meta=observer.instrument_pair(self.vertex,data,sha(self.vertex),sha(data))
        self.assertEqual(meta['fragment']['input_id'],81)
        validate('literal-collision-observer',observed)
        self.assertEqual(observer.restore(observed,meta['fragment']['proof']),data)
        # A real use of the loaded input by ExtInst must still decline.
        bad=text.replace('NClamp %asfloat','NClamp %immediate')
        asm.write_text(bad)
        subprocess.run(['spirv-as','--preserve-numeric-ids','--target-env','spv1.3',str(asm),'-o',str(path)],check=True)
        data=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'immediate input has other uses'):
            observer.instrument_pair(self.vertex,data,sha(self.vertex),sha(data))

    def test_spirv14_private_interfaces(self):
        v=assemble("vertex14",VERTEX.replace("BufferBlock","Block").replace("Uniform","StorageBuffer").replace("%position %out0\n","%position %out0 %buffer\n"),"spv1.4")
        f=assemble("fragment14",FRAGMENT,"spv1.4")
        vo,fo,meta=observer.instrument_pair(v,f,sha(v),sha(f))
        validate("vertex14-observer",vo,"vulkan1.2")
        validate("fragment14-observer",fo,"vulkan1.2")
        self.assertEqual(observer.restore(fo,meta["fragment"]["proof"]),f)


if __name__=="__main__":
    unittest.main()

