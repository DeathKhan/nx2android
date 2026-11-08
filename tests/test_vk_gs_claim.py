"""The production GS ownership transform preserves raster output or refuses the module."""
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = '''OpCapability Shader
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %position
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 1
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpDecorate %position BuiltIn Position
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %buffer DescriptorSet 2
OpDecorate %buffer Binding 0
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%v4 = OpTypeVector %float 4
%f0 = OpConstant %float 0
%v0 = OpConstantComposite %v4 %f0 %f0 %f0 %f0
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%true = OpConstantTrue %bool
%block = OpTypeStruct %uint
%pb = OpTypePointer StorageBuffer %block
%pu = OpTypePointer StorageBuffer %uint
%po = OpTypePointer Output %v4
%position = OpVariable %po Output
%buffer = OpVariable %pb StorageBuffer
%ft = OpTypeFunction %void
HELPERS
%main = OpFunction %void None %ft
%entry = OpLabel
%ptr = OpAccessChain %pu %buffer %u0
%old = OpAtomicIAdd %uint %ptr %u1 %u0 %u1
BODY
OpStore %position %v0
OpEmitVertex
OpStore %position %v0
OpEmitVertex
OpStore %position %v0
OpEmitVertex
OpEndPrimitive
OpReturn
OpFunctionEnd
'''
HARNESS = '''#include "gfx_vk_gs_claim.h"
#include <stdio.h>
#include <stdlib.h>
int main(int n,char**v){(void)n;FILE*f=fopen(v[1],"rb");fseek(f,0,SEEK_END);long z=ftell(f);rewind(f);uint32_t*w=malloc(z);fread(w,1,z,f);fclose(f);uint32_t nw,iv;char why[128];uint32_t*r=spv_gs_claim(w,z/4,2,95,&nw,&iv,why,sizeof why);puts(why);if(r){f=fopen(v[2],"wb");fwrite(r,4,nw,f);fclose(f);}int ok=r!=NULL;free(w);free(r);return ok?0:2;}
'''


@unittest.skipUnless(all(shutil.which(x) for x in ('cc', 'spirv-as', 'spirv-val')), 'C compiler and SPIR-V Tools required')
class GeometryClaims(unittest.TestCase):
    def test_valid_modules_and_conservative_refusals(self):
        scratch=ROOT/'target/tests';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='gs-claim-',dir=scratch) as tmp:
            p=Path(tmp);(p/'main.c').write_text(HARNESS)
            subprocess.run(['cc','-std=c11','-g','-O1','-fsanitize=address,undefined','-I'+str(ROOT/'nxnative/dm/android'),str(ROOT/'nxnative/dm/android/gfx_vk_gs_claim.c'),str(p/'main.c'),'-o',str(p/'run')],check=True)
            def run(name, source, accept, env='vulkan1.1'):
                with self.subTest(name=name):
                    src=p/(name+'.asm');inp=p/(name+'.spv');out=p/(name+'.out.spv');src.write_text(source)
                    subprocess.run(['spirv-as','--target-env',env,str(src),'-o',str(inp)],check=True,capture_output=True)
                    subprocess.run(['spirv-val','--target-env',env,str(inp)],check=True,capture_output=True)
                    r=subprocess.run([str(p/'run'),str(inp),str(out)],capture_output=True,text=True)
                    self.assertEqual(r.returncode,0 if accept else 2,r.stdout+r.stderr)
                    if accept:subprocess.run(['spirv-val','--target-env',env,str(out)],check=True,capture_output=True)
                    return inp
            base=BASE.replace('HELPERS','').replace('BODY','')
            inp=run('direct',base,True)
            run('reused_atomic_wrapper_signature',base.replace('%old = OpAtomicIAdd %uint %ptr %u1 %u0 %u1','%old = OpAtomicIAdd %uint %ptr %u1 %u0 %u1\n%old2 = OpAtomicIAdd %uint %ptr %u1 %u0 %u1'),True)
            # Output and branches may not depend on an omitted atomic result.
            tainted=BASE.replace('HELPERS','').replace('BODY','%f = OpConvertUToF %float %old\n%v = OpCompositeConstruct %v4 %f %f %f %f').replace('OpStore %position %v0','OpStore %position %v')
            run('atomic_output',tainted,False)
            branch=BASE.replace('HELPERS','').replace('BODY','%test = OpIEqual %bool %old %u0\nOpSelectionMerge %merge None\nOpBranchConditional %test %taken %merge\n%taken = OpLabel\nOpBranch %merge\n%merge = OpLabel')
            run('atomic_branch',branch,False)
            # A CAS retry loop must be bypassed as a whole, not assigned a fake CAS return on each iteration.
            helper='''%helper = OpFunction %void None %ft
%hentry = OpLabel
%hp = OpAccessChain %pu %buffer %u0
OpBranch %loop
%loop = OpLabel
OpLoopMerge %end %cont None
OpBranch %work
%work = OpLabel
%seen = OpLoad %uint %hp
%cas = OpAtomicCompareExchange %uint %hp %u1 %u0 %u0 %u1 %seen
%retry = OpINotEqual %bool %cas %seen
OpBranch %cont
%cont = OpLabel
OpBranchConditional %retry %loop %end
%end = OpLabel
OpReturn
OpFunctionEnd'''
            hbase=BASE.replace('HELPERS',helper).replace('BODY','%call = OpFunctionCall %void %helper')
            run('whole_cas_helper',hbase,True)
            run('helper_emits',hbase.replace('%hentry = OpLabel','%hentry = OpLabel\nOpEmitVertex'),False)
            writer='''%writer = OpFunction %void None %ft
%wentry = OpLabel
OpStore %position %v0
OpReturn
OpFunctionEnd
'''
            run('nested_output_helper',hbase.replace('HELPERS','').replace('%helper = OpFunction',writer+'%helper = OpFunction').replace('%hentry = OpLabel','%hentry = OpLabel\n%wc = OpFunctionCall %void %writer'),False)
            # Atomic-tainted data through a proven pointer copy must still taint the original storage.
            alias=base.replace('%ft = OpTypeFunction','%private = OpTypePointer Private %uint\n%memory = OpVariable %private Private\n%ft = OpTypeFunction')
            alias=alias.replace('OpStore %position %v0','%alias = OpCopyObject %private %memory\nOpStore %alias %old\n%loaded = OpLoad %uint %memory\n%fl = OpConvertUToF %float %loaded\n%vo = OpCompositeConstruct %v4 %fl %fl %fl %fl\nOpStore %position %vo',1)
            run('pointer_copy_taint',alias,False)
            run('volatile_store_taint',alias.replace('OpStore %alias %old','OpStore %alias %old Volatile'),False)
            run('volatile_output_taint',tainted.replace('OpStore %position %v','OpStore %position %v Volatile'),False)
            modf=base.replace('OpMemoryModel','%glsl = OpExtInstImport "GLSL.std.450"\nOpMemoryModel').replace('%ft = OpTypeFunction','%pf = OpTypePointer Private %float\n%fraction = OpVariable %pf Private\n%ft = OpTypeFunction').replace('%old = OpAtomicIAdd','%whole = OpExtInst %float %glsl Modf %f0 %fraction\n%old = OpAtomicIAdd')
            run('extinst_pointer_out',modf,False)
            loaded=tainted.replace('%f = OpConvertUToF %float %old','%readback = OpLoad %uint %ptr\n%f = OpConvertUToF %float %readback')
            run('atomic_memory_reaches_output',loaded,False)
            aliased=loaded.replace('OpDecorate %buffer Binding 0','OpDecorate %buffer Binding 0\nOpDecorate %other DescriptorSet 2\nOpDecorate %other Binding 1').replace('%buffer = OpVariable %pb StorageBuffer','%buffer = OpVariable %pb StorageBuffer\n%other = OpVariable %pb StorageBuffer').replace('%readback = OpLoad %uint %ptr','%other_ptr = OpAccessChain %pu %other %u0\n%readback = OpLoad %uint %other_ptr')
            run('potential_storage_binding_alias',aliased,False)
            signed=base.replace('%position\nOpExecutionMode','%position %primitive\nOpExecutionMode',1).replace('OpDecorate %position','OpDecorate %primitive BuiltIn PrimitiveId\nOpDecorate %position',1).replace('%ft = OpTypeFunction','%pi = OpTypePointer Input %int\n%primitive = OpVariable %pi Input\n%ft = OpTypeFunction')
            run('signed_primitive_id',signed,True)
            output_id=base.replace('%position\nOpExecutionMode','%position %primitive_out\nOpExecutionMode',1).replace('OpDecorate %position','OpDecorate %primitive_out BuiltIn PrimitiveId\nOpDecorate %position',1).replace('%ft = OpTypeFunction','%pout = OpTypePointer Output %uint\n%primitive_out = OpVariable %pout Output\n%ft = OpTypeFunction')
            run('output_id_is_not_input',output_id,True)
            selected=base.replace('OpCapability Geometry','OpCapability Geometry\nOpCapability VariablePointersStorageBuffer\nOpExtension "SPV_KHR_variable_pointers"').replace('%old = OpAtomicIAdd','%alias = OpSelect %pu %true %ptr %ptr\n%old = OpAtomicIAdd')
            run('unproven_pointer_select',selected,False)
            run('binding_collision',base.replace('Binding 0','Binding 95'),False)
            run('multiple_invocations',base.replace('Invocations 1','Invocations 2'),False)
            run('no_effects',base.replace('%old = OpAtomicIAdd %uint %ptr %u1 %u0 %u1',''),False)
            # SPIR-V1.4+ entry interfaces include all newly used global variables.
            v14=base.replace('%position\nOpExecutionMode','%position %buffer\nOpExecutionMode',1)
            run('all_globals_interface',v14,True,'vulkan1.2')
            words=list(struct.unpack('<%dI'%(inp.stat().st_size//4),inp.read_bytes()))
            at=5
            while (words[at]&65535)!=54:at+=words[at]>>16
            bad=list(words);bad[at+2]=words[3]+10
            malformed=p/'bad.spv';malformed.write_bytes(struct.pack('<%dI'%len(bad),*bad))
            r=subprocess.run([str(p/'run'),str(malformed),str(p/'bad.out')],capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stdout+r.stderr)
            bad=list(words);bad[at]=(1<<16)|54
            malformed.write_bytes(struct.pack('<%dI'%len(bad),*bad))
            r=subprocess.run([str(p/'run'),str(malformed),str(p/'bad.out')],capture_output=True,text=True)
            self.assertEqual(r.returncode,2,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
