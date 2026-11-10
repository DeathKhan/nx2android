"""Integer storage views must select the existing raw-word SPIR-V variant."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cc   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASM = """
OpCapability Shader
OpCapability StorageImageWriteWithoutFormat
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %img DescriptorSet 0
OpDecorate %img Binding 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%i32 = OpTypeInt 32 1
%f32 = OpTypeFloat 32
%v2i = OpTypeVector %i32 2
%v4f = OpTypeVector %f32 4
%image = OpTypeImage %f32 2D 0 0 0 2 Unknown
%ptr = OpTypePointer UniformConstant %image
%img = OpVariable %ptr UniformConstant
%zeroi = OpConstant %i32 0
%zero = OpConstant %f32 0
%packed = OpConstant %u32 2140833954
%coord = OpConstantComposite %v2i %zeroi %zeroi
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %image %img
%bits = OpBitcast %f32 %packed
%texel = OpCompositeConstruct %v4f %bits %zero %zero %zero
OpImageWrite %loaded %coord %texel
OpReturn
OpFunctionEnd
"""
C_PREFIX = """
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
"""
C_MAIN = r"""
int main(int argc,char **argv) {
 assert(argc==4);FILE *f=fopen(argv[1],"rb");assert(f);assert(!fseek(f,0,SEEK_END));long bytes=ftell(f);rewind(f);
 assert(bytes>=20&&bytes%4==0);uint32_t *w=malloc(bytes);assert(fread(w,1,bytes,f)==(size_t)bytes);fclose(f);
 uint16_t binding=3;uint8_t kind=(uint8_t)atoi(argv[3]);uint32_t n=0;
 uint32_t *out=spv_int_images(w,(uint32_t)bytes/4,&binding,&kind,1,0,&n);assert(out&&n>(uint32_t)bytes/4);
 f=fopen(argv[2],"wb");assert(f);assert(fwrite(out,4,n,f)==n);fclose(f);free(w);free(out);return 0;
}
"""


class StorageRetype(unittest.TestCase):
    @unittest.skipUnless(all(shutil.which(t) for t in ("cc", "spirv-as", "spirv-dis", "spirv-val")), "SPIR-V tools and C compiler required")
    def test_actual_image_write_retype(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_shader.c").read_text(encoding='utf-8')
        start = source.index("typedef struct { const uint32_t *w; uint32_t fn, bound,")
        end = source.index("// A variant of a fragment shader that writes", start)
        scratch = ROOT / "target/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="storage-retype-", dir=scratch) as folder:
            p = Path(folder)
            (p / "check.c").write_text(C_PREFIX + source[start:end] + C_MAIN, encoding='utf-8')
            subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(p / "check.c"), "-o", str(p / "check")], check=True)
            (p / "input.spvasm").write_text(ASM, encoding='utf-8')
            subprocess.run(["spirv-as", "--target-env", "vulkan1.1", str(p / "input.spvasm"), "-o", str(p / "input.spv")], check=True)
            for kind in (1, 2):
                output = p / f"out{kind}.spv"
                subprocess.run([str(p / "check"), str(p / "input.spv"), str(output), str(kind)], check=True)
                subprocess.run(["spirv-val", "--target-env", "vulkan1.1", str(output)], check=True)
                text = subprocess.check_output(["spirv-dis", str(output)], text=True)
                # The actual OpImageWrite texel must be an integer vec4 bitcast, not a numeric float conversion.
                lines = dict(re.findall(r"^\s*(%\w+) = (.+)$", text, re.M))
                write = re.search(r"OpImageWrite %\w+ %\w+ (%\w+)", text).group(1)
                cast = lines[write].split()
                self.assertEqual(cast[0], "OpBitcast")
                vector = lines[cast[1]].split()
                self.assertEqual(vector[0], "OpTypeVector")
                self.assertEqual(vector[-1], "4")
                self.assertEqual(lines[vector[1]], f"OpTypeInt 32 {kind - 1}")
                self.assertNotIn("OpConvertFTo", text)

    @unittest.skipUnless(shutil.which("cc"), "C compiler required")
    @unittest.skipUnless(host_cc.sanitizers(), 'a host C compiler with ASan and UBSan is required (MinGW has neither)')
    def test_actual_descriptor_branch_records_storage_type(self):
        source = (ROOT / "nxnative/dm/android/gfx_vk_exec.c").read_text(encoding='utf-8')
        start = source.index("        case RK_IMG: case RK_IMG_UBO: {", source.index("static VkDescriptorSet stage_set("))
        end = source.index("        case RK_TBUF:", start)
        branch = source[start:end]
        code = C_PREFIX + r"""
typedef struct { int kind,reg,off,binding; } dmsv_bind;
enum {RK_IMG=1,RK_IMG_UBO=2,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE=3};
typedef struct { int dummy; } VkDescriptorImageInfo;
typedef struct { void *pImageInfo;int descriptorType; } Write;
static struct {uint64_t img[6][32];} S;
static uint64_t g_num_um,g_num_im;
static int numeric;
static uint64_t ubo_word(int st,int reg,int off){assert(st==5&&reg==4&&off==8);return 77;}
static uint64_t mem_handle(uint64_t x){return x;}
static int tex_desc_for(int st,const dmsv_bind *b,uint64_t h,VkDescriptorImageInfo *ii,int storage){
 assert(st==5&&b->binding==23&&h==77&&ii&&storage==1);return numeric;
}
static void run(int kind,int num,int index){
 dmsv_bind bind={kind,4,8,23};const dmsv_bind *b=&bind;int st=5,nii=0,nw=0;VkDescriptorImageInfo ii[1];Write w={0},*wr=&w;
 numeric=num;g_num_um=g_num_im=0;S.img[5][4]=77;
 for(int i=index;i==index;i++){switch(kind){
""" + branch + r"""
 default:assert(0);}}
 assert(nii==1&&nw==1&&w.pImageInfo==&ii[0]&&w.descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
 uint64_t bit=index<64?1ull<<index:0;
 assert(g_num_um==(num==1?bit:0)&&g_num_im==(num==2?bit:0));
}
int main(void){for(int k=1;k<=2;k++)for(int n=0;n<=2;n++){run(k,n,3);run(k,n,64);}return 0;}
"""
        scratch = ROOT / "target/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="storage-mask-", dir=scratch) as folder:
            p = Path(folder)
            (p / "check.c").write_text(code, encoding='utf-8')
            subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined", str(p / "check.c"), "-o", str(p / "check")], check=True)
            subprocess.run([str(p / "check")], check=True)


if __name__ == "__main__":
    unittest.main()
