#!/usr/bin/env python3
"""Run the real loader preamble guard under ASan/UBSan, with real SPIR-V probes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("fixtures",HERE/"test_spv_intermediate.py")
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
p=f.probe
OUT=f.OUT

RESOURCE_PREFIX=f.PREFIX.replace("OpDecorate %3 Location 13\n", """OpDecorate %3 Location 13
OpDecorate %50 Block
OpMemberDecorate %50 0 Offset 0
OpDecorate %52 DescriptorSet 2
OpDecorate %52 Binding 10
""")+"""%50 = OpTypeStruct %6
%51 = OpTypePointer Uniform %50
%52 = OpVariable %51 Uniform
"""

class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src=(HERE.parent/"nxnative/dm/android/gfx_vk_shader.c").read_text()
        start=src.index("static int spv_runtime_extra(")
        end=src.index("// Separate, opt-in paired observer contract.",start)
        helper=src[start:end]
        harness=r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
""" + helper + r"""
static uint32_t *readfile(const char *path, uint32_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) exit(3);
    fseek(f,0,SEEK_END); long size=ftell(f); rewind(f);
    if (size<0 || size%4 || size>(64<<20)) exit(3);
    uint32_t *p=malloc((size_t)size+4); if (!p) exit(3);
    if (fread(p,1,(size_t)size,f)!=(size_t)size) exit(3);
    fclose(f); *n=(uint32_t)size/4; return p;
}
int main(int argc,char **argv) {
    if(argc!=4) return 3;
    uint32_t an,bn;uint32_t *a=readfile(argv[1],&an),*b=readfile(argv[2],&bn);
    int r=spv_runtime_contract(a,an,b,bn), want=atoi(argv[3]);
    free(a);free(b);if(r!=want){fprintf(stderr,"contract=%d expected=%d\n",r,want);return 1;}
    return 0;
}
"""
        c=OUT/"runtime-contract.c";c.write_text(harness)
        cls.exe=OUT/"runtime-contract"
        subprocess.run(["cc","-std=gnu11","-Wall","-Wextra","-Werror",
                        "-fsanitize=address,undefined","-fno-omit-frame-pointer",
                        str(c),"-o",str(cls.exe)],check=True)
        cls.count=0

    def check(self,base,alt,want):
        type(self).count+=1;name=str(type(self).count)
        b=OUT/("contract-base-"+name+".spv");a=OUT/("contract-alt-"+name+".spv")
        b.write_bytes(base);a.write_bytes(alt)
        subprocess.run([str(self.exe),str(b),str(a),str(int(want))],check=True,
                       env={**os.environ,"ASAN_OPTIONS":"detect_leaks=0"})

    def test_actual_probes_and_14_private_interface(self):
        for ver in ("spv1.3","spv1.4"):
            base=f.assemble("contract-"+ver,RESOURCE_PREFIX+f.BRANCH,ver)
            self.check(base,base,True)
            for args in ({"target":27},{"target":31},{"target":33},{"target":35},
                         {"input_location":13,"components":"xy"}):
                out,_=p.instrument(base,**args)
                f.valid("contract-validated-"+ver+"-"+str(len(out)),out)
                self.check(base,out,True)
        # Same guard against the actual extracted current bundle when available.
        actual=HERE.parent/"dist/cca-intermediate-probes/origin/original.spv"
        if actual.exists():
            base=actual.read_bytes()
            for args in ({"target":1484},{"target":1387},{"target":2312},{"input_location":13}):
                out,_=p.instrument(base,**args);self.check(base,out,True)

    def test_wrong_layout_capability_length_and_malformed(self):
        base=f.assemble("contract-mutations",RESOURCE_PREFIX+f.BRANCH)
        good,_=p.instrument(base,target=27)
        for decoration in (33,34):
            h,ins=p.decode(good)
            for i,(op,a) in enumerate(ins):
                if op==71 and a[:2]==[52,decoration]:
                    ins[i]=(op,a[:2]+[a[2]+1]);break
            self.check(base,p.encode(h,ins),False)
        h,ins=p.decode(good)
        ins=[x for x in ins if x!=(71,[50,2])]
        self.check(base,p.encode(h,ins),False) # lost Block
        h,ins=p.decode(good)
        at=next(i for i,(op,a) in enumerate(ins) if op==54)
        # A new resource variable is prohibited even with a fresh result ID.
        ins.insert(at,(59,[51,h[3],2]));h[3]+=1
        self.check(base,p.encode(h,ins),False)
        h,ins=p.decode(good)
        ins.insert(0,(17,[10])) # unsupported/new Float64 capability
        self.check(base,p.encode(h,ins),False)
        h,ins=p.decode(good);h[3]=p.decode(base)[0][3]-1
        self.check(base,p.encode(h,ins),False)
        self.check(base,good+b"\0\0\0\0",False)
        self.check(base,good[:-4],False)
        self.check(base,b"\0"*16,False)

    def test_private_only_interface_and_original_id_immutability(self):
        base=f.assemble("contract-interface",RESOURCE_PREFIX+f.BRANCH,"spv1.4")
        good,meta=p.instrument(base,target=27)
        h,ins=p.decode(good)
        for i,(op,a) in enumerate(ins):
            if op==15:ins[i]=(op,a[:-1]+[52]);break
        self.check(base,p.encode(h,ins),False) # appended Uniform, not Private
        h,ins=p.decode(good)
        for i,(op,a) in enumerate(ins):
            if op==59 and a[1]==meta["private_debug_id"]:
                ins[i]=(op,[a[0],52,a[2]]);break
        self.check(base,p.encode(h,ins),False) # reuses original descriptor ID

if __name__=="__main__":unittest.main()

