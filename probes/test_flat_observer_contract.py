#!/usr/bin/env python3
"""Exercise the actual isolated paired-IO guard with generated diagnostics."""
import os
from pathlib import Path
import subprocess
import unittest
import flat_carrier_observer as observer
import test_flat_carrier_observer as fixtures
import spv_intermediate as spv

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'dist/tests/flat-observer-contract'
OUT.mkdir(parents=True,exist_ok=True)

class ObserverContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=(HERE.parent/'nxnative/dm/android/gfx_vk_shader.c').read_text()
        first=source.index('static int spv_runtime_extra(')
        last=source.index('int vk_shader_observer_pair(',first)
        helper=source[first:last]
        harness=r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
'''+helper+r'''
static uint32_t *readfile(const char *path,uint32_t *n){
 FILE*f=fopen(path,"rb");if(!f)exit(3);fseek(f,0,SEEK_END);long z=ftell(f);rewind(f);
 if(z<0||z%4||z>(64<<20))exit(3);uint32_t*p=malloc((size_t)z+4);if(!p)exit(3);
 if(fread(p,1,(size_t)z,f)!=(size_t)z)exit(3);fclose(f);*n=(uint32_t)z/4;return p;
}
int main(int argc,char**argv){if(argc!=5)return 3;uint32_t bn,an;
 uint32_t*b=readfile(argv[1],&bn),*a=readfile(argv[2],&an);
 int got=spv_observer_contract(b,bn,a,an,atoi(argv[3])),want=atoi(argv[4]);
 free(a);free(b);if(got!=want){fprintf(stderr,"observer contract=%d want=%d\n",got,want);return 1;}return 0;
}
'''
        c=OUT/'guard.c';c.write_text(harness);cls.exe=OUT/'guard'
        subprocess.run(['clang','-std=c11','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined','-fno-omit-frame-pointer',str(c),'-o',str(cls.exe)],check=True)
        cls.v=fixtures.assemble('guard-v',fixtures.VERTEX)
        cls.f=fixtures.assemble('guard-f',fixtures.FRAGMENT)
        cls.vo,cls.fo,cls.meta=observer.instrument_pair(cls.v,cls.f,fixtures.sha(cls.v),fixtures.sha(cls.f))
        cls.count=0

    def check(self,base,alt,fragment,want,name):
        type(self).count+=1;n=str(type(self).count)+'-'+name
        bp=OUT/(n+'-base.spv');ap=OUT/(n+'-alt.spv')
        bp.write_bytes(base);ap.write_bytes(alt)
        subprocess.run([str(self.exe),str(bp),str(ap),str(int(fragment)),str(int(want))],check=True,
                       env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'},capture_output=True)

    def mutate(self,base,alt,fragment,name,fn):
        h,ins=spv.decode(alt);fn(h,ins)
        self.check(base,spv.encode(h,ins),fragment,False,name)

    def test_generated_pair_and_spirv14(self):
        # Fragment already has Input Location1; Output Location1 is separately vacant.
        self.check(self.v,self.vo,False,True,'vs')
        self.check(self.f,self.fo,True,True,'fs-input1')
        self.check(self.v,self.vo,True,False,'wrong-stage')
        v=fixtures.assemble('guard-v14',fixtures.VERTEX.replace('BufferBlock','Block').replace('Uniform','StorageBuffer').replace('%position %out0\n','%position %out0 %buffer\n'),'spv1.4')
        f=fixtures.assemble('guard-f14',fixtures.FRAGMENT,'spv1.4')
        vo,fo,_=observer.instrument_pair(v,f,fixtures.sha(v),fixtures.sha(f))
        self.check(v,vo,False,True,'vs14');self.check(f,fo,True,True,'fs14')

    def test_exact_new_io_contract(self):
        for stage,base,alt,frag in [('vertex',self.v,self.vo,False),('fragment',self.f,self.fo,True)]:
            side=self.meta[stage]['sideband_id']
            def missing(h,ins):ins.remove((71,[side,14]))
            def duplicate(h,ins):ins.insert(next(i for i,x in enumerate(ins) if x==(71,[side,14])),(71,[side,14]))
            def location(h,ins):ins[ins.index((71,[side,30,12]))]=(71,[side,30,11])
            def extra_decoration(h,ins):ins.insert(ins.index((71,[side,14])),(71,[side,16]))
            def missing_interface(h,ins):
                i=next(i for i,(op,a) in enumerate(ins) if op==15);op,a=ins[i];ins[i]=(op,[x for x in a if x!=side])
            def duplicate_interface(h,ins):
                i=next(i for i,(op,a) in enumerate(ins) if op==15);op,a=ins[i];ins[i]=(op,a+[side])
            for name,fn in [('missing-flat',missing),('duplicate-flat',duplicate),('wrong-location',location),('extra-decoration',extra_decoration),('missing-interface',missing_interface),('duplicate-interface',duplicate_interface)]:
                self.mutate(base,alt,frag,stage+'-'+name,fn)
            for scalar_op in (21,22):
                def wrongtype(h,ins,scalar_op=scalar_op):
                    var=next(a for op,a in ins if op==59 and a[1]==side)
                    ptr=var[0];idx=next(i for i,(op,a) in enumerate(ins) if op==32 and a[0]==ptr)
                    typ=next(a[0] for op,a in ins if op==scalar_op and (scalar_op==22 or a[1:]==[32,1]))
                    op,a=ins[idx];ins[idx]=(op,a[:2]+[typ])
                self.mutate(base,alt,frag,stage+'-wrong-type'+str(scalar_op),wrongtype)
        debug=self.meta['fragment']['debug_output_id']
        def badoutput(h,ins):ins[ins.index((71,[debug,30,1]))]=(71,[debug,30,2])
        self.mutate(self.f,self.fo,True,'wrong-output-location',badoutput)
        def output_float(h,ins):
            var=next(a for op,a in ins if op==59 and a[1]==debug)
            original=next(a for op,a in ins if op==59 and a[1]==self.meta['fragment']['original_output_id'])
            idx=next(i for i,(op,a) in enumerate(ins) if op==59 and a[1]==debug)
            ins[idx]=(59,[original[0],var[1],var[2]])
        self.mutate(self.f,self.fo,True,'float-output',output_float)

    def test_all_original_interfaces_and_resources_preserved(self):
        for stage,base,alt,frag in [('vertex',self.v,self.vo,False),('fragment',self.f,self.fo,True)]:
            bound=spv.decode(base)[0][3]
            _,orig=spv.decode(base)
            for op,args in orig:
                if op==71 and args[1] in (30,33,34):
                    def change(h,ins,args=args):ins[ins.index((71,args))]=(71,args[:-1]+[args[-1]+1])
                    self.mutate(base,alt,frag,stage+'-base-decoration'+str(args),change)
                if op==59 and args[2] in (1,2,3):
                    def remove(h,ins,args=args):ins.remove((59,args))
                    self.mutate(base,alt,frag,stage+'-base-variable'+str(args[1]),remove)
            def newresource(h,ins):
                at=next(i for i,(op,a) in enumerate(ins) if op==54)
                uid=next(a[0] for op,a in ins if op==21 and a[1:]==[32,0])
                ptr,var=h[3],h[3]+1;h[3]+=2
                ins[at:at]=[(32,[ptr,2,uid]),(59,[ptr,var,2])]
            self.mutate(base,alt,frag,stage+'-new-descriptor',newresource)
            def appendedoriginal(h,ins):
                i=next(i for i,(op,a) in enumerate(ins) if op==15);op,a=ins[i]
                existing=next(a[1] for op,a in ins if op==59 and a[1]<bound and a[2] in (1,3))
                ins[i]=(op,a+[existing])
            self.mutate(base,alt,frag,stage+'-appended-original',appendedoriginal)
        for name,bad in [('truncated',self.fo[:-4]),('zero-word',self.fo+b'\0'*4),('short',b'\0'*16)]:
            self.check(self.f,bad,True,False,name)

if __name__=='__main__':unittest.main()
