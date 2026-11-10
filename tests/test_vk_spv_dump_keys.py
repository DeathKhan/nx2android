#!/usr/bin/env python3
"""Extract the real bounded runtime-SPIRV dump selector and writer."""
from pathlib import Path
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist/tests/spv-dump-keys"


class DumpKeysTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        src = (ROOT / "nxnative/dm/android/gfx_vk_shader.c").read_text(encoding='utf-8')
        begin = src.index("static int spv_runtime_dump_match(")
        selector = src[begin:src.index("static int spv_runtime_extra(", begin)]
        begin = src.index("static void spv_runtime_dump(", src.index("static int spv_runtime_contract("))
        writer = src[begin:src.index("// DM_VK_SPV_DIR=", begin)]
        harness = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef struct { uint64_t key; int stage, set, npool; } vkshader_t;
static const char *directory, *keys;
static int key_requests;
static const char *test_getenv(const char *name) {
    if (!strcmp(name,"DM_VK_SPV_DUMP_DIR")) return directory;
    assert(!strcmp(name,"DM_VK_SPV_DUMP_KEY")); key_requests++; return keys;
}
static int vk_pool_n(void) { return 65536; }
static int vk_pool_samp_n(void) { return 4096; }
#define VKLOG(...) ((void)0)
#define getenv test_getenv
''' + selector + writer + r'''
static int exists(const char *dir,uint64_t key) {
    char path[1024]; snprintf(path,sizeof path,"%s/%016llx.spv",dir,(unsigned long long)key);
    return !access(path,F_OK);
}
int main(int argc,char **argv) {
    assert(argc==3); const char *mode=argv[1];
    if (!strcmp(mode,"parser")) {
        assert(spv_runtime_dump_match(NULL,1)); assert(spv_runtime_dump_match("",1));
        assert(spv_runtime_dump_match("cca3264814bd7187",0xcca3264814bd7187ull));
        assert(spv_runtime_dump_match("CCA3264814BD7187",0xcca3264814bd7187ull));
        assert(!spv_runtime_dump_match("cca3264814bd7187",3));
        const char *bad[]={"0","1","0xcca3264814bd7187"," cca3264814bd7187", "+ca3264814bd7187", "cca3264814bd7187 ", "cca3264814bd7187,", ",cca3264814bd7187", "cca3264814bd7187,,0000000000000001", "cca3264814bd7187,000000000000000g", "cca3264814bd71870"};
        for(unsigned i=0;i<sizeof bad/sizeof *bad;i++) assert(!spv_runtime_dump_match(bad[i],0xcca3264814bd7187ull));
        for(unsigned len=1;len<16;len++) { char *p=malloc(len+1);memset(p,'a',len);p[len]=0;assert(!spv_runtime_dump_match(p,0xaaaaaaaaaaaaaaaaull));free(p); }
        char list[33*17+1]={0}; size_t used=0;
        for(unsigned i=0;i<33;i++) {
            used+=(size_t)snprintf(list+used,sizeof list-used,"%s%016x",i?",":"",i+1);
            for(unsigned key=1;key<=i+1;key++) assert(spv_runtime_dump_match(list,key)==(i<32));
        }
        return 0;
    }
    vkshader_t shader={1,1,2,0};uint32_t words[]={0x07230203,0x10300,0,1,0};
    directory=NULL;keys="invalid";spv_runtime_dump(&shader,words,5);assert(!key_requests);
    directory="";spv_runtime_dump(&shader,words,5);assert(!key_requests);
    directory=argv[2];
    if (!strcmp(mode,"writer")) {
        keys="0000000000000001,invalid";spv_runtime_dump(&shader,words,5);assert(!exists(directory,1));
        char list[32*17+1]={0};size_t used=0;
        for(unsigned i=0;i<32;i++) used+=(size_t)snprintf(list+used,sizeof list-used,"%s%016x",i?",":"",i+1);
        keys=list;
        for(unsigned i=1;i<=32;i++) {shader.key=i;spv_runtime_dump(&shader,words,5);assert(exists(directory,i));}
        keys="0000000000000021";shader.key=33;spv_runtime_dump(&shader,words,5);assert(!exists(directory,33));
        char path[1024];snprintf(path,sizeof path,"%s/%016x.spv",directory,32);FILE*f=fopen(path,"rb");assert(f);uint32_t got[5];assert(fread(got,4,5,f)==5);fclose(f);assert(!memcmp(got,words,sizeof words));
    } else {
        assert(!strcmp(mode,"unfiltered")); keys=NULL;
        for(unsigned i=1;i<=17;i++){shader.key=i;spv_runtime_dump(&shader,words,5);assert(exists(directory,i)==(i<=16));}
    }
    return 0;
}
'''
        c = OUT / "dump-keys.c"
        c.write_text(harness, encoding='utf-8')
        cls.exe = OUT / "dump-keys"
        subprocess.run(["cc", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                        str(c), "-o", str(cls.exe)], check=True)

    def run_case(self, mode):
        output = OUT / mode
        output.mkdir(exist_ok=True)
        for p in output.glob("*.spv*"):
            p.unlink()
        subprocess.run([str(self.exe), mode, str(output)], check=True,
                       env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1"})

    def test_strict_parser_all_or_nothing_and_bounds(self):
        self.run_case("parser")

    def test_actual_dump_budget_and_disabled_fast_path(self):
        self.run_case("writer")
        self.run_case("unfiltered")


if __name__ == "__main__":
    unittest.main()
