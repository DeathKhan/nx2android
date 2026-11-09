// Offline upload reproducer: link the exact NVN layout/deswizzle sources used by Vulkan.
// Arguments preserve nvn_texdesc fields; no synthetic texture or replacement decoder.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../nxnative/dm/gfx/tex/nvn_deswizzle.h"
static unsigned number(const char *s) {
    char *e; unsigned long n=strtoul(s,&e,0);
    if (!*s || *e || n>UINT32_MAX) { fprintf(stderr,"invalid number: %s\n",s);exit(2); }
    return (unsigned)n;
}
int main(int argc,char **argv) {
    if (argc!=14) {
        fprintf(stderr,"usage: %s guest.bin output-prefix target format flags samples width height depth levels stride pkg_block_h switch2\n",argv[0]);return 2;
    }
    nvn_texdesc d={.target=(int)number(argv[3]),.format=(int)number(argv[4]),
        .flags=(int)number(argv[5]),.samples=(int)number(argv[6]),.width=number(argv[7]),
        .height=number(argv[8]),.depth=number(argv[9]),.levels=number(argv[10]),
        .stride=number(argv[11]),.pkg_block_h=(int)number(argv[12])};
    nvn_gob_switch2=(int)number(argv[13]);
    if (!d.width || !d.height || d.width>16384 || d.height>16384 || d.depth>2048 ||
        d.levels>16 || d.pkg_block_h>256 || nvn_gob_switch2>1) return 2;
    nvn_texlayout L;nvn_tex_layout(&d,&L);
    FILE *f=fopen(argv[1],"rb");if(!f){perror(argv[1]);return 2;}
    fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);
    if(n<=0 || (uint64_t)n<L.size || n>(256L<<20)){fprintf(stderr,"storage size mismatch: file%ld layout%llu\n",n,(unsigned long long)L.size);fclose(f);return 2;}
    uint8_t *src=malloc((size_t)n);if(!src)return 2;
    if(fread(src,1,(size_t)n,f)!=(size_t)n)return 2;
    fclose(f);
    printf("{\"size\":%llu,\"layer_size\":%llu,\"tileH\":%u,\"tileD\":%u,\"layers\":%u,\"subresources\":[",
        (unsigned long long)L.size,(unsigned long long)L.layer_size,L.tileH,L.tileD,L.layers);
    int first=1;
    for(unsigned level=0;level<(d.levels?d.levels:1);level++) for(unsigned layer=0;layer<L.layers;layer++){
        uint32_t row,rows,slices,bpb;uint64_t base;unsigned th,td;
        if(nvn_deswizzle_geom(&d,level,layer,&row,&rows,&slices,&base,&th,&td,&bpb))return 2;
        size_t slab=(size_t)row*rows*slices;
        uint8_t *dst=malloc(slab);if(!dst)return 2;memset(dst,0xa5,slab);
        if(nvn_deswizzle(&d,level,layer,src,dst,row))return 2;
        char path[1024];if(snprintf(path,sizeof path,"%s-l%u-a%u.bin",argv[2],level,layer)>=(int)sizeof path)return 2;
        f=fopen(path,"wb");if(!f)return 2;
        int ok=fwrite(dst,1,slab,f)==slab;if(fclose(f))ok=0;free(dst);if(!ok)return 2;
        printf("%s{\"level\":%u,\"layer\":%u,\"base\":%llu,\"row_bytes\":%u,\"rows\":%u,\"slices\":%u,\"tileH\":%u,\"tileD\":%u,\"bytes\":%zu,\"path\":\"%s\"}",
            first?"":",",level,layer,(unsigned long long)base,row,rows,slices,th,td,slab,path);first=0;
    }
    printf("]}\n");free(src);return 0;
}

