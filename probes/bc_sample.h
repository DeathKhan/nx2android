#pragma once
// Little-endian64-byte header, tight16-byte compressed blocks, then float4 points.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    char magic[8];
    uint32_t version,width,height,layers,vk_format,metal_format,point_count,reserved;
    uint64_t block_bytes,point_bytes;
    uint8_t padding[8];
} bc_sample_header;
_Static_assert(sizeof(bc_sample_header)==64,"BCSAMP01 header");
typedef struct { bc_sample_header h; uint8_t *blocks; float *points; } bc_sample_input;
static int bc_sample_read(const char *path,bc_sample_input *in) {
    memset(in,0,sizeof *in);
    FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
    if(fread(&in->h,1,64,f)!=64){fclose(f);return 0;}
    bc_sample_header *h=&in->h;
    if(memcmp(h->magic,"BCSAMP01",8)||h->version!=1||!h->width||!h->height||!h->layers||
       h->width>16384||h->height>16384||h->layers>64||!h->point_count||h->point_count>16384||
       !((h->vk_format==143&&h->metal_format==151)||(h->vk_format==145&&h->metal_format==152))) {fclose(f);return 0;}
    uint64_t blocks=(uint64_t)((h->width+3)/4)*((h->height+3)/4)*16*h->layers;
    if(blocks!=h->block_bytes||blocks>(256u<<20)||h->point_bytes!=(uint64_t)h->point_count*16){fclose(f);return 0;}
    in->blocks=malloc((size_t)blocks);in->points=malloc((size_t)h->point_bytes);
    if(!in->blocks||!in->points){fclose(f);free(in->blocks);free(in->points);memset(in,0,sizeof *in);return 0;}
    int ok=fread(in->blocks,1,(size_t)blocks,f)==blocks &&
           fread(in->points,1,(size_t)h->point_bytes,f)==h->point_bytes && fgetc(f)==EOF;
    fclose(f);
    if(!ok){free(in->blocks);free(in->points);memset(in,0,sizeof *in);}
    return ok;
}
static int bc_sample_write(const char *path,const void *data,size_t bytes) {
    FILE *f=fopen(path,"wb");if(!f){perror(path);return 0;}
    int ok=fwrite(data,1,bytes,f)==bytes;
    if(fclose(f))ok=0;
    return ok;
}

