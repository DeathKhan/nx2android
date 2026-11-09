// Standalone fixed-input Metal compressed-texture sampling, no game host required.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include "bc_sample.h"
#define REQUIRE(x,msg) do { if(!(x)){fprintf(stderr,"%s\n",msg);return 2;} } while(0)
int main(int argc,char **argv) { @autoreleasepool {
    if(argc!=4){fprintf(stderr,"usage: metal_bc_sample input.bcs output.rgba32f shader.metal\n");return 2;}
    bc_sample_input in;REQUIRE(bc_sample_read(argv[1],&in),"invalid BCSAMP01 input");
    bc_sample_header *h=&in.h;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    REQUIRE(dev,"no Metal device");REQUIRE(dev.supportsBCTextureCompression,"device has no BC texture compression");
    NSString *text=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[3]] encoding:NSUTF8StringEncoding error:nil];
    REQUIRE(text,"shader source could not be read");
    MTLCompileOptions *opt=[MTLCompileOptions new];opt.fastMathEnabled=NO;
    NSError *error=nil;
    id<MTLLibrary> lib=[dev newLibraryWithSource:text options:opt error:&error];
    if(!lib){fprintf(stderr,"Metal compile: %s\n",error.localizedDescription.UTF8String);return 2;}
    id<MTLFunction> fn=[lib newFunctionWithName:@"bc_sample"];
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:fn error:&error];
    if(!pipeline){fprintf(stderr,"Metal pipeline: %s\n",error.localizedDescription.UTF8String);return 2;}
    MTLTextureDescriptor *td=[MTLTextureDescriptor new];
    td.textureType=MTLTextureType2DArray;td.pixelFormat=(MTLPixelFormat)h->metal_format;
    td.width=h->width;td.height=h->height;td.depth=1;td.arrayLength=h->layers;td.mipmapLevelCount=1;
    td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageShaderRead;
    id<MTLTexture> image=[dev newTextureWithDescriptor:td];REQUIRE(image,"compressed texture allocation failed");
    size_t row=((h->width+3)/4)*16,rows=(h->height+3)/4,pitch=(row+255)&~(size_t)255,slice=pitch*rows,total=slice*h->layers;
    id<MTLBuffer> staging=[dev newBufferWithLength:total options:MTLResourceStorageModeShared];
    id<MTLBuffer> roundtrip=[dev newBufferWithLength:total options:MTLResourceStorageModeShared];
    id<MTLBuffer> points=[dev newBufferWithBytes:in.points length:(NSUInteger)h->point_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> result=[dev newBufferWithLength:(NSUInteger)h->point_bytes options:MTLResourceStorageModeShared];
    REQUIRE(staging&&roundtrip&&points&&result,"buffer allocation failed");
    memset(staging.contents,0,total);memset(roundtrip.contents,0xa5,total);memset(result.contents,0xcd,(size_t)h->point_bytes);
    for(uint32_t ly=0;ly<h->layers;ly++)for(size_t y=0;y<rows;y++)
        memcpy((uint8_t*)staging.contents+ly*slice+y*pitch,in.blocks+(ly*rows+y)*row,row);
    MTLSamplerDescriptor *sd=[MTLSamplerDescriptor new];
    sd.normalizedCoordinates=YES;sd.minFilter=MTLSamplerMinMagFilterLinear;sd.magFilter=MTLSamplerMinMagFilterLinear;
    sd.mipFilter=MTLSamplerMipFilterNearest;sd.sAddressMode=MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode=MTLSamplerAddressModeClampToEdge;sd.rAddressMode=MTLSamplerAddressModeClampToEdge;
    sd.maxAnisotropy=1;sd.compareFunction=MTLCompareFunctionNever;sd.lodMinClamp=0;sd.lodMaxClamp=0;
    id<MTLSamplerState> sampler=[dev newSamplerStateWithDescriptor:sd];REQUIRE(sampler,"sampler failed");
    id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[cb blitCommandEncoder];
    MTLSize extent=MTLSizeMake(h->width,h->height,1);MTLOrigin origin=MTLOriginMake(0,0,0);
    for(uint32_t ly=0;ly<h->layers;ly++)
        [blit copyFromBuffer:staging sourceOffset:ly*slice sourceBytesPerRow:pitch sourceBytesPerImage:slice sourceSize:extent toTexture:image destinationSlice:ly destinationLevel:0 destinationOrigin:origin];
    [blit endEncoding];
    id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
    [ce setComputePipelineState:pipeline];[ce setTexture:image atIndex:0];[ce setSamplerState:sampler atIndex:0];
    [ce setBuffer:points offset:0 atIndex:0];[ce setBuffer:result offset:0 atIndex:1];[ce setBytes:&h->point_count length:4 atIndex:2];
    [ce dispatchThreadgroups:MTLSizeMake((h->point_count+31)/32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [ce endEncoding];
    blit=[cb blitCommandEncoder];
    for(uint32_t ly=0;ly<h->layers;ly++)
        [blit copyFromTexture:image sourceSlice:ly sourceLevel:0 sourceOrigin:origin sourceSize:extent toBuffer:roundtrip destinationOffset:ly*slice destinationBytesPerRow:pitch destinationBytesPerImage:slice];
    [blit endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"Metal execution: %s\n",cb.error.localizedDescription.UTF8String);return 2;}
    size_t bad=0;
    for(uint32_t ly=0;ly<h->layers;ly++)for(size_t y=0;y<rows;y++)
        for(size_t x=0;x<row;x++)bad+=((uint8_t*)roundtrip.contents)[ly*slice+y*pitch+x]!=in.blocks[(ly*rows+y)*row+x];
    REQUIRE(bc_sample_write(argv[2],result.contents,(size_t)h->point_bytes),"output write failed");
    const float *v=result.contents;unsigned nonfinite=0,alpha_bad=0;float lo=INFINITY,hi=-INFINITY;
    for(uint32_t i=0;i<h->point_count;i++){alpha_bad+=v[i*4+3]!=1.f;for(int c=0;c<4;c++){float x=v[i*4+c];nonfinite+=!isfinite(x);if(isfinite(x)){if(x<lo)lo=x;if(x>hi)hi=x;}}}
    printf("device=%s api=Metal format=%u dimensions=%ux%ux%u points=%u bytes=%llu upload_roundtrip_byte_differences=%zu nonfinite_components=%u alpha_not_one=%u sample_range=[%.9g,%.9g]\n",
        dev.name.UTF8String,h->metal_format,h->width,h->height,h->layers,h->point_count,(unsigned long long)h->point_bytes,bad,nonfinite,alpha_bad,lo,hi);
    free(in.blocks);free(in.points);return bad?1:0;
} }

