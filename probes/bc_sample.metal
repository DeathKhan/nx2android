#include <metal_stdlib>
using namespace metal;
kernel void bc_sample(texture2d_array<float, access::sample> src [[texture(0)]],
                      sampler smp [[sampler(0)]],
                      const device float4 *points [[buffer(0)]],
                      device float4 *results [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if(id<count) results[id]=src.sample(smp,points[id].xy,uint(points[id].z),level(0.0f));
}

