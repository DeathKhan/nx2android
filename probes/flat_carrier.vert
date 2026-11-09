#version 450
layout(set=0,binding=0,std430) readonly buffer Payload { uint words[]; } payload;
layout(location=0) flat out float floatCarrier;
layout(location=1) flat out uint uintCarrier;
layout(location=2) flat out uint tag;
layout(push_constant) uniform Config { uint count; } config;
void main() {
    const vec2 p[6]=vec2[6](vec2(0,0),vec2(1,0),vec2(0,1),
                           vec2(0,1),vec2(1,0),vec2(1,1));
    uint index=uint(gl_InstanceIndex);
    uint raw=payload.words[index*21u+4u]; // Native mesh vertex word4, read at runtime.
    floatCarrier=uintBitsToFloat(raw);
    uintCarrier=raw;
    tag=index;
    vec2 point=p[gl_VertexIndex];
    gl_Position=vec4(2.0*(float(index)+point.x)/float(config.count)-1.0,
                     2.0*point.y-1.0,0.0,1.0);
}
