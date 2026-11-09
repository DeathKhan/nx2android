#version 450
layout(location=0) flat in float floatCarrier;
layout(location=1) flat in uint uintCarrier;
layout(location=2) flat in uint tag;
layout(location=0) out uvec4 observed;
void main() {
    observed=uvec4(floatBitsToUint(floatCarrier),uintCarrier,tag,0xc0decafeu);
}
