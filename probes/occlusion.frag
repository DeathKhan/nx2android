#version 450
layout(constant_id = 0) const bool kill = false;
layout(location = 0) out vec4 color;
void main() { if (kill) discard; color = vec4(1); }
