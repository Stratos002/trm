#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec3 fragColor;
layout(set = 0, binding = 0) uniform UniformBuffer
{
	vec4 color;
} uniformBuffer;

void main()
{
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = uniformBuffer.color.rgb;
}