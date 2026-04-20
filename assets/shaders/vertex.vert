#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 fragColor;
layout(set = 0, binding = 0) uniform UniformBuffer
{
	mat4 projection;
	mat4 transformation;
} uniformBuffer;

void main()
{
	gl_Position = uniformBuffer.projection * uniformBuffer.transformation * vec4(inPosition, 1.0);
	fragColor = vec3(0.0, 1.0, 0.0);
}