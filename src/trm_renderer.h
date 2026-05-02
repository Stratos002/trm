#ifndef TRM_RENDERER
#define TRM_RENDERER

#include "vulkan/vulkan.h"
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#include <stdint.h>
#include <stdbool.h>

#define TRM_RENDERER_SWAPCHAIN_IMAGE UINT32_MAX
#define TRM_RENDERER_SWAPCHAIN_IMAGE_FORMAT UINT32_MAX

enum TRM_Renderer_BufferUsage
{
	TRM_RENDERER_BUFFER_USAGE_UNIFORM = (1 << 0),
	TRM_RENDERER_BUFFER_USAGE_STORAGE = (1 << 1),
	TRM_RENDERER_BUFFER_USAGE_TRANSFER_SRC = (1 << 2),
	TRM_RENDERER_BUFFER_USAGE_TRANSFER_DST = (1 << 3),
	TRM_RENDERER_BUFFER_USAGE_VERTEX = (1 << 4)
};

struct TRM_Renderer_BufferCreateInfo
{
	uint32_t sizeInBytes;
	bool hostVisible;
	enum TRM_Renderer_BufferUsage usage;
};

enum TRM_Renderer_ImageUsage
{
	TRM_RENDERER_IMAGE_USAGE_COLOR_ATTACHMENT = (1 << 0),
	TRM_RENDERER_IMAGE_USAGE_DEPTH_ATTACHMENT = (1 << 1),
	TRM_RENDERER_IMAGE_USAGE_SAMPLED = (1 << 2),
	TRM_RENDERER_IMAGE_USAGE_STORAGE = (1 << 3),
	TRM_RENDERER_IMAGE_USAGE_TRANSFER_SRC = (1 << 4),
	TRM_RENDERER_IMAGE_USAGE_TRANSFER_DST = (1 << 5)
};

struct TRM_Renderer_ImageCreateInfo
{
	uint32_t width;
	uint32_t height;
	VkFormat format;
	enum TRM_Renderer_ImageUsage usage;
};

enum TRM_Renderer_DescriptorType
{
	TRM_RENDERER_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	TRM_RENDERER_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	TRM_RENDERER_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	TRM_RENDERER_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
};

enum TRM_Renderer_PassType
{
	TRM_RENDERER_PASS_TYPE_DISPATCH,
	TRM_RENDERER_PASS_TYPE_DRAW,
	TRM_RENDERER_PASS_TYPE_IMAGE_TO_IMAGE_COPY,
	TRM_RENDERER_PASS_TYPE_BUFFER_TO_IMAGE_COPY,
	TRM_RENDERER_PASS_TYPE_BUFFER_TO_BUFFER_COPY,
	TRM_RENDERER_PASS_TYPE_BLIT,
	TRM_RENDERER_PASS_TYPE_PRESENT
};

enum TRM_Renderer_ResourceAccessFlag
{
	TRM_RENDERER_SHADER_ACCESS_FLAG_READ = (1 << 0),
	TRM_RENDERER_SHADER_ACCESS_FLAG_WRITE = (1 << 1)
};

struct TRM_Renderer_DescriptorInfo
{
	enum TRM_Renderer_DescriptorType descriptorType;
	uint32_t resourceAccessFlags;
};

struct TRM_Renderer_DrawPassCreateInfo
{
	uint32_t vertexCodeSize;
	uint32_t* pVertexCode;
	uint32_t fragmentCodeSize;
	uint32_t* pFragmentCode;
	uint32_t width;
	uint32_t height;
	VkFormat colorImageFormat;
	VkFormat depthImageFormat;
	uint32_t descriptorInfoCount;
	struct TRM_Renderer_DescriptorInfo* pDescriptorInfos;
};

struct TRM_Renderer_DispatchPassCreateInfo
{
	uint32_t codeSize;
	uint32_t* pCode;
	uint32_t descriptorInfoCount;
	struct TRM_Renderer_DescriptorInfo* pDescriptorInfos;
};

struct TRM_Renderer_DispatchPassInstance
{
	uint32_t pass;
	uint32_t groupCountX;
	uint32_t groupCountY;
	uint32_t groupCountZ;
	uint32_t bindingCount;
	uint32_t* pBindings;
};

struct TRM_Renderer_DrawPassInstance
{
	uint32_t pass;
	uint32_t vertexCount;
	uint32_t vertexBuffer;
	uint32_t bindingCount;
	uint32_t colorImage;
	uint32_t depthImage;
	uint32_t* pBindings;
};

struct TRM_Renderer_CopyImageToImagePassInstanceInfo
{
	uint32_t srcImage;
	uint32_t dstImage;
	uint32_t width;
	uint32_t height;
};

struct TRM_Renderer_CopyBufferToImagePassInstanceInfo
{
	uint32_t srcBuffer;
	uint32_t dstImage;
	uint32_t width;
	uint32_t height;
};

struct TRM_Renderer_CopyBufferToBufferPassInstanceInfo
{
	uint32_t srcBuffer;
	uint32_t dstBuffer;
	uint32_t sizeInBytes;
};

struct TRM_Renderer_BlitPassInstanceInfo
{
	uint32_t srcImage;
	uint32_t dstImage;
	uint32_t width;
	uint32_t height;
};

struct TRM_Renderer_PassInstance
{
	enum TRM_Renderer_PassType type;
	union
	{
		struct TRM_Renderer_DispatchPassInstance dispatch;
		struct TRM_Renderer_DrawPassInstance draw;
		struct TRM_Renderer_CopyImageToImagePassInstanceInfo imageToImageCopy;
		struct TRM_Renderer_CopyBufferToImagePassInstanceInfo bufferToImageCopy;
		struct TRM_Renderer_CopyBufferToBufferPassInstanceInfo bufferToBufferCopy;
		struct TRM_Renderer_BlitPassInstanceInfo blit;
	} info;
};

// TODO : remove
struct Vertex
{
	float x;
	float y;
	float z;
};

void TRM_Renderer_start(GLFWwindow* pWindow, uint32_t windowWidth, uint32_t windowHeight);

void TRM_Renderer_terminate(void);

void TRM_Renderer_beginFrame(void);

void TRM_Renderer_endFrame(uint32_t passInstanceCount, struct TRM_Renderer_PassInstance* pPassInstances);

void TRM_Renderer_createBuffer(struct TRM_Renderer_BufferCreateInfo info, uint32_t* pHandle);

void TRM_Renderer_createImage(struct TRM_Renderer_ImageCreateInfo info, uint32_t* pHandle);

void TRM_Renderer_destroyResource(uint32_t handle);

void TRM_Renderer_writeBuffer(uint32_t sizeInBytes, const void* pData, uint32_t handle);

void TRM_Renderer_createDispatchPass(struct TRM_Renderer_DispatchPassCreateInfo info, uint32_t* pHandle);

void TRM_Renderer_createDrawPass(struct TRM_Renderer_DrawPassCreateInfo info, uint32_t* pHandle);

void TRM_Renderer_destroyPass(uint32_t handle);

#endif
