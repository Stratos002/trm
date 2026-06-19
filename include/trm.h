#ifndef TRM_H
#define TRM_H

#if defined(TRM_PLATFORM_WINDOWS)
#include <windows.h>
#elif defined(TRM_PLATFORM_LINUX)
#include <wayland-client.h>
#endif

#include <stdint.h>
#include <stdbool.h>

#define TRM_SWAPCHAIN_IMAGE UINT32_MAX
#define TRM_SWAPCHAIN_IMAGE_FORMAT UINT32_MAX

enum TRM_ResourceType
{
	TRM_RESOURCE_TYPE_BUFFER,
	TRM_RESOURCE_TYPE_IMAGE
};

enum TRM_BufferUsage
{
	TRM_BUFFER_USAGE_UNIFORM = (1 << 0),
	TRM_BUFFER_USAGE_STORAGE = (1 << 1),
	TRM_BUFFER_USAGE_TRANSFER_SRC = (1 << 2),
	TRM_BUFFER_USAGE_TRANSFER_DST = (1 << 3),
	TRM_BUFFER_USAGE_VERTEX = (1 << 4),
	TRM_BUFFER_USAGE_INDEX = (1 << 5),
	TRM_BUFFER_USAGE_DRAW_INDIRECT = (1 << 6),
	TRM_BUFFER_USAGE_MAX = (1 << 7)
};

struct TRM_BufferCreateInfo
{
	uint32_t sizeInBytes;
	bool hostVisible;
	enum TRM_BufferUsage usage;
};

/*
UNORM :		input : [0 : 255]	 -> shader : [0.0 : 1.0]
SNORM :		input : [-128 : 127] -> shader : [-1.0 : 1.0]
USCALED :	input : [0 : 255]	 -> shader : [0.0 : 255.0]
SSCALED :	input : [-128 : 127] -> shader : [-128.0 : 127.0]
UINT :		input : [0 : 255]	 -> shader : [0 : 255]
SINT :		input : [-128 : 127] -> shader : [-128 : 127]
SRGB :		input : [0 : 255]	 -> shader : [0.0 : 1.0] (gamma corr.)
SFLOAT :	just a float
*/
enum TRM_Format
{
	TRM_FORMAT_D32_SFLOAT,
	TRM_FORMAT_R8_UNORM,
	TRM_FORMAT_R8_SNORM,
	TRM_FORMAT_R8_USCALED,
	TRM_FORMAT_R8_SSCALED,
	TRM_FORMAT_R8_UINT,
	TRM_FORMAT_R8_SINT,
	TRM_FORMAT_R8_SRGB,
	TRM_FORMAT_R8G8_UNORM,
	TRM_FORMAT_R8G8_SNORM,
	TRM_FORMAT_R8G8_USCALED,
	TRM_FORMAT_R8G8_SSCALED,
	TRM_FORMAT_R8G8_UINT,
	TRM_FORMAT_R8G8_SINT,
	TRM_FORMAT_R8G8_SRGB,
	TRM_FORMAT_R8G8B8_UNORM,
	TRM_FORMAT_R8G8B8_SNORM,
	TRM_FORMAT_R8G8B8_USCALED,
	TRM_FORMAT_R8G8B8_SSCALED,
	TRM_FORMAT_R8G8B8_UINT,
	TRM_FORMAT_R8G8B8_SINT,
	TRM_FORMAT_R8G8B8_SRGB,
	TRM_FORMAT_B8G8R8_UNORM,
	TRM_FORMAT_B8G8R8_SNORM,
	TRM_FORMAT_B8G8R8_USCALED,
	TRM_FORMAT_B8G8R8_SSCALED,
	TRM_FORMAT_B8G8R8_UINT,
	TRM_FORMAT_B8G8R8_SINT,
	TRM_FORMAT_B8G8R8_SRGB,
	TRM_FORMAT_R8G8B8A8_UNORM,
	TRM_FORMAT_R8G8B8A8_SNORM,
	TRM_FORMAT_R8G8B8A8_USCALED,
	TRM_FORMAT_R8G8B8A8_SSCALED,
	TRM_FORMAT_R8G8B8A8_UINT,
	TRM_FORMAT_R8G8B8A8_SINT,
	TRM_FORMAT_R8G8B8A8_SRGB,
	TRM_FORMAT_B8G8R8A8_UNORM,
	TRM_FORMAT_B8G8R8A8_SNORM,
	TRM_FORMAT_B8G8R8A8_USCALED,
	TRM_FORMAT_B8G8R8A8_SSCALED,
	TRM_FORMAT_B8G8R8A8_UINT,
	TRM_FORMAT_B8G8R8A8_SINT,
	TRM_FORMAT_B8G8R8A8_SRGB,
	TRM_FORMAT_R16G16_UNORM,
	TRM_FORMAT_R16G16_SNORM,
	TRM_FORMAT_R16G16_USCALED,
	TRM_FORMAT_R16G16_SSCALED,
	TRM_FORMAT_R16G16_UINT,
	TRM_FORMAT_R16G16_SINT,
	TRM_FORMAT_R16G16_SFLOAT,
	TRM_FORMAT_R16G16B16_UNORM,
	TRM_FORMAT_R16G16B16_SNORM,
	TRM_FORMAT_R16G16B16_USCALED,
	TRM_FORMAT_R16G16B16_SSCALED,
	TRM_FORMAT_R16G16B16_UINT,
	TRM_FORMAT_R16G16B16_SINT,
	TRM_FORMAT_R16G16B16_SFLOAT,
	TRM_FORMAT_R16G16B16A16_UNORM,
	TRM_FORMAT_R16G16B16A16_SNORM,
	TRM_FORMAT_R16G16B16A16_USCALED,
	TRM_FORMAT_R16G16B16A16_SSCALED,
	TRM_FORMAT_R16G16B16A16_UINT,
	TRM_FORMAT_R16G16B16A16_SINT,
	TRM_FORMAT_R16G16B16A16_SFLOAT,
	TRM_FORMAT_R32_UINT,
	TRM_FORMAT_R32_SINT,
	TRM_FORMAT_R32_SFLOAT,
	TRM_FORMAT_R32G32_UINT,
	TRM_FORMAT_R32G32_SINT,
	TRM_FORMAT_R32G32_SFLOAT,
	TRM_FORMAT_R32G32B32_UINT,
	TRM_FORMAT_R32G32B32_SINT,
	TRM_FORMAT_R32G32B32_SFLOAT,
	TRM_FORMAT_R32G32B32A32_UINT,
	TRM_FORMAT_R32G32B32A32_SINT,
	TRM_FORMAT_R32G32B32A32_SFLOAT
};

enum TRM_ImageUsage
{
	TRM_IMAGE_USAGE_COLOR_ATTACHMENT = (1 << 0),
	TRM_IMAGE_USAGE_DEPTH_ATTACHMENT = (1 << 1),
	TRM_IMAGE_USAGE_SAMPLED = (1 << 2),
	TRM_IMAGE_USAGE_STORAGE = (1 << 3),
	TRM_IMAGE_USAGE_TRANSFER_SRC = (1 << 4),
	TRM_IMAGE_USAGE_TRANSFER_DST = (1 << 5),
	TRM_IMAGE_USAGE_MAX = (1 << 6)
};

struct TRM_ImageCreateInfo
{
	uint32_t width;
	uint32_t height;
	enum TRM_Format format;
	enum TRM_ImageUsage usage;
};

struct TRM_ResourceCreateInfo
{
	enum TRM_ResourceType type;
	union
	{
		struct TRM_BufferCreateInfo buffer;
		struct TRM_ImageCreateInfo image;
	} info;
};

enum TRM_DescriptorType
{
	TRM_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	TRM_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	TRM_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
};

enum TRM_PipelineType
{
	TRM_PIPELINE_TYPE_COMPUTE,
	TRM_PIPELINE_TYPE_GRAPHICS
};

enum TRM_ResourceAccessFlag
{
	TRM_SHADER_ACCESS_FLAG_READ = (1 << 0),
	TRM_SHADER_ACCESS_FLAG_WRITE = (1 << 1)
};

struct TRM_DescriptorInfo
{
	enum TRM_DescriptorType descriptorType;
	uint32_t resourceAccessFlags;
};

struct TRM_VertexAttributeDescription
{
	uint32_t shaderLocation;
	uint32_t offset;
	enum TRM_Format format;
};

struct TRM_GraphicsPipelineCreateInfo
{
	uint32_t vertexCodeSize;
	const void* pVertexCode;
	uint32_t fragmentCodeSize;
	const void* pFragmentCode;
	uint32_t colorOutputImageCount;
	enum TRM_Format* pColorOutputImageFormats;
	enum TRM_Format depthOutputFormat;
	uint32_t vertexStride;
	uint32_t vertexAttributeDescriptionCount;
	const struct TRM_VertexAttributeDescription* pVertexAttributeDescriptions;
	uint32_t descriptorInfoCount;
	const struct TRM_DescriptorInfo* pDescriptorInfos;
};

struct TRM_ComputePipelineCreateInfo
{
	uint32_t codeSize;
	const uint32_t* pCode;
	uint32_t descriptorInfoCount;
	const struct TRM_DescriptorInfo* pDescriptorInfos;
};

enum TRM_PassType
{
	TRM_PASS_TYPE_DISPATCH,
	TRM_PASS_TYPE_DRAW_VERTEX,
	TRM_PASS_TYPE_DRAW_INDEXED,
	TRM_PASS_TYPE_DRAW_VERTEX_INDIRECT,
	TRM_PASS_TYPE_DRAW_INDEXED_INDIRECT,
	TRM_PASS_TYPE_IMAGE_TO_IMAGE_COPY,
	TRM_PASS_TYPE_BUFFER_TO_IMAGE_COPY,
	TRM_PASS_TYPE_BUFFER_TO_BUFFER_COPY,
	TRM_PASS_TYPE_BLIT,
	TRM_PASS_TYPE_PRESENT
};

struct TRM_DispatchPassInfo
{
	uint32_t pipeline;
	uint32_t groupCountX;
	uint32_t groupCountY;
	uint32_t groupCountZ;
	uint32_t bindingCount;
	const uint32_t* pBindings;
};

struct TRM_ClearColor
{
	float color[4];
};

struct TRM_DrawVertexPassInfo
{
	uint32_t pipeline;
	uint32_t width;
	uint32_t height;
	uint32_t vertexCount;
	bool useVertexBuffer;
	uint32_t vertexBuffer;
	uint32_t colorOutputImageCount;
	uint32_t* pColorOutputImages;
	const struct TRM_ClearColor* pClearColors;
	uint32_t depthOutputImage;
	uint32_t bindingCount;
	const uint32_t* pBindings;
};

struct TRM_DrawIndexedPassInfo
{
	uint32_t pipeline;
	uint32_t width;
	uint32_t height;
	uint32_t indexCount;
	uint32_t indexBuffer;
	uint32_t vertexBuffer;
	uint32_t colorOutputImageCount;
	uint32_t* pColorOutputImages;
	const struct TRM_ClearColor* pClearColors;
	uint32_t depthOutputImage;
	uint32_t bindingCount;
	const uint32_t* pBindings;
};

struct TRM_DrawVertexIndirectPassInfo
{
	uint32_t pipeline;
	uint32_t width;
	uint32_t height;
	uint32_t drawCount;
	uint32_t buffer;
	bool useVertexBuffer;
	uint32_t vertexBuffer;
	uint32_t colorOutputImageCount;
	uint32_t* pColorOutputImages;
	const struct TRM_ClearColor* pClearColors;
	uint32_t depthOutputImage;
	uint32_t bindingCount;
	const uint32_t* pBindings;
};

struct TRM_DrawIndexedIndirectPassInfo
{
	uint32_t pipeline;
	uint32_t width;
	uint32_t height;
	uint32_t drawCount;
	uint32_t buffer;
	uint32_t indexBuffer;
	uint32_t vertexBuffer;
	uint32_t colorOutputImageCount;
	uint32_t* pColorOutputImages;
	const struct TRM_ClearColor* pClearColors;
	uint32_t depthOutputImage;
	uint32_t bindingCount;
	const uint32_t* pBindings;
};

struct TRM_CopyImageToImagePassInfo
{
	uint32_t srcImage;
	uint32_t dstImage;
	uint32_t width;
	uint32_t height;
};

struct TRM_CopyBufferToImagePassInfo
{
	uint32_t srcBuffer;
	uint32_t dstImage;
	uint32_t width;
	uint32_t height;
};

struct TRM_CopyBufferToBufferPassInfo
{
	uint32_t srcBuffer;
	uint32_t dstBuffer;
	uint32_t sizeInBytes;
};

struct TRM_BlitPassInfo
{
	uint32_t srcImage;
	uint32_t dstImage;
	uint32_t srcWidth;
	uint32_t srcHeight;
	uint32_t dstWidth;
	uint32_t dstHeight;
};

struct TRM_Pass
{
	enum TRM_PassType type;
	union
	{
		struct TRM_DispatchPassInfo dispatch;
		struct TRM_DrawVertexPassInfo drawVertex;
		struct TRM_DrawIndexedPassInfo drawIndexed;
		struct TRM_DrawVertexIndirectPassInfo drawVertexIndirect;
		struct TRM_DrawIndexedIndirectPassInfo drawIndexedIndirect;
		struct TRM_CopyImageToImagePassInfo imageToImageCopy;
		struct TRM_CopyBufferToImagePassInfo bufferToImageCopy;
		struct TRM_CopyBufferToBufferPassInfo bufferToBufferCopy;
		struct TRM_BlitPassInfo blit;
	} info;
};

struct TRM_NativeWindow
{
#if defined(TRM_PLATFORM_WINDOWS)
	HINSTANCE hinstance;
	HWND hwnd;
#elif defined(TRM_PLATFORM_LINUX)
	struct wl_display* display;
	struct wl_surface* surface;
#endif
};

void TRM_start(struct TRM_NativeWindow nativeWindow);

void TRM_terminate(void);

void TRM_beginFrame(void);

void TRM_endFrame(uint32_t passCount, struct TRM_Pass* pPasses, uint32_t windowWidth, uint32_t windowHeight);

void TRM_createResource(struct TRM_ResourceCreateInfo info, uint32_t* pHandle);

void TRM_destroyResource(uint32_t handle);

void TRM_writeBuffer(uint32_t sizeInBytes, const void* pData, uint32_t handle);

void TRM_createComputePipeline(struct TRM_ComputePipelineCreateInfo info, uint32_t* pHandle);

void TRM_createGraphicsPipeline(struct TRM_GraphicsPipelineCreateInfo info, uint32_t* pHandle);

void TRM_destroyPipeline(uint32_t handle);

uint32_t TRM_getDrawIndirectCommandSize(void);

uint32_t TRM_getDrawIndexedIndirectCommandSize(void);

#endif
