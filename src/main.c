#include "GLFW/glfw3.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "trm_memory.h"
#include "trm_renderer.h"
#include "trm_maths.h"
#include "trm_containers.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TRM_WINDOW_WIDTH 500
#define TRM_WINDOW_HEIGHT 500

struct UniformBuffer
{
	struct TRM_Matrix4x4 projection;
	struct TRM_Matrix4x4 transformation;
};

struct Vertex
{
	float x;
	float y;
	float z;
};

static void TRM_readShader(const char* pPath, uint32_t* pSize, uint32_t** ppCode)
{
	FILE* pFile = fopen(pPath, "rb");
	if(pFile == NULL)
	{
		printf("could not open shader : %s\n", pPath);
		exit(EXIT_FAILURE);
	}

	fseek(pFile, 0, SEEK_END);
	*pSize = (uint32_t)ftell(pFile);
	rewind(pFile);

	TRM_Memory_allocate(sizeof(uint32_t) * (*pSize), (void**)ppCode);

	fread(*ppCode, 1, *pSize, pFile);

	fclose(pFile);
}

static void TRM_readImage(const char* pPath, uint32_t* pWidth, uint32_t* pHeight, uint32_t* pChannels, uint8_t** ppData)
{
	*ppData = (uint8_t*)stbi_load(pPath, (int32_t*)pWidth, (int32_t*)pHeight, (int32_t*)pChannels, 4);
}


int main(void)
{
	if(!glfwInit())
	{
		return 1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* pWindow = glfwCreateWindow(TRM_WINDOW_WIDTH, TRM_WINDOW_HEIGHT, "TRM", NULL, NULL);
	if(pWindow == NULL)
	{
		return 1;
	}

	TRM_Memory_start();
	TRM_Renderer_start(pWindow, TRM_WINDOW_WIDTH, TRM_WINDOW_HEIGHT);

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t channels = 0;
	uint8_t* pTextureData = NULL;

	TRM_readImage(PROJECT_ROOT "/assets/textures/concrete.jpg", &width, &height, &channels, &pTextureData);

	struct Vertex vertices[3] = {
		{-1.0f, -1.0f, 0.0f},
		{0.0f, 1.0f, 0.0f},
		{1.0f, -1.0f, 0.0f}
	};

	uint32_t colorImage = 0;
	uint32_t depthImage = 0;
	uint32_t vertexBuffer = 0; 
	uint32_t uniformBuffer = 0;
	uint32_t texture = 0;
	uint32_t bufferUpload = 0;

	VkFormat colorImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
	VkFormat depthImageFormat = VK_FORMAT_D32_SFLOAT;
	
	uint32_t drawPass = 0;
	uint32_t computePass = 0;

	{
		struct TRM_Renderer_ImageCreateInfo colorImageCreateInfo = {0};
		colorImageCreateInfo.width = TRM_WINDOW_WIDTH;
		colorImageCreateInfo.height = TRM_WINDOW_HEIGHT;
		colorImageCreateInfo.format = colorImageFormat;
		colorImageCreateInfo.usage = TRM_RENDERER_IMAGE_USAGE_COLOR_ATTACHMENT | TRM_RENDERER_IMAGE_USAGE_TRANSFER_SRC | TRM_RENDERER_IMAGE_USAGE_STORAGE;

		struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
		resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
		resourceCreateInfo.info.image = colorImageCreateInfo;

		TRM_Renderer_createResource(resourceCreateInfo, &colorImage);
	}

	{
		struct TRM_Renderer_ImageCreateInfo depthImageCreateInfo = {0};
		depthImageCreateInfo.width = TRM_WINDOW_WIDTH;
		depthImageCreateInfo.height = TRM_WINDOW_HEIGHT;
		depthImageCreateInfo.format = depthImageFormat;
		depthImageCreateInfo.usage = TRM_RENDERER_IMAGE_USAGE_DEPTH_ATTACHMENT;

		struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
		resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
		resourceCreateInfo.info.image = depthImageCreateInfo;

		TRM_Renderer_createResource(resourceCreateInfo, &depthImage);
	}

	{
		struct TRM_Renderer_BufferCreateInfo vertexBufferCreateInfo = {0};
		vertexBufferCreateInfo.sizeInBytes = sizeof(vertices);
		vertexBufferCreateInfo.hostVisible = true;
		vertexBufferCreateInfo.usage = TRM_RENDERER_BUFFER_USAGE_VERTEX;

		struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
		resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_BUFFER;
		resourceCreateInfo.info.buffer = vertexBufferCreateInfo;

		TRM_Renderer_createResource(resourceCreateInfo, &vertexBuffer);
	}

	{
		struct TRM_Renderer_BufferCreateInfo uniformBufferCreateInfo = {0};
		uniformBufferCreateInfo.sizeInBytes = sizeof(struct UniformBuffer);
		uniformBufferCreateInfo.hostVisible = true;
		uniformBufferCreateInfo.usage = TRM_RENDERER_BUFFER_USAGE_UNIFORM;

		struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
		resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_BUFFER;
		resourceCreateInfo.info.buffer = uniformBufferCreateInfo;

		TRM_Renderer_createResource(resourceCreateInfo, &uniformBuffer);
	}

	{
		struct TRM_Renderer_ImageCreateInfo textureCreateInfo = {0};
		textureCreateInfo.width = width;
		textureCreateInfo.height = height;
		textureCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		textureCreateInfo.usage = TRM_RENDERER_IMAGE_USAGE_TRANSFER_DST | TRM_RENDERER_IMAGE_USAGE_SAMPLED;

		struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
		resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
		resourceCreateInfo.info.image = textureCreateInfo;

		TRM_Renderer_createResource(resourceCreateInfo, &texture);
	}

	{
		struct TRM_Renderer_BufferCreateInfo bufferUploadCreateInfo = {0};
		bufferUploadCreateInfo.sizeInBytes = width * height * 4;
		bufferUploadCreateInfo.hostVisible = true;
		bufferUploadCreateInfo.usage = TRM_RENDERER_BUFFER_USAGE_TRANSFER_SRC;

		struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
		resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_BUFFER;
		resourceCreateInfo.info.buffer = bufferUploadCreateInfo;

		TRM_Renderer_createResource(resourceCreateInfo, &bufferUpload);
	}

	{
		uint32_t vertexCodeSize = 0;
		uint32_t* pVertexCode = NULL;
		TRM_readShader(PROJECT_ROOT "/assets/shaders/vertex.spv", &vertexCodeSize, &pVertexCode);


		uint32_t fragmentCodeSize = 0;
		uint32_t* pFragmentCode = NULL;
		TRM_readShader(PROJECT_ROOT "/assets/shaders/fragment.spv", &fragmentCodeSize, &pFragmentCode);

		struct TRM_Renderer_DescriptorInfo descriptorInfos[2];
		descriptorInfos[0].descriptorType = TRM_RENDERER_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorInfos[0].resourceAccessFlags = TRM_RENDERER_SHADER_ACCESS_FLAG_READ;

		descriptorInfos[1].descriptorType = TRM_RENDERER_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorInfos[1].resourceAccessFlags = TRM_RENDERER_SHADER_ACCESS_FLAG_READ;

		struct TRM_Renderer_VertexAttributeDescription vertexAttributeDescriptions[1];
		vertexAttributeDescriptions[0].shaderLocation = 0;
		vertexAttributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributeDescriptions[0].offset = 0;

		struct TRM_Renderer_DrawPassCreateInfo drawPassCreateInfo = {0};
		drawPassCreateInfo.vertexCodeSize = vertexCodeSize;
		drawPassCreateInfo.pVertexCode = pVertexCode;
		drawPassCreateInfo.fragmentCodeSize = fragmentCodeSize;
		drawPassCreateInfo.pFragmentCode = pFragmentCode;
		drawPassCreateInfo.colorOutputImageCount = 1;
		drawPassCreateInfo.pColorOutputImageFormats = &colorImageFormat;
		drawPassCreateInfo.vertexStride = sizeof(struct Vertex);
		drawPassCreateInfo.vertexAttributeDescriptionCount = sizeof(vertexAttributeDescriptions) / sizeof(vertexAttributeDescriptions[0]);
		drawPassCreateInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions;
		drawPassCreateInfo.descriptorInfoCount = sizeof(descriptorInfos) / sizeof(descriptorInfos[0]);
		drawPassCreateInfo.pDescriptorInfos = descriptorInfos;

		TRM_Renderer_createDrawPass(drawPassCreateInfo, &drawPass);
		
		TRM_Memory_deallocate(pVertexCode);
		TRM_Memory_deallocate(pFragmentCode);
	}

	{
		uint32_t computeCodeSize = 0;
		uint32_t* pComputeCode = NULL;
		TRM_readShader(PROJECT_ROOT "/assets/shaders/compute.spv", &computeCodeSize, &pComputeCode);

		struct TRM_Renderer_DescriptorInfo descriptorInfos[1];
		descriptorInfos->descriptorType = TRM_RENDERER_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorInfos->resourceAccessFlags = TRM_RENDERER_SHADER_ACCESS_FLAG_WRITE;

		struct TRM_Renderer_DispatchPassCreateInfo dispatchPassCreateInfo = {0};
		dispatchPassCreateInfo.codeSize = computeCodeSize;
		dispatchPassCreateInfo.pCode = pComputeCode;
		dispatchPassCreateInfo.descriptorInfoCount = sizeof(descriptorInfos) / sizeof(descriptorInfos[0]);
		dispatchPassCreateInfo.pDescriptorInfos = descriptorInfos;

		TRM_Renderer_createDispatchPass(dispatchPassCreateInfo, &computePass);

		TRM_Memory_deallocate(pComputeCode);
	}

	uint32_t framebufferWidth = TRM_WINDOW_WIDTH;
	uint32_t framebufferHeight = TRM_WINDOW_HEIGHT;
	uint32_t currentWindowWidth = TRM_WINDOW_WIDTH;
	uint32_t currentWindowHeight = TRM_WINDOW_HEIGHT;

	while(!glfwWindowShouldClose(pWindow))
	{
		glfwPollEvents();
		
		TRM_Renderer_beginFrame();

		glfwGetFramebufferSize(pWindow, (int*)&currentWindowWidth, (int*)&currentWindowHeight);
		if(framebufferWidth != currentWindowWidth || framebufferHeight != currentWindowHeight)
		{
			TRM_Renderer_destroyResource(colorImage);
			TRM_Renderer_destroyResource(depthImage);

			{
				struct TRM_Renderer_ImageCreateInfo colorImageCreateInfo = {0};
				colorImageCreateInfo.width = currentWindowWidth;
				colorImageCreateInfo.height = currentWindowHeight;
				colorImageCreateInfo.format = colorImageFormat;
				colorImageCreateInfo.usage = TRM_RENDERER_IMAGE_USAGE_COLOR_ATTACHMENT | TRM_RENDERER_IMAGE_USAGE_TRANSFER_SRC | TRM_RENDERER_IMAGE_USAGE_STORAGE;

				struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
				resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
				resourceCreateInfo.info.image = colorImageCreateInfo;

				TRM_Renderer_createResource(resourceCreateInfo, &colorImage);
			}

			{
				struct TRM_Renderer_ImageCreateInfo depthImageCreateInfo = {0};
				depthImageCreateInfo.width = currentWindowWidth;
				depthImageCreateInfo.height = currentWindowHeight;
				depthImageCreateInfo.format = depthImageFormat;
				depthImageCreateInfo.usage = TRM_RENDERER_IMAGE_USAGE_DEPTH_ATTACHMENT;

				struct TRM_Renderer_ResourceCreateInfo resourceCreateInfo = {0};
				resourceCreateInfo.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
				resourceCreateInfo.info.image = depthImageCreateInfo;

				TRM_Renderer_createResource(resourceCreateInfo, &depthImage);
			}
		}

		struct UniformBuffer uniformBufferData;

		struct TRM_Matrix4x4 rotation;
		struct TRM_Matrix4x4 translation;

		struct TRM_Vector3 tra;
		tra.x = 0.0f;
		tra.y = 0.0f;
		tra.z = 5.0f;

		static float toy = 0.0f;
		TRM_Matrix4x4_getProjection(1.2f, currentWindowWidth / (float)currentWindowHeight, &uniformBufferData.projection);
		TRM_Matrix4x4_transpose(&uniformBufferData.projection);
		TRM_Matrix4x4_getRotation(toy, 0.0f, 0.0f, &rotation);
		TRM_Matrix4x4_getTranslation(tra, &translation);
		TRM_Matrix4x4_multiplyWithMatrix4x4(translation, rotation, &uniformBufferData.transformation);
		TRM_Matrix4x4_transpose(&uniformBufferData.transformation);
		toy += 0.001f;

		TRM_Renderer_writeBuffer(sizeof(struct UniformBuffer), &uniformBufferData, uniformBuffer);
		
		TRM_Renderer_writeBuffer(sizeof(vertices), vertices, vertexBuffer);

		TRM_Renderer_writeBuffer(width * height * 4, pTextureData, bufferUpload);

		struct TRM_Renderer_PassInstance passInstances[5];

		{
			passInstances[0].type = TRM_RENDERER_PASS_TYPE_BUFFER_TO_IMAGE_COPY;
			passInstances[0].info.bufferToImageCopy.srcBuffer = bufferUpload;
			passInstances[0].info.bufferToImageCopy.dstImage = texture;
			passInstances[0].info.bufferToImageCopy.width = width;
			passInstances[0].info.bufferToImageCopy.height = height;
		}

		uint32_t drawBindings[2] = {
			uniformBuffer,
			texture
		};

		uint32_t colorOutputImage = colorImage;
		struct TRM_Renderer_ClearColor clearColor;
		clearColor.color[0] = cosf(toy);
		clearColor.color[1] = sinf(toy);
		clearColor.color[2] = 0.0f;
		clearColor.color[3] = 1.0f;

		{
			passInstances[1].type = TRM_RENDERER_PASS_TYPE_DRAW;
			passInstances[1].info.draw.pass = drawPass;
			passInstances[1].info.draw.width = currentWindowWidth;
			passInstances[1].info.draw.height = currentWindowHeight;
			passInstances[1].info.draw.vertexCount = 3;
			passInstances[1].info.draw.vertexBuffer = vertexBuffer;
			passInstances[1].info.draw.colorOutputImageCount = 1;
			passInstances[1].info.draw.pColorOutputImages = &colorOutputImage;
			passInstances[1].info.draw.pClearColors = &clearColor;
			passInstances[1].info.draw.depthOutputImage = depthImage;
			passInstances[1].info.draw.bindingCount = sizeof(drawBindings) / sizeof(drawBindings[0]);
			passInstances[1].info.draw.pBindings = drawBindings;
		}
		
		{
			passInstances[2].type = TRM_RENDERER_PASS_TYPE_DISPATCH;
			passInstances[2].info.dispatch.pass = computePass;
			passInstances[2].info.dispatch.groupCountX = (currentWindowWidth + 8 - 1) / 8;
			passInstances[2].info.dispatch.groupCountY = (currentWindowHeight + 8 - 1) / 8;
			passInstances[2].info.dispatch.groupCountZ = 1;
			passInstances[2].info.dispatch.bindingCount = 1;
			passInstances[2].info.dispatch.pBindings = &colorImage;
		}

		{
			passInstances[3].type = TRM_RENDERER_PASS_TYPE_BLIT;
			passInstances[3].info.blit.srcImage = colorImage;
			passInstances[3].info.blit.dstImage = TRM_RENDERER_SWAPCHAIN_IMAGE;
			passInstances[3].info.blit.srcWidth = currentWindowWidth;
			passInstances[3].info.blit.srcHeight = currentWindowHeight;
			passInstances[3].info.blit.dstWidth = currentWindowWidth;
			passInstances[3].info.blit.dstHeight = currentWindowHeight;
		}

		{
			passInstances[4].type = TRM_RENDERER_PASS_TYPE_PRESENT;
		}

		TRM_Renderer_endFrame(sizeof(passInstances) / sizeof(passInstances[0]), passInstances, currentWindowWidth, currentWindowHeight);
	}

	TRM_Renderer_destroyPass(drawPass);
	TRM_Renderer_destroyPass(computePass);

	stbi_image_free(pTextureData);

	TRM_Renderer_terminate();
	TRM_Memory_terminate();

	glfwDestroyWindow(pWindow);
	glfwTerminate();

	return 0;
}

