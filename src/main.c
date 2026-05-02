#include "GLFW/glfw3.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "trm_memory.h"
#include "trm_renderer.h"
#include "trm_maths.h"

#define TRM_WINDOW_WIDTH 1000
#define TRM_WINDOW_HEIGHT 1000

struct UniformBuffer
{
	struct TRM_Matrix4x4 projection;
	struct TRM_Matrix4x4 transformation;
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

	struct Vertex vertices[3] = {
		{-1.0f, -1.0f, 0.0f},
		{0.0f, 1.0f, 0.0f},
		{1.0f, -1.0f, 0.0f}
	};

	uint32_t colorImage = 0;
	uint32_t depthImage = 0;
	uint32_t vertexBuffer = 0;
	uint32_t uniformBuffer = 0;

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
		TRM_Renderer_createImage(colorImageCreateInfo, &colorImage);
	}

	{
		struct TRM_Renderer_ImageCreateInfo depthImageCreateInfo = {0};
		depthImageCreateInfo.width = TRM_WINDOW_WIDTH;
		depthImageCreateInfo.height = TRM_WINDOW_HEIGHT;
		depthImageCreateInfo.format = depthImageFormat;
		depthImageCreateInfo.usage = TRM_RENDERER_IMAGE_USAGE_DEPTH_ATTACHMENT;
		TRM_Renderer_createImage(depthImageCreateInfo, &depthImage);
	}

	{
		struct TRM_Renderer_BufferCreateInfo vertexBufferCreateInfo = {0};
		vertexBufferCreateInfo.sizeInBytes = sizeof(vertices);
		vertexBufferCreateInfo.hostVisible = true;
		vertexBufferCreateInfo.usage = TRM_RENDERER_BUFFER_USAGE_VERTEX;
		TRM_Renderer_createBuffer(vertexBufferCreateInfo, &vertexBuffer);
	}

	{
		struct TRM_Renderer_BufferCreateInfo uniformBufferCreateInfo = {0};
		uniformBufferCreateInfo.sizeInBytes = sizeof(struct UniformBuffer);
		uniformBufferCreateInfo.hostVisible = true;
		uniformBufferCreateInfo.usage = TRM_RENDERER_BUFFER_USAGE_UNIFORM;
		TRM_Renderer_createBuffer(uniformBufferCreateInfo, &uniformBuffer);
	}

	{
		uint32_t vertexCodeSize = 0;
		uint32_t* pVertexCode = NULL;
		TRM_readShader(PROJECT_ROOT "/assets/shaders/vertex.spv", &vertexCodeSize, &pVertexCode);


		uint32_t fragmentCodeSize = 0;
		uint32_t* pFragmentCode = NULL;
		TRM_readShader(PROJECT_ROOT "/assets/shaders/fragment.spv", &fragmentCodeSize, &pFragmentCode);

		struct TRM_Renderer_DescriptorInfo descriptorInfos[1];
		descriptorInfos->descriptorType = TRM_RENDERER_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorInfos->resourceAccessFlags = TRM_RENDERER_SHADER_ACCESS_FLAG_READ;

		struct TRM_Renderer_DrawPassCreateInfo drawPassCreateInfo = {0};
		drawPassCreateInfo.vertexCodeSize = vertexCodeSize;
		drawPassCreateInfo.pVertexCode = pVertexCode;
		drawPassCreateInfo.fragmentCodeSize = fragmentCodeSize;
		drawPassCreateInfo.pFragmentCode = pFragmentCode;
		drawPassCreateInfo.width = TRM_WINDOW_WIDTH;
		drawPassCreateInfo.height = TRM_WINDOW_HEIGHT;
		drawPassCreateInfo.descriptorInfoCount = sizeof(descriptorInfos) / sizeof(descriptorInfos[0]);
		drawPassCreateInfo.pDescriptorInfos = descriptorInfos;
		drawPassCreateInfo.colorImageFormat = colorImageFormat;
		drawPassCreateInfo.depthImageFormat = depthImageFormat;

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


	while(!glfwWindowShouldClose(pWindow))
	{
		glfwPollEvents();
		
		TRM_Renderer_beginFrame();

		struct UniformBuffer uniformBufferData;

		struct TRM_Matrix4x4 rotation;
		struct TRM_Matrix4x4 translation;

		struct TRM_Vector3 tra;
		tra.x = 0.0f;
		tra.y = 0.0f;
		tra.z = 5.0f;

		static float toy = 0.0f;
		TRM_Matrix4x4_getProjection(1.2f, 1.0f, &uniformBufferData.projection);
		TRM_Matrix4x4_transpose(&uniformBufferData.projection);
		TRM_Matrix4x4_getRotation(toy, 0.0f, 0.0f, &rotation);
		TRM_Matrix4x4_getTranslation(tra, &translation);
		TRM_Matrix4x4_multiplyWithMatrix4x4(translation, rotation, &uniformBufferData.transformation);
		TRM_Matrix4x4_transpose(&uniformBufferData.transformation);
		toy += 0.001f;

		TRM_Renderer_writeBuffer(sizeof(struct UniformBuffer), &uniformBufferData, uniformBuffer);
		
		TRM_Renderer_writeBuffer(sizeof(vertices), vertices, vertexBuffer);

		struct TRM_Renderer_PassInstance passInstances[4];

		{
			uint32_t binding = uniformBuffer;

			passInstances[0].type = TRM_RENDERER_PASS_TYPE_DRAW;
			passInstances[0].info.draw.pass = drawPass;
			passInstances[0].info.draw.bindingCount = 1;
			passInstances[0].info.draw.pBindings = &binding;
			passInstances[0].info.draw.vertexCount = 3;
			passInstances[0].info.draw.vertexBuffer = vertexBuffer;
			passInstances[0].info.draw.colorImage = colorImage;
			passInstances[0].info.draw.depthImage = depthImage;
			passInstances[0].info.draw.bindingCount = 1;
			passInstances[0].info.draw.pBindings = &binding;
		}
		
		{
			uint32_t binding = colorImage;

			passInstances[1].type = TRM_RENDERER_PASS_TYPE_DISPATCH;
			passInstances[1].info.dispatch.pass = computePass;
			passInstances[1].info.dispatch.groupCountX = (TRM_WINDOW_WIDTH + 8 - 1) / 8;
			passInstances[1].info.dispatch.groupCountY = (TRM_WINDOW_HEIGHT + 8 - 1) / 8;
			passInstances[1].info.dispatch.groupCountZ = 1;
			passInstances[1].info.dispatch.bindingCount = 1;
			passInstances[1].info.dispatch.pBindings = &binding;
		}

		
		{
			passInstances[2].type = TRM_RENDERER_PASS_TYPE_IMAGE_TO_IMAGE_COPY;
			passInstances[2].info.imageToImageCopy.srcImage = colorImage;
			passInstances[2].info.imageToImageCopy.dstImage = TRM_RENDERER_SWAPCHAIN_IMAGE;
			passInstances[2].info.imageToImageCopy.width = TRM_WINDOW_WIDTH;
			passInstances[2].info.imageToImageCopy.height = TRM_WINDOW_HEIGHT;
		}

		{
			passInstances[3].type = TRM_RENDERER_PASS_TYPE_PRESENT;
		}

		TRM_Renderer_endFrame(sizeof(passInstances) / sizeof(passInstances[0]), passInstances);
	}

	TRM_Renderer_destroyPass(drawPass);
	TRM_Renderer_destroyPass(computePass);

	TRM_Renderer_destroyResource(uniformBuffer);
	TRM_Renderer_destroyResource(vertexBuffer);
	TRM_Renderer_destroyResource(colorImage);
	TRM_Renderer_destroyResource(depthImage);

	TRM_Renderer_terminate();
	TRM_Memory_terminate();

	glfwDestroyWindow(pWindow);
	glfwTerminate();

	return 0;
}
