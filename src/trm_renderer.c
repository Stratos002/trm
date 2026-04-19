#include "trm_renderer.h"
#include "trm_memory.h"
#include "trm_containers.h"
#include "trm_maths.h"

#define VOLK_IMPLEMENTATION
#include "volk.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define TRM_RENDERER_FRAME_COUNT 3
#define TRM_RENDERER_MAX_RESOURCE_COUNT 64

struct TRM_Renderer_ResourceInfoBuffer
{
	VkBuffer buffer;
	VkDeviceMemory memory;
};

struct TRM_Renderer_ResourceInfoImage
{
	VkImage image;
	VkImageView imageView;
	VkDeviceMemory memory;
	VkImageLayout layout;
	VkImageAspectFlags aspectFlags;
	bool swapchainImage;
};

enum TRM_Renderer_ResourceType
{
	TRM_RENDERER_RESOURCE_TYPE_BUFFER,
	TRM_RENDERER_RESOURCE_TYPE_IMAGE
};

struct TRM_Renderer_Resource
{
	enum TRM_Renderer_ResourceType type;
	VkPipelineStageFlags stageFlags;
	VkAccessFlags accessFlags;
	union
	{
		struct TRM_Renderer_ResourceInfoBuffer buffer;
		struct TRM_Renderer_ResourceInfoImage image;
	} info;
};

enum TRM_Renderer_PassType
{
	TRM_RENDERER_PASS_TYPE_DISPATCH,
	TRM_RENDERER_PASS_TYPE_DRAW,
	TRM_RENDERER_PASS_TYPE_IMAGE_COPY,
	TRM_RENDERER_PASS_TYPE_PRESENT
};

struct TRM_Renderer_PassInfoDispatch
{
	uint32_t groupCountX;
	uint32_t groupCountY;
	uint32_t groupCountZ;
	VkPipelineLayout pipelineLayout;
	uint32_t descriptorSetCount;
	VkDescriptorSet* pDescriptorSets;
	VkPipeline pipeline;
};

struct TRM_Renderer_PassInfoDraw
{
	uint32_t width;
	uint32_t height;
	uint32_t vertexCount;
	VkRenderPass renderPass;
	VkFramebuffer framebuffer;
	VkPipelineLayout pipelineLayout;
	uint32_t descriptorSetCount;
	VkDescriptorSet* pDescriptorSets;
	VkPipeline pipeline;
	uint32_t attachmentCount;
	VkClearValue* pClearValues;
};

struct TRM_Renderer_PassInfoImageCopy
{
	VkImageCopy imageCopy;
};

struct TRM_Renderer_ResourceAccess
{
	uint32_t resourceIndex;
	VkPipelineStageFlags stageFlags;
	VkAccessFlags accessFlags;
	VkImageLayout layout;
};

struct TRM_Renderer_Pass
{
	enum TRM_Renderer_PassType type;
	uint32_t inputCount;
	uint32_t outputCount;
	struct TRM_Renderer_ResourceAccess* pInputs;
	struct TRM_Renderer_ResourceAccess* pOutputs;
	union
	{
		struct TRM_Renderer_PassInfoDispatch dispatch;
		struct TRM_Renderer_PassInfoDraw draw; // input0 : vertex buffer / outputs = attachments
		struct TRM_Renderer_PassInfoImageCopy imageCopy;
	} info;
};

struct TRM_Renderer_FrameInfo
{
	VkCommandBuffer commandBuffer;
	VkFence commandBufferExecutedFence;
	VkSemaphore imageAvailableSemaphore;
	VkDescriptorSet descriptorSet;
	uint32_t vertexBufferIndex;
	uint32_t uniformBufferIndex;
};

struct TRM_Renderer_SwapchainImageInfo
{
	VkSemaphore imageRenderedSemaphore;
	uint32_t colorImageIndex;
	uint32_t depthImageIndex;
	VkFramebuffer framebuffer;
};

struct TRM_Renderer_State
{
	const VkAllocationCallbacks* pAllocator;
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkSurfaceKHR surface;
	uint32_t queueFamilyIndex;
	VkDevice device;
	VkQueue queue;
	VkCommandPool commandPool;
	VkDescriptorPool descriptorPool;
	VkSwapchainKHR swapchain;
	VkFormat swapchainFormat;
	uint32_t swapchainWidth;
	uint32_t swapchainHeight;
	uint32_t swapchainImageCount;
	struct TRM_Renderer_SwapchainImageInfo* pSwapchainImageInfos;
	struct TRM_Renderer_FrameInfo* pFrameInfos;
	uint32_t frameIndex;
	struct TRM_Arena resources;
	VkDescriptorSetLayout descriptorSetLayout;
	VkRenderPass renderPass;
	VkPipeline graphicsPipeline;
	VkPipelineLayout graphicsPipelineLayout;
};

static struct TRM_Renderer_State* pState = NULL;

static void TRM_Renderer_createInstance(const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
	VkApplicationInfo applicationInfo = {0};
	applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	applicationInfo.pNext = NULL;
	applicationInfo.pApplicationName = "TRM";
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pEngineName = "TRM";
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.apiVersion = VK_MAKE_API_VERSION(1, 0, 0, 0);

	const char* pValidationLayerName = "VK_LAYER_KHRONOS_validation";

	uint32_t GLFWInstanceExtensionCount = 0;
	const char** ppGLFWInstanceExtensionNames = glfwGetRequiredInstanceExtensions(&GLFWInstanceExtensionCount);

	VkInstanceCreateInfo instanceCreateInfo = {0};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.flags = 0;
	instanceCreateInfo.pApplicationInfo = &applicationInfo;
	instanceCreateInfo.enabledLayerCount = 1;
	instanceCreateInfo.ppEnabledLayerNames = &pValidationLayerName;
	instanceCreateInfo.enabledExtensionCount = GLFWInstanceExtensionCount;
	instanceCreateInfo.ppEnabledExtensionNames = ppGLFWInstanceExtensionNames;

	if(vkCreateInstance(&instanceCreateInfo, pAllocator, pInstance) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_pickPhysicalDevice(VkInstance instance, VkPhysicalDevice* pPhysicalDevice)
{
	uint32_t physicalDeviceCount = 0;
	VkPhysicalDevice* pPhysicalDevices = NULL;

	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, NULL);

	if(physicalDeviceCount == 0)
	{
		exit(EXIT_FAILURE);
	}

	TRM_Memory_allocate(sizeof(VkPhysicalDevice) * physicalDeviceCount, (void**)&pPhysicalDevices);

	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, pPhysicalDevices);

	for(uint32_t physicalDeviceIndex = 0; physicalDeviceIndex < physicalDeviceCount; ++physicalDeviceIndex)
	{
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(pPhysicalDevices[physicalDeviceIndex], &physicalDeviceProperties);

		if(physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			printf("discrete GPU found : %s\n", physicalDeviceProperties.deviceName);

			*pPhysicalDevice = pPhysicalDevices[physicalDeviceIndex];
			TRM_Memory_deallocate(pPhysicalDevices);
			return;
		}
	}

	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(pPhysicalDevices[0], &physicalDeviceProperties);
	*pPhysicalDevice = pPhysicalDevices[0];
	TRM_Memory_deallocate(pPhysicalDevices);

	printf("no discrete GPU found, fallback GPU %s\n", physicalDeviceProperties.deviceName);
}

static void TRM_Renderer_createSurface(
	const VkAllocationCallbacks* pAllocator,
	VkInstance instance,
	GLFWwindow* pWindow,
	VkSurfaceKHR* pSurface)
{
	if(glfwCreateWindowSurface(instance, pWindow, pAllocator, pSurface) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_findQueueFamilyIndex(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pQueueFamilyIndex)
{
	uint32_t queueFamilyPropertyCount = 0;
	VkQueueFamilyProperties* pQueueFamilyProperties = NULL;

	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, NULL);
	TRM_Memory_allocate(sizeof(VkQueueFamilyProperties) * queueFamilyPropertyCount, (void**)&pQueueFamilyProperties);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, pQueueFamilyProperties);

	for(uint32_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilyPropertyCount; ++queueFamilyIndex)
	{
		VkQueueFlags queueFlags = (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT);

		VkBool32 presentationSupported = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamilyIndex, surface, &presentationSupported);
		if((pQueueFamilyProperties[queueFamilyIndex].queueFlags & queueFlags) == queueFlags && presentationSupported)
		{
			*pQueueFamilyIndex = queueFamilyIndex;
			TRM_Memory_deallocate(pQueueFamilyProperties);
			return;
		}
	}

	TRM_Memory_deallocate(pQueueFamilyProperties);
	exit(EXIT_FAILURE);
}

static void TRM_Renderer_createDevice(
	const VkAllocationCallbacks* pAllocator,
	VkPhysicalDevice physicalDevice,
	uint32_t queueFamilyIndex,
	VkDevice* pDevice)
{
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo deviceQueueCreateInfo = {0};
	deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	deviceQueueCreateInfo.pNext = NULL;
	deviceQueueCreateInfo.flags = 0;
	deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
	deviceQueueCreateInfo.queueCount = 1;
	deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

	const char* ppEnabledExtensionNames[] = {
		"VK_KHR_swapchain"
	};

	VkDeviceCreateInfo deviceCreateInfo = {0};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext = NULL;
	deviceCreateInfo.flags = 0;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
	deviceCreateInfo.enabledLayerCount = 0;
	deviceCreateInfo.ppEnabledLayerNames = NULL;
	deviceCreateInfo.enabledExtensionCount = sizeof(ppEnabledExtensionNames) / sizeof(ppEnabledExtensionNames[0]);
	deviceCreateInfo.ppEnabledExtensionNames = ppEnabledExtensionNames;
	deviceCreateInfo.pEnabledFeatures = NULL;

	if(vkCreateDevice(physicalDevice, &deviceCreateInfo, pAllocator, pDevice) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createSwapchain(
	const VkAllocationCallbacks* pAllocator,
	VkPhysicalDevice physicalDevice,
	VkDevice device,
	VkSurfaceKHR surface,
	uint32_t wantedWidth,
	uint32_t wantedHeight,
	bool vsync,
	uint32_t queueFamilyIndex,
	VkSwapchainKHR* pSwapchain, 
	VkFormat* pFormat, 
	uint32_t* pWidth,
	uint32_t* pHeight)
{
	VkSurfaceCapabilitiesKHR surfaceCapabilities = {0};
	if(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	// query image count
	uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
	if(surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount)
	{
		imageCount = surfaceCapabilities.maxImageCount;
	}

	// query extent
	if(surfaceCapabilities.currentExtent.width != UINT32_MAX)
	{
		*pWidth = surfaceCapabilities.currentExtent.width;
		*pHeight = surfaceCapabilities.currentExtent.height;
	}
	else
	{
		*pWidth = wantedWidth;
		if(*pWidth < surfaceCapabilities.minImageExtent.width)
		{
			*pWidth = surfaceCapabilities.minImageExtent.width;
		}
		if(*pWidth > surfaceCapabilities.maxImageExtent.width)
		{
			*pWidth = surfaceCapabilities.maxImageExtent.width;
		}
		
		*pHeight = wantedHeight;
		if(*pHeight < surfaceCapabilities.minImageExtent.height)
		{
			*pHeight = surfaceCapabilities.minImageExtent.height;
		}
		if(*pHeight > surfaceCapabilities.maxImageExtent.height)
		{
			*pHeight = surfaceCapabilities.maxImageExtent.height;
		}
	}

	// query format
	uint32_t formatCount = 0;
	VkSurfaceFormatKHR* pSurfaceFormats = NULL;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, NULL);
	TRM_Memory_allocate(sizeof(VkSurfaceFormatKHR) * formatCount, (void**)&pSurfaceFormats);
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, pSurfaceFormats);

	if(formatCount == 0)
	{
		exit(EXIT_FAILURE);
	}

	VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	bool formatFound = false;
	for(uint32_t formatIndex = 0; formatIndex < formatCount; ++formatIndex)
	{
		if(pSurfaceFormats[formatIndex].format == VK_FORMAT_B8G8R8A8_SRGB && 
			pSurfaceFormats[formatIndex].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			*pFormat = VK_FORMAT_B8G8R8A8_SRGB;
			colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
			formatFound = true;
			break;
		}
	}

	if(!formatFound)
	{
		exit(EXIT_FAILURE);
	}

	*pFormat = pSurfaceFormats[0].format;
	
	TRM_Memory_deallocate((void*)pSurfaceFormats);

	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	if(!vsync)
	{
		// query present mode
		uint32_t presentModeCount = 0;
		VkPresentModeKHR* pPresentModes = NULL;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, NULL);
		TRM_Memory_allocate(sizeof(VkPresentModeKHR) * presentModeCount, (void**)&pPresentModes);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, pPresentModes);

		for(uint32_t presentModeIndex = 0; presentModeIndex < presentModeCount; ++presentModeIndex)
		{
			if(pPresentModes[presentModeIndex] == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
			}
		}

		TRM_Memory_deallocate((void*)pPresentModes);
	}

	VkSwapchainCreateInfoKHR swapchainCreateInfo = {0};
	swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainCreateInfo.pNext = NULL;
	swapchainCreateInfo.flags = 0;
	swapchainCreateInfo.surface = surface;
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = *pFormat;
	swapchainCreateInfo.imageColorSpace = colorSpace;
	swapchainCreateInfo.imageExtent.width = *pWidth;
	swapchainCreateInfo.imageExtent.height = *pHeight;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainCreateInfo.queueFamilyIndexCount = 1;
	swapchainCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
	swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainCreateInfo.presentMode = presentMode;
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

	if(vkCreateSwapchainKHR(device, &swapchainCreateInfo, pAllocator, pSwapchain) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_findMemoryTypeIndex(
	VkPhysicalDevice physicalDevice, 
	uint32_t compatibleMemoryTypeBits, 
	VkMemoryPropertyFlags memoryPropertyFlags, 
	uint32_t* pMemoryTypeIndex)
{
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	for(uint32_t availableMemoryTypeIndex = 0; availableMemoryTypeIndex < memoryProperties.memoryTypeCount; ++availableMemoryTypeIndex)
	{
		if((compatibleMemoryTypeBits & (1 << availableMemoryTypeIndex)) &&
			(memoryProperties.memoryTypes[availableMemoryTypeIndex].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags)
		{
			*pMemoryTypeIndex = availableMemoryTypeIndex;
			return;
		}
	}

	exit(EXIT_FAILURE);
}

static void TRM_Renderer_allocateMemoryForBuffer(
	const VkAllocationCallbacks* pAllocator,
	VkPhysicalDevice physicalDevice,
	VkDevice device, 
	VkBuffer buffer,
	VkMemoryPropertyFlags memoryPropertyFlags,
	VkDeviceMemory* pMemory)
{
	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

	uint32_t memoryTypeIndex = 0;
	TRM_Renderer_findMemoryTypeIndex(physicalDevice, memoryRequirements.memoryTypeBits, memoryPropertyFlags, &memoryTypeIndex);

	VkMemoryAllocateInfo memoryAllocateInfo = {0};
	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.pNext = NULL;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

	if(vkAllocateMemory(device, &memoryAllocateInfo, pAllocator, pMemory) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_allocateMemoryForImage(
	const VkAllocationCallbacks* pAllocator,
	VkPhysicalDevice physicalDevice,
	VkDevice device,
	VkImage image,
	VkMemoryPropertyFlags memoryPropertyFlags,
	VkDeviceMemory* pMemory)
{
	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, image, &memoryRequirements);

	uint32_t memoryTypeIndex = 0;
	TRM_Renderer_findMemoryTypeIndex(physicalDevice, memoryRequirements.memoryTypeBits, memoryPropertyFlags, &memoryTypeIndex);

	VkMemoryAllocateInfo memoryAllocateInfo = {0};
	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.pNext = NULL;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

	if(vkAllocateMemory(device, &memoryAllocateInfo, pAllocator, pMemory) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createBuffer(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkDeviceSize size,
	VkBufferUsageFlags usage,
	uint32_t queueFamilyIndex,
	VkBuffer* pBuffer)
{
	VkBufferCreateInfo bufferCreateInfo = {0};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.pNext = NULL;
	bufferCreateInfo.flags = 0;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usage;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.queueFamilyIndexCount = 1;
	bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	if(vkCreateBuffer(device, &bufferCreateInfo, pAllocator, pBuffer) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createImage(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	uint32_t width,
	uint32_t height,
	VkFormat format,
	VkImageLayout layout,
	VkImageUsageFlags usage,
	uint32_t queueFamilyIndex,
	VkImage* pImage)
{
	VkImageCreateInfo imageCreateInfo = {0};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.pNext = NULL;
	imageCreateInfo.flags = 0;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.format = format;
	imageCreateInfo.extent.width = width;
	imageCreateInfo.extent.height = height;
	imageCreateInfo.extent.depth = 1;
	imageCreateInfo.mipLevels = 1;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCreateInfo.usage = usage;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.queueFamilyIndexCount = 1;
	imageCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
	imageCreateInfo.initialLayout = layout;

	if(vkCreateImage(device, &imageCreateInfo, pAllocator, pImage) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createImageView(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkImage image,
	VkFormat format,
	VkImageAspectFlags imageAspect,
	VkImageView* pImageView)
{
	VkImageViewCreateInfo imageViewCreateInfo = {0};
	imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCreateInfo.pNext = NULL;
	imageViewCreateInfo.flags = 0;
	imageViewCreateInfo.image = image;
	imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCreateInfo.format = format;
	imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	imageViewCreateInfo.subresourceRange.aspectMask = imageAspect;
	imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	imageViewCreateInfo.subresourceRange.levelCount = 1;
	imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	imageViewCreateInfo.subresourceRange.layerCount = 1;

	if(vkCreateImageView(device, &imageViewCreateInfo, pAllocator, pImageView) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createCommandPool(
	const VkAllocationCallbacks* pAllocator, 
	VkDevice device, 
	uint32_t queueFamilyIndex, 
	VkCommandPool* pCommandPool)
{
	VkCommandPoolCreateInfo commandPoolCreateInfo = {0};
	commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolCreateInfo.pNext = NULL;
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;

	if(vkCreateCommandPool(device, &commandPoolCreateInfo, pAllocator, pCommandPool) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createFence(const VkAllocationCallbacks* pAllocator, VkDevice device, VkFence* pFence)
{
	VkFenceCreateInfo fenceCreateInfo = {0};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.pNext = NULL;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if(vkCreateFence(device, &fenceCreateInfo, pAllocator, pFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createSemaphore(const VkAllocationCallbacks* pAllocator, VkDevice device, VkSemaphore* pSemaphore)
{
	VkSemaphoreCreateInfo semaphoreCreateInfo = {0};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = NULL;
	semaphoreCreateInfo.flags = 0;

	if(vkCreateSemaphore(device, &semaphoreCreateInfo, pAllocator, pSemaphore) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_allocateCommandBuffer(VkCommandPool commandPool, VkDevice device, VkCommandBuffer* pCommandBuffer)
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = {0};
	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.pNext = NULL;
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;

	if(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, pCommandBuffer) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createDescriptorPool(const VkAllocationCallbacks* pAllocator, VkDevice device, VkDescriptorPool* pDescriptorPool)
{
	VkDescriptorPoolSize uniformBufferDescriptorPoolSize = {0};
	uniformBufferDescriptorPoolSize.descriptorCount = 10;
	uniformBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

	VkDescriptorPoolSize storageBufferDescriptorPoolSize = {0};
	storageBufferDescriptorPoolSize.descriptorCount = 10;
	storageBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

	VkDescriptorPoolSize storageImageDescriptorPoolSize = {0};
	storageImageDescriptorPoolSize.descriptorCount = 10;
	storageImageDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

	VkDescriptorPoolSize descriptorPoolSizes[] = {
		uniformBufferDescriptorPoolSize,
		storageBufferDescriptorPoolSize,
		storageImageDescriptorPoolSize
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {0};
	descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolCreateInfo.pNext = NULL;
	descriptorPoolCreateInfo.flags = 0;
	descriptorPoolCreateInfo.maxSets = 10;
	descriptorPoolCreateInfo.poolSizeCount = sizeof(descriptorPoolSizes) / sizeof(descriptorPoolSizes[0]);
	descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes;

	if(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, pAllocator, pDescriptorPool) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_allocateDescriptorSet(
	VkDevice device, 
	VkDescriptorPool descriptorPool, 
	VkDescriptorSetLayout descriptorSetLayout, 
	VkDescriptorSet* pDescriptorSet)
{
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {0};
	descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetAllocateInfo.pNext = NULL;
	descriptorSetAllocateInfo.descriptorPool = descriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = 1;
	descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

	if(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, pDescriptorSet) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createDescriptorSetLayout(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	uint32_t bindingCount,
	const VkDescriptorSetLayoutBinding* pBindings,
	VkDescriptorSetLayout* pDescriptorSetLayout)
{
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};
	descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorSetLayoutCreateInfo.pNext = NULL;
	descriptorSetLayoutCreateInfo.flags = 0;
	descriptorSetLayoutCreateInfo.bindingCount = bindingCount;
	descriptorSetLayoutCreateInfo.pBindings = pBindings;

	if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, pAllocator, pDescriptorSetLayout) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createPipelineLayout(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	uint32_t descriptorSetLayoutCount,
	const VkDescriptorSetLayout* pDescriptorSetLayouts,
	VkPipelineLayout* pPipelineLayout)
{
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {0};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.pNext = NULL;
	pipelineLayoutCreateInfo.flags = 0;
	pipelineLayoutCreateInfo.setLayoutCount = descriptorSetLayoutCount;
	pipelineLayoutCreateInfo.pSetLayouts = pDescriptorSetLayouts;
	pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	pipelineLayoutCreateInfo.pPushConstantRanges = NULL;

	if(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, pAllocator, pPipelineLayout) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createShaderModule(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	size_t codeSize, 
	const uint32_t* pCode, 
	VkShaderModule* pShaderModule)
{
	VkShaderModuleCreateInfo shaderModuleCreateInfo = {0};
	shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModuleCreateInfo.pNext = NULL;
	shaderModuleCreateInfo.flags = 0;
	shaderModuleCreateInfo.codeSize = codeSize;
	shaderModuleCreateInfo.pCode = pCode;

	if(vkCreateShaderModule(device, &shaderModuleCreateInfo, pAllocator, pShaderModule) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

/*
static void TRM_Renderer_createComputePipeline(
	const VkAllocationCallbacks* pAllocator, 
	VkDevice device, 
	VkShaderModule shaderModule, 
	VkPipelineLayout pipelineLayout, 
	VkPipeline* pComputePipeline)
{
	VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {0};
	shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStageCreateInfo.pNext = NULL;
	shaderStageCreateInfo.flags = 0;
	shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStageCreateInfo.module = shaderModule;
	shaderStageCreateInfo.pName = "main";
	shaderStageCreateInfo.pSpecializationInfo = NULL;

	VkComputePipelineCreateInfo computePipelineCreateInfo = {0};
	computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCreateInfo.pNext = VK_NULL_HANDLE;
	computePipelineCreateInfo.flags = 0;
	computePipelineCreateInfo.stage = shaderStageCreateInfo;
	computePipelineCreateInfo.layout = pipelineLayout;
	computePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	computePipelineCreateInfo.basePipelineIndex = 0;

	if(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, pAllocator, pComputePipeline) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}
*/

static void TRM_Renderer_createRenderPass(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device, 
	uint32_t attachmentDescriptionCount,
	VkAttachmentDescription* pAttachmentDescriptions,
	VkSubpassDescription subpassDescription,
	VkRenderPass* pRenderPass)
{
	VkRenderPassCreateInfo renderPassCreateInfo = {0};
	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.pNext = NULL;
	renderPassCreateInfo.flags = 0;
	renderPassCreateInfo.attachmentCount = attachmentDescriptionCount;
	renderPassCreateInfo.pAttachments = pAttachmentDescriptions;
	renderPassCreateInfo.subpassCount = 1;
	renderPassCreateInfo.pSubpasses = &subpassDescription;
	renderPassCreateInfo.dependencyCount = 0;
	renderPassCreateInfo.pDependencies = NULL;

	if(vkCreateRenderPass(device, &renderPassCreateInfo, pAllocator, pRenderPass) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createFramebuffer(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkRenderPass renderPass,
	uint32_t attachmentCount,
	VkImageView* pAttachments,
	uint32_t width,
	uint32_t height,
	VkFramebuffer* pFramebuffer)
{
	VkFramebufferCreateInfo framebufferCreateInfo = {0};
	framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebufferCreateInfo.pNext = NULL;
	framebufferCreateInfo.flags = 0;
	framebufferCreateInfo.renderPass = renderPass;
	framebufferCreateInfo.attachmentCount = attachmentCount;
	framebufferCreateInfo.pAttachments = pAttachments;
	framebufferCreateInfo.width = width;
	framebufferCreateInfo.height = height;
	framebufferCreateInfo.layers = 1;

	if(vkCreateFramebuffer(device, &framebufferCreateInfo, pAllocator, pFramebuffer) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createGraphicsPipeline(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkShaderModule vertexShaderModule,
	VkShaderModule fragmentShaderModule,
	VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo,
	VkPipelineLayout pipelineLayout,
	VkRenderPass renderPass,
	uint32_t viewportWidth,
	uint32_t viewportHeight,
	VkPipeline* pGraphicsPipeline)
{
	VkPipelineShaderStageCreateInfo shaderStageCreateInfos[2];
	shaderStageCreateInfos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStageCreateInfos[0].pNext = NULL;
	shaderStageCreateInfos[0].flags = 0;
	shaderStageCreateInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderStageCreateInfos[0].module = vertexShaderModule;
	shaderStageCreateInfos[0].pName = "main";
	shaderStageCreateInfos[0].pSpecializationInfo = NULL;

	shaderStageCreateInfos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStageCreateInfos[1].pNext = NULL;
	shaderStageCreateInfos[1].flags = 0;
	shaderStageCreateInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderStageCreateInfos[1].module = fragmentShaderModule;
	shaderStageCreateInfos[1].pName = "main";
	shaderStageCreateInfos[1].pSpecializationInfo = NULL;

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo = {0};
	inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyStateCreateInfo.pNext = NULL;
	inputAssemblyStateCreateInfo.flags = 0;
	inputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {0};
	viewport.x = 0;
	viewport.y = 0;
	viewport.width = (float)viewportWidth;
	viewport.height = (float)viewportHeight;
	viewport.minDepth = 0;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {0};
	scissor.offset.x = 0;
	scissor.offset.y = 0;
	scissor.extent.width = viewportWidth;
	scissor.extent.height = viewportHeight;

	VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {0};
	viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateCreateInfo.pNext = NULL;
	viewportStateCreateInfo.flags = 0;
	viewportStateCreateInfo.viewportCount = 1;
	viewportStateCreateInfo.pViewports = &viewport;
	viewportStateCreateInfo.scissorCount = 1;
	viewportStateCreateInfo.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo = {0};
	rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationStateCreateInfo.pNext = NULL;
	rasterizationStateCreateInfo.flags = 0;
	rasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
	rasterizationStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
	rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
	rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
	rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;
	rasterizationStateCreateInfo.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo = {0};
	multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleStateCreateInfo.pNext = NULL;
	multisampleStateCreateInfo.flags = 0;
	multisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampleStateCreateInfo.sampleShadingEnable = VK_FALSE;
	multisampleStateCreateInfo.minSampleShading = 0.0f;
	multisampleStateCreateInfo.pSampleMask = NULL;
	multisampleStateCreateInfo.alphaToCoverageEnable = VK_FALSE;
	multisampleStateCreateInfo.alphaToOneEnable = VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo = {0};
	depthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilStateCreateInfo.pNext = NULL;
	depthStencilStateCreateInfo.flags = 0;
	depthStencilStateCreateInfo.depthTestEnable = VK_TRUE;
	depthStencilStateCreateInfo.depthWriteEnable = VK_TRUE;
	depthStencilStateCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencilStateCreateInfo.depthBoundsTestEnable = VK_FALSE;
	depthStencilStateCreateInfo.stencilTestEnable = VK_FALSE;
	depthStencilStateCreateInfo.minDepthBounds = 0.0f;
	depthStencilStateCreateInfo.maxDepthBounds = 1.0f;

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {0};
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT |
		VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT |
		VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo colorBlendStateInfo = {0};
	colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendStateInfo.pNext = NULL;
	colorBlendStateInfo.flags = 0;
	colorBlendStateInfo.logicOpEnable = VK_FALSE;
	colorBlendStateInfo.logicOp = VK_LOGIC_OP_COPY;
	colorBlendStateInfo.attachmentCount = 1;
	colorBlendStateInfo.pAttachments = &colorBlendAttachment;
	colorBlendStateInfo.blendConstants[0] = 0.0f;
	colorBlendStateInfo.blendConstants[1] = 0.0f;
	colorBlendStateInfo.blendConstants[2] = 0.0f;
	colorBlendStateInfo.blendConstants[3] = 0.0f;
	
	VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo = {0};
	graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	graphicsPipelineCreateInfo.pNext = NULL;
	graphicsPipelineCreateInfo.flags = 0;
	graphicsPipelineCreateInfo.stageCount = sizeof(shaderStageCreateInfos) / sizeof(shaderStageCreateInfos[0]);
	graphicsPipelineCreateInfo.pStages = shaderStageCreateInfos;
	graphicsPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
	graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
	graphicsPipelineCreateInfo.pTessellationState = NULL;
	graphicsPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
	graphicsPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
	graphicsPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
	graphicsPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
	graphicsPipelineCreateInfo.pColorBlendState = &colorBlendStateInfo;
	graphicsPipelineCreateInfo.pDynamicState = NULL;
	graphicsPipelineCreateInfo.layout = pipelineLayout;
	graphicsPipelineCreateInfo.renderPass = renderPass;
	graphicsPipelineCreateInfo.subpass = 0;
	graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	graphicsPipelineCreateInfo.basePipelineIndex = 0;

	if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, pAllocator, pGraphicsPipeline) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

// todo : move this
static void TRM_Renderer_readShader(const char* pPath, uint32_t* pSize, uint32_t** ppCode)
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

// refacto
// draw indexed
// draw indirect
// samplers
// refine API design
// validation
// reordering

static void TRM_Renderer_fillCommandBuffer(
	struct TRM_Renderer_Resource* pResources, 
	uint32_t passCount,
	struct TRM_Renderer_Pass* pPasses,
	VkCommandBuffer commandBuffer)
{
	vkResetCommandBuffer(commandBuffer, 0);

	VkCommandBufferBeginInfo commandBufferBeginInfo = {0};
	commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	commandBufferBeginInfo.pNext = NULL;
	commandBufferBeginInfo.flags = 0;
	commandBufferBeginInfo.pInheritanceInfo = NULL;

	if(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	for(uint32_t passIndex = 0; passIndex < passCount; ++passIndex)
	{
		uint32_t needBarrier = 0;

		VkPipelineStageFlags srcStageFlags = 0;
		VkPipelineStageFlags dstStageFlags = 0;
		struct TRM_DynamicArray bufferMemoryBarriers;
		struct TRM_DynamicArray imageMemoryBarriers;

		TRM_DynamicArray_create(sizeof(VkBufferMemoryBarrier), &bufferMemoryBarriers);
		TRM_DynamicArray_create(sizeof(VkImageMemoryBarrier), &imageMemoryBarriers);

		// read after write (memory) + layout
		for(uint32_t inputIndex = 0; inputIndex < pPasses[passIndex].inputCount; ++inputIndex)
		{
			const struct TRM_Renderer_ResourceAccess currentResourceAccess = pPasses[passIndex].pInputs[inputIndex];
			const uint32_t resourceIndex = currentResourceAccess.resourceIndex;
			const struct TRM_Renderer_Resource* pResourceState = &pResources[resourceIndex];

			const uint32_t lastWasWrite = (pResourceState->accessFlags & (
				VK_ACCESS_SHADER_WRITE_BIT |
				VK_ACCESS_TRANSFER_WRITE_BIT |
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));

			uint32_t layoutChanged = 0;
			if(pResourceState->type == TRM_RENDERER_RESOURCE_TYPE_IMAGE)
			{
				layoutChanged = (pResourceState->info.image.layout != currentResourceAccess.layout);
			}

			if(lastWasWrite || layoutChanged)
			{
				needBarrier = 1;

				srcStageFlags |= pResourceState->stageFlags;
				dstStageFlags |= currentResourceAccess.stageFlags;

				if(pResourceState->type == TRM_RENDERER_RESOURCE_TYPE_BUFFER)
				{
					VkBufferMemoryBarrier bufferMemoryBarrier = {0};
					bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
					bufferMemoryBarrier.pNext = NULL;
					bufferMemoryBarrier.srcAccessMask = pResourceState->accessFlags;
					bufferMemoryBarrier.dstAccessMask = currentResourceAccess.accessFlags;
					bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bufferMemoryBarrier.buffer = pResourceState->info.buffer.buffer;
					bufferMemoryBarrier.offset = 0;
					bufferMemoryBarrier.size = VK_WHOLE_SIZE;

					TRM_DynamicArray_push(&bufferMemoryBarrier, &bufferMemoryBarriers);
				}
				else if(pResourceState->type == TRM_RENDERER_RESOURCE_TYPE_IMAGE)
				{
					VkImageMemoryBarrier imageMemoryBarrier = {0};
					imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					imageMemoryBarrier.pNext = NULL;
					imageMemoryBarrier.srcAccessMask = pResourceState->accessFlags;
					imageMemoryBarrier.dstAccessMask = currentResourceAccess.accessFlags;
					imageMemoryBarrier.oldLayout = pResourceState->info.image.layout;
					imageMemoryBarrier.newLayout = currentResourceAccess.layout;
					imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imageMemoryBarrier.image = pResourceState->info.image.image;
					imageMemoryBarrier.subresourceRange.aspectMask = pResourceState->info.image.aspectFlags;
					imageMemoryBarrier.subresourceRange.layerCount = 1;
					imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
					imageMemoryBarrier.subresourceRange.levelCount = 1;
					imageMemoryBarrier.subresourceRange.baseMipLevel = 0;

					TRM_DynamicArray_push(&imageMemoryBarrier, &imageMemoryBarriers);
				}
			}
		}

		// write after read/write (execution) + layout
		for(uint32_t outputIndex = 0; outputIndex < pPasses[passIndex].outputCount; ++outputIndex)
		{
			const struct TRM_Renderer_ResourceAccess currentResourceAccess = pPasses[passIndex].pOutputs[outputIndex];
			const uint32_t resourceIndex = currentResourceAccess.resourceIndex;
			const struct TRM_Renderer_Resource* pResourceState = &pResources[resourceIndex];

			const uint32_t lastWasWrite = (pResourceState->accessFlags & (
				VK_ACCESS_SHADER_WRITE_BIT |
				VK_ACCESS_TRANSFER_WRITE_BIT |
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));

			const uint32_t currentIsWrite = (currentResourceAccess.accessFlags & (
				VK_ACCESS_SHADER_WRITE_BIT | 
				VK_ACCESS_TRANSFER_WRITE_BIT | 
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | 
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));

			uint32_t layoutChanged = 0;
			if(pResourceState->type == TRM_RENDERER_RESOURCE_TYPE_IMAGE)
			{
				layoutChanged = (pResourceState->info.image.layout != currentResourceAccess.layout);
			}

			if(lastWasWrite || currentIsWrite || layoutChanged)
			{
				needBarrier = 1;

				srcStageFlags |= pResourceState->stageFlags;
				dstStageFlags |= currentResourceAccess.stageFlags;

				if(pResourceState->type == TRM_RENDERER_RESOURCE_TYPE_BUFFER)
				{
					VkBufferMemoryBarrier bufferMemoryBarrier = {0};
					bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
					bufferMemoryBarrier.pNext = NULL;
					bufferMemoryBarrier.srcAccessMask = VK_ACCESS_NONE;
					bufferMemoryBarrier.dstAccessMask = VK_ACCESS_NONE;
					bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bufferMemoryBarrier.buffer = pResourceState->info.buffer.buffer;
					bufferMemoryBarrier.offset = 0;
					bufferMemoryBarrier.size = VK_WHOLE_SIZE;

					TRM_DynamicArray_push(&bufferMemoryBarrier, &bufferMemoryBarriers);
				}
				else if(pResourceState->type == TRM_RENDERER_RESOURCE_TYPE_IMAGE)
				{
					VkImageMemoryBarrier imageMemoryBarrier = {0};
					imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					imageMemoryBarrier.pNext = NULL;
					imageMemoryBarrier.srcAccessMask = VK_ACCESS_NONE;
					imageMemoryBarrier.dstAccessMask = VK_ACCESS_NONE;
					imageMemoryBarrier.oldLayout = pResourceState->info.image.layout;
					imageMemoryBarrier.newLayout = currentResourceAccess.layout;
					imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imageMemoryBarrier.image = pResourceState->info.image.image;
					imageMemoryBarrier.subresourceRange.aspectMask = pResourceState->info.image.aspectFlags;
					imageMemoryBarrier.subresourceRange.layerCount = 1;
					imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
					imageMemoryBarrier.subresourceRange.levelCount = 1;
					imageMemoryBarrier.subresourceRange.baseMipLevel = 0;

					TRM_DynamicArray_push(&imageMemoryBarrier, &imageMemoryBarriers);
				}
			}
		}

		if(needBarrier == 1)
		{
			vkCmdPipelineBarrier(
				commandBuffer,
				srcStageFlags,
				dstStageFlags,
				0,
				0,
				NULL,
				bufferMemoryBarriers.elementCount,
				(VkBufferMemoryBarrier*)bufferMemoryBarriers.pData,
				imageMemoryBarriers.elementCount,
				(VkImageMemoryBarrier*)imageMemoryBarriers.pData);

			TRM_DynamicArray_destroy(&bufferMemoryBarriers);
			TRM_DynamicArray_destroy(&imageMemoryBarriers);
		}

		if(pPasses[passIndex].type == TRM_RENDERER_PASS_TYPE_DISPATCH)
		{
			vkCmdBindDescriptorSets(
				commandBuffer, 
				VK_PIPELINE_BIND_POINT_COMPUTE, 
				pPasses[passIndex].info.dispatch.pipelineLayout, 
				0, 
				pPasses[passIndex].info.dispatch.descriptorSetCount, 
				pPasses[passIndex].info.dispatch.pDescriptorSets, 
				0, 
				NULL);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pPasses[passIndex].info.dispatch.pipeline);
			
			vkCmdDispatch(
				commandBuffer, 
				pPasses[passIndex].info.dispatch.groupCountX, 
				pPasses[passIndex].info.dispatch.groupCountY, 
				pPasses[passIndex].info.dispatch.groupCountZ);
		}
		else if(pPasses[passIndex].type == TRM_RENDERER_PASS_TYPE_DRAW)
		{
			VkRenderPassBeginInfo renderPassBeginInfo = {0};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.pNext = NULL;
			renderPassBeginInfo.renderPass = pPasses[passIndex].info.draw.renderPass;
			renderPassBeginInfo.framebuffer = pPasses[passIndex].info.draw.framebuffer;
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = pPasses[passIndex].info.draw.width;
			renderPassBeginInfo.renderArea.extent.height = pPasses[passIndex].info.draw.height;
			renderPassBeginInfo.clearValueCount = pPasses[passIndex].info.draw.attachmentCount;
			renderPassBeginInfo.pClearValues = pPasses[passIndex].info.draw.pClearValues;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &pResources[pPasses[passIndex].pInputs[0].resourceIndex].info.buffer.buffer, &offset);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPasses[passIndex].info.draw.pipeline);
			
			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pPasses[passIndex].info.draw.pipelineLayout,
				0,
				pPasses[passIndex].info.draw.descriptorSetCount,
				pPasses[passIndex].info.draw.pDescriptorSets,
				0,
				NULL);

			vkCmdDraw(commandBuffer, pPasses[passIndex].info.draw.vertexCount, 1, 0, 0);

			vkCmdEndRenderPass(commandBuffer);
		}
		else if(pPasses[passIndex].type == TRM_RENDERER_PASS_TYPE_IMAGE_COPY)
		{
			vkCmdCopyImage(
				commandBuffer,
				pResources[pPasses[passIndex].pInputs[0].resourceIndex].info.image.image,
				pResources[pPasses[passIndex].pInputs[0].resourceIndex].info.image.layout,
				pResources[pPasses[passIndex].pOutputs[0].resourceIndex].info.image.image,
				pResources[pPasses[passIndex].pOutputs[0].resourceIndex].info.image.layout,
				1,
				&pPasses[passIndex].info.imageCopy.imageCopy);
		}

		// update last access
		for(uint32_t inputIndex = 0; inputIndex < pPasses[passIndex].inputCount; ++inputIndex)
		{
			const struct TRM_Renderer_ResourceAccess resourceAccess = pPasses[passIndex].pInputs[inputIndex];
			
			pResources[resourceAccess.resourceIndex].accessFlags = resourceAccess.accessFlags;
			pResources[resourceAccess.resourceIndex].stageFlags = resourceAccess.stageFlags;

			if(pResources[resourceAccess.resourceIndex].type == TRM_RENDERER_RESOURCE_TYPE_IMAGE)
			{
				pResources[resourceAccess.resourceIndex].info.image.layout = resourceAccess.layout;
			}
		}
		for(uint32_t outputIndex = 0; outputIndex < pPasses[passIndex].outputCount; ++outputIndex)
		{
			const struct TRM_Renderer_ResourceAccess resourceAccess = pPasses[passIndex].pOutputs[outputIndex];
			
			pResources[resourceAccess.resourceIndex].accessFlags = resourceAccess.accessFlags;
			pResources[resourceAccess.resourceIndex].stageFlags = resourceAccess.stageFlags;

			if(pResources[resourceAccess.resourceIndex].type == TRM_RENDERER_RESOURCE_TYPE_IMAGE)
			{
				pResources[resourceAccess.resourceIndex].info.image.layout = resourceAccess.layout;
			}
		}
	}

	if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

struct Vertex
{
	float x;
	float y;
	float z;
};

void TRM_Renderer_start(GLFWwindow* pWindow, uint32_t windowWidth, uint32_t windowHeight)
{
	if(pState != NULL)
	{
		exit(EXIT_FAILURE);
	}

	TRM_Memory_allocate(sizeof(struct TRM_Renderer_State), (void**)&pState);
	TRM_Memory_memzero(sizeof(struct TRM_Renderer_State), pState);

	if(volkInitialize() != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	pState->pAllocator = NULL;

	TRM_Renderer_createInstance(pState->pAllocator, &pState->instance);
	volkLoadInstance(pState->instance);
	TRM_Renderer_pickPhysicalDevice(pState->instance, &pState->physicalDevice);
	TRM_Renderer_createSurface(pState->pAllocator, pState->instance, pWindow, &pState->surface);
	TRM_Renderer_findQueueFamilyIndex(pState->physicalDevice, pState->surface, &pState->queueFamilyIndex);
	TRM_Renderer_createDevice(pState->pAllocator, pState->physicalDevice, pState->queueFamilyIndex, &pState->device);
	vkGetDeviceQueue(pState->device, pState->queueFamilyIndex, 0, &pState->queue);
	TRM_Renderer_createCommandPool(pState->pAllocator, pState->device, pState->queueFamilyIndex, &pState->commandPool);
	TRM_Renderer_createDescriptorPool(pState->pAllocator, pState->device, &pState->descriptorPool);

	TRM_Renderer_createSwapchain(
		pState->pAllocator,
		pState->physicalDevice,
		pState->device,
		pState->surface,
		windowWidth,
		windowHeight,
		true,
		pState->queueFamilyIndex,
		&pState->swapchain, 
		&pState->swapchainFormat, 
		&pState->swapchainWidth, 
		&pState->swapchainHeight);

	TRM_Arena_create(sizeof(struct TRM_Renderer_Resource), TRM_RENDERER_MAX_RESOURCE_COUNT, &pState->resources);

	// === SWAPCHAIN IMAGE INFOS SETUP ===

	VkImage* pSwapchainImages = NULL;
	vkGetSwapchainImagesKHR(pState->device, pState->swapchain, &pState->swapchainImageCount, NULL);
	TRM_Memory_allocate(sizeof(VkImage) * pState->swapchainImageCount, (void**)&pSwapchainImages);
	vkGetSwapchainImagesKHR(pState->device, pState->swapchain, &pState->swapchainImageCount, pSwapchainImages);

	TRM_Memory_allocate(sizeof(struct TRM_Renderer_SwapchainImageInfo) * pState->swapchainImageCount, (void**)&pState->pSwapchainImageInfos);

	for(uint32_t swapchainImageIndex = 0; swapchainImageIndex < pState->swapchainImageCount; ++swapchainImageIndex)
	{
		struct TRM_Renderer_Resource swapchainColorImage = {0};
		swapchainColorImage.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
		swapchainColorImage.info.image.image = pSwapchainImages[swapchainImageIndex];
		swapchainColorImage.stageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		swapchainColorImage.accessFlags = VK_ACCESS_NONE;
		
		TRM_Renderer_createImageView(
			pState->pAllocator,
			pState->device,
			pSwapchainImages[swapchainImageIndex],
			pState->swapchainFormat,
			VK_IMAGE_ASPECT_COLOR_BIT,
			&swapchainColorImage.info.image.imageView);

		swapchainColorImage.info.image.memory = VK_NULL_HANDLE;
		swapchainColorImage.info.image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
		swapchainColorImage.info.image.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
		swapchainColorImage.info.image.swapchainImage = true;

		TRM_Arena_add(
			(void*)&swapchainColorImage, 
			&pState->resources, 
			&pState->pSwapchainImageInfos[swapchainImageIndex].colorImageIndex);

		struct TRM_Renderer_Resource swapchainDepthImage = {0};
		swapchainDepthImage.type = TRM_RENDERER_RESOURCE_TYPE_IMAGE;
		swapchainDepthImage.stageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		swapchainDepthImage.accessFlags = VK_ACCESS_NONE;

		TRM_Renderer_createImage(
			pState->pAllocator, 
			pState->device,
			pState->swapchainWidth,
			pState->swapchainHeight,
			VK_FORMAT_D32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			pState->queueFamilyIndex,
			&swapchainDepthImage.info.image.image);

		TRM_Renderer_allocateMemoryForImage(
			pState->pAllocator,
			pState->physicalDevice,
			pState->device,
			swapchainDepthImage.info.image.image,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&swapchainDepthImage.info.image.memory);

		vkBindImageMemory(
			pState->device, 
			swapchainDepthImage.info.image.image, 
			swapchainDepthImage.info.image.memory, 0);

		TRM_Renderer_createImageView(
			pState->pAllocator,
			pState->device,
			swapchainDepthImage.info.image.image,
			VK_FORMAT_D32_SFLOAT,
			VK_IMAGE_ASPECT_DEPTH_BIT,
			&swapchainDepthImage.info.image.imageView);

		swapchainDepthImage.info.image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
		swapchainDepthImage.info.image.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		swapchainDepthImage.info.image.swapchainImage = false;

		TRM_Arena_add(
			(void*)&swapchainDepthImage, 
			&pState->resources, 
			&pState->pSwapchainImageInfos[swapchainImageIndex].depthImageIndex);

		TRM_Renderer_createSemaphore(
			pState->pAllocator, 
			pState->device, 
			&pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore);
	}

	TRM_Memory_deallocate(pSwapchainImages);

	VkAttachmentDescription attachmentDescriptions[2];
	attachmentDescriptions[0].flags = 0;
	attachmentDescriptions[0].format = pState->swapchainFormat;
	attachmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	attachmentDescriptions[1].flags = 0;
	attachmentDescriptions[1].format = VK_FORMAT_D32_SFLOAT;
	attachmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachmentReference = {0};
	colorAttachmentReference.attachment = 0;
	colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentReference = {0};
	depthAttachmentReference.attachment = 1;
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {0};
	subpassDescription.flags = 0;
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = NULL;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorAttachmentReference;
	subpassDescription.pResolveAttachments = NULL;
	subpassDescription.pDepthStencilAttachment = &depthAttachmentReference;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = NULL;

	TRM_Renderer_createRenderPass(
		pState->pAllocator, 
		pState->device, 
		sizeof(attachmentDescriptions) / sizeof(attachmentDescriptions[0]), 
		attachmentDescriptions, 
		subpassDescription, 
		&pState->renderPass);

	for(uint32_t swapchainImageInfoIndex = 0; swapchainImageInfoIndex < pState->swapchainImageCount; ++swapchainImageInfoIndex)
	{
		struct TRM_Renderer_Resource* pColorImage = NULL;
		TRM_Arena_get(pState->pSwapchainImageInfos[swapchainImageInfoIndex].colorImageIndex, pState->resources, (void**)&pColorImage);

		struct TRM_Renderer_Resource* pDepthImage = NULL;
		TRM_Arena_get(pState->pSwapchainImageInfos[swapchainImageInfoIndex].depthImageIndex, pState->resources, (void**)&pDepthImage);

		VkImageView attachments[] = {
			pColorImage->info.image.imageView,
			pDepthImage->info.image.imageView
		};

		TRM_Renderer_createFramebuffer(
			pState->pAllocator, 
			pState->device, 
			pState->renderPass, 
			sizeof(attachments) / sizeof(attachments[0]), 
			attachments, 
			pState->swapchainWidth, 
			pState->swapchainHeight, 
			&pState->pSwapchainImageInfos[swapchainImageInfoIndex].framebuffer);
	}

	// === FRAME INFOS SETUP ====

	VkDescriptorSetLayoutBinding bindings[1];
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	bindings[0].pImmutableSamplers = NULL;

	TRM_Renderer_createDescriptorSetLayout(pState->pAllocator, pState->device, sizeof(bindings) / sizeof(bindings[0]), bindings, &pState->descriptorSetLayout);

	TRM_Memory_allocate(sizeof(struct TRM_Renderer_FrameInfo) * TRM_RENDERER_FRAME_COUNT, (void**)&pState->pFrameInfos);

	for(uint32_t frameIndex = 0; frameIndex < TRM_RENDERER_FRAME_COUNT; ++frameIndex)
	{
		TRM_Renderer_allocateCommandBuffer(pState->commandPool, pState->device, &pState->pFrameInfos[frameIndex].commandBuffer);
		TRM_Renderer_createFence(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndex].commandBufferExecutedFence);
		TRM_Renderer_createSemaphore(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndex].imageAvailableSemaphore);
		TRM_Renderer_allocateDescriptorSet(pState->device, pState->descriptorPool, pState->descriptorSetLayout, &pState->pFrameInfos[frameIndex].descriptorSet);

		struct TRM_Renderer_Resource vertexBuffer = {0};

		vertexBuffer.type = TRM_RENDERER_RESOURCE_TYPE_BUFFER;
		vertexBuffer.accessFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		vertexBuffer.stageFlags = VK_ACCESS_NONE;

		TRM_Renderer_createBuffer(
			pState->pAllocator,
			pState->device,
			sizeof(struct Vertex) * 3,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			pState->queueFamilyIndex,
			&vertexBuffer.info.buffer.buffer);

		TRM_Renderer_allocateMemoryForBuffer(
			pState->pAllocator,
			pState->physicalDevice,
			pState->device,
			vertexBuffer.info.buffer.buffer,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&vertexBuffer.info.buffer.memory);

		if(vkBindBufferMemory(
			pState->device,
			vertexBuffer.info.buffer.buffer,
			vertexBuffer.info.buffer.memory,
			0) != VK_SUCCESS)
		{
			exit(EXIT_FAILURE);
		}

		struct Vertex vertices[3] = {
			{-1.0f, 0.0f, 1.0f},
			{0.0f, 1.0f, 1.0f},
			{1.0f, 0.0f, 1.0f}
		};

		void* pData = NULL;
		vkMapMemory(pState->device, vertexBuffer.info.buffer.memory, 0, sizeof(struct Vertex) * 3, 0, &pData);
		TRM_Memory_memcpy(sizeof(struct Vertex) * 3, vertices, pData);
		vkUnmapMemory(pState->device, vertexBuffer.info.buffer.memory);

		TRM_Arena_add(&vertexBuffer, &pState->resources, &pState->pFrameInfos[frameIndex].vertexBufferIndex);

		struct TRM_Renderer_Resource uniformBuffer = {0};

		TRM_Renderer_createBuffer(
			pState->pAllocator,
			pState->device,
			sizeof(float) * 4,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			pState->queueFamilyIndex,
			&uniformBuffer.info.buffer.buffer);

		TRM_Renderer_allocateMemoryForBuffer(
			pState->pAllocator,
			pState->physicalDevice,
			pState->device,
			uniformBuffer.info.buffer.buffer,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffer.info.buffer.memory);

		if(vkBindBufferMemory(
			pState->device,
			uniformBuffer.info.buffer.buffer,
			uniformBuffer.info.buffer.memory,
			0) != VK_SUCCESS)
		{
			exit(EXIT_FAILURE);
		}

		TRM_Arena_add(&uniformBuffer, &pState->resources, &pState->pFrameInfos[frameIndex].uniformBufferIndex);
	}

	pState->frameIndex = 0;
	
	VkShaderModule vertexShaderModule;
	{
		uint32_t codeSize = 0;
		uint32_t* pCode = NULL;
		TRM_Renderer_readShader(PROJECT_ROOT "/assets/shaders/vertex.spv", &codeSize, &pCode);
		TRM_Renderer_createShaderModule(pState->pAllocator, pState->device, codeSize, pCode, &vertexShaderModule);
		TRM_Memory_deallocate(pCode);
	}

	VkShaderModule fragmentShaderModule;
	
	{
		uint32_t codeSize = 0;
		uint32_t* pCode = NULL;
		TRM_Renderer_readShader(PROJECT_ROOT "/assets/shaders/fragment.spv", &codeSize, &pCode);
		TRM_Renderer_createShaderModule(pState->pAllocator, pState->device, codeSize, pCode, &fragmentShaderModule);
		TRM_Memory_deallocate(pCode);
	}

	TRM_Renderer_createPipelineLayout(pState->pAllocator, pState->device, 1, &pState->descriptorSetLayout, &pState->graphicsPipelineLayout);

	VkVertexInputBindingDescription vertexBindingDescription = {0};
	vertexBindingDescription.binding = 0;
	vertexBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	vertexBindingDescription.stride = sizeof(struct Vertex);

	VkVertexInputAttributeDescription vertexAttributeDescriptions = {0};
	vertexAttributeDescriptions.binding = 0;
	vertexAttributeDescriptions.format = VK_FORMAT_R32G32B32_SFLOAT;
	vertexAttributeDescriptions.location = 0;
	vertexAttributeDescriptions.offset = 0;

	VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {0};
	vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputStateCreateInfo.pNext = NULL;
	vertexInputStateCreateInfo.flags = 0;
	vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
	vertexInputStateCreateInfo.pVertexBindingDescriptions = &vertexBindingDescription;
	vertexInputStateCreateInfo.vertexAttributeDescriptionCount = 1;
	vertexInputStateCreateInfo.pVertexAttributeDescriptions = &vertexAttributeDescriptions;

	TRM_Renderer_createGraphicsPipeline(
		pState->pAllocator, 
		pState->device, 
		vertexShaderModule, 
		fragmentShaderModule, 
		vertexInputStateCreateInfo, 
		pState->graphicsPipelineLayout, 
		pState->renderPass, 
		pState->swapchainWidth,
		pState->swapchainHeight,
		&pState->graphicsPipeline);
	
	vkDestroyShaderModule(pState->device, vertexShaderModule, pState->pAllocator);
	vkDestroyShaderModule(pState->device, fragmentShaderModule, pState->pAllocator);
}

void TRM_Renderer_terminate(void)
{
	if(pState != NULL)
	{
		if(vkDeviceWaitIdle(pState->device) != VK_SUCCESS)
		{
			exit(EXIT_FAILURE);
		}

		for(uint32_t frameIndex = 0; frameIndex < TRM_RENDERER_FRAME_COUNT; ++frameIndex)
		{
			struct TRM_Renderer_Resource* pVertexBuffer;
			TRM_Arena_get(pState->pFrameInfos[frameIndex].vertexBufferIndex, pState->resources, (void**)&pVertexBuffer);

			vkDestroyBuffer(pState->device, pVertexBuffer->info.buffer.buffer, pState->pAllocator);
			vkFreeMemory(pState->device, pVertexBuffer->info.buffer.memory, pState->pAllocator);

			struct TRM_Renderer_Resource* pUniformBuffer;
			TRM_Arena_get(pState->pFrameInfos[frameIndex].uniformBufferIndex, pState->resources, (void**)&pUniformBuffer);

			vkDestroyBuffer(pState->device, pUniformBuffer->info.buffer.buffer, pState->pAllocator);
			vkFreeMemory(pState->device, pUniformBuffer->info.buffer.memory, pState->pAllocator);

			vkDestroySemaphore(pState->device, pState->pFrameInfos[frameIndex].imageAvailableSemaphore, pState->pAllocator);
			vkDestroyFence(pState->device, pState->pFrameInfos[frameIndex].commandBufferExecutedFence, pState->pAllocator);
		}

		vkDestroyPipeline(pState->device, pState->graphicsPipeline, pState->pAllocator);
		vkDestroyPipelineLayout(pState->device, pState->graphicsPipelineLayout, pState->pAllocator);
		vkDestroyDescriptorSetLayout(pState->device, pState->descriptorSetLayout, pState->pAllocator);
		vkDestroyRenderPass(pState->device, pState->renderPass, pState->pAllocator);
		
		TRM_Memory_deallocate(pState->pFrameInfos);

		for(uint32_t swapchainImageIndex = 0; swapchainImageIndex < pState->swapchainImageCount; ++swapchainImageIndex)
		{
			vkDestroyFramebuffer(pState->device, pState->pSwapchainImageInfos[swapchainImageIndex].framebuffer, pState->pAllocator);
			
			struct TRM_Renderer_Resource* pSwapchainColorImage;
			TRM_Arena_get(pState->pSwapchainImageInfos[swapchainImageIndex].colorImageIndex, pState->resources, (void**)&pSwapchainColorImage);

			vkDestroyImageView(pState->device, pSwapchainColorImage->info.image.imageView, pState->pAllocator);
			vkFreeMemory(pState->device, pSwapchainColorImage->info.image.memory, pState->pAllocator);

			struct TRM_Renderer_Resource* pSwapchainDepthImage;
			TRM_Arena_get(pState->pSwapchainImageInfos[swapchainImageIndex].depthImageIndex, pState->resources, (void**)&pSwapchainDepthImage);

			vkDestroyImage(pState->device, pSwapchainDepthImage->info.image.image, pState->pAllocator);
			vkDestroyImageView(pState->device, pSwapchainDepthImage->info.image.imageView, pState->pAllocator);
			vkFreeMemory(pState->device, pSwapchainDepthImage->info.image.memory, pState->pAllocator);

			vkDestroySemaphore(pState->device, pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore, pState->pAllocator);
		}

		TRM_Memory_deallocate(pState->pSwapchainImageInfos);
		
		TRM_Arena_destroy(&pState->resources);
		
		vkDestroySwapchainKHR(pState->device, pState->swapchain, pState->pAllocator);
		vkDestroyDescriptorPool(pState->device, pState->descriptorPool, pState->pAllocator);
		vkDestroyCommandPool(pState->device, pState->commandPool, pState->pAllocator);
		vkDestroyDevice(pState->device, pState->pAllocator);
		vkDestroySurfaceKHR(pState->instance, pState->surface, pState->pAllocator);
		vkDestroyInstance(pState->instance, pState->pAllocator);

		TRM_Memory_deallocate(pState);
		pState = NULL;
	}
}

void TRM_Renderer_render(void)
{
	if(vkWaitForFences(pState->device, 1, &pState->pFrameInfos[pState->frameIndex].commandBufferExecutedFence, VK_FALSE, UINT64_MAX) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	if(vkResetFences(pState->device, 1, &pState->pFrameInfos[pState->frameIndex].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_Renderer_Resource* pUniformBuffer;
	TRM_Arena_get(pState->pFrameInfos[pState->frameIndex].uniformBufferIndex, pState->resources, (void**)&pUniformBuffer);

	{
		static float toy = 0.0f;
		float color[4] = {(cosf(toy) + 1) / 2, (sinf(toy) + 1) / 2, (cosf(toy) + 1) / 2, 1.0f};
		toy += 0.01f;

		void* pData = NULL;
		vkMapMemory(pState->device, pUniformBuffer->info.buffer.memory, 0, sizeof(color), 0, &pData);
		TRM_Memory_memcpy(sizeof(color), color, pData);
		vkUnmapMemory(pState->device, pUniformBuffer->info.buffer.memory);
	}

	uint32_t imageIndex = 0;
	if(vkAcquireNextImageKHR(
		pState->device,
		pState->swapchain,
		UINT64_MAX,
		pState->pFrameInfos[pState->frameIndex].imageAvailableSemaphore,
		VK_NULL_HANDLE,
		&imageIndex) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_Renderer_Resource* pSwapchainColorImage;
	TRM_Arena_get(pState->pSwapchainImageInfos[imageIndex].colorImageIndex, pState->resources, (void**)&pSwapchainColorImage);

	VkDescriptorBufferInfo bufferInfo = {0};
	bufferInfo.buffer = pUniformBuffer->info.buffer.buffer;
	bufferInfo.offset = 0;
	bufferInfo.range = VK_WHOLE_SIZE;

	VkWriteDescriptorSet descriptorWrites[1];
	descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[0].pNext = NULL;
	descriptorWrites[0].dstSet = pState->pFrameInfos[pState->frameIndex].descriptorSet;
	descriptorWrites[0].dstBinding = 0;
	descriptorWrites[0].dstArrayElement = 0;
	descriptorWrites[0].descriptorCount = 1;
	descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrites[0].pImageInfo = NULL;
	descriptorWrites[0].pBufferInfo = &bufferInfo;
	descriptorWrites[0].pTexelBufferView = NULL;

	vkUpdateDescriptorSets(pState->device, sizeof(descriptorWrites) / sizeof(descriptorWrites[0]), descriptorWrites, 0, NULL);

	struct TRM_Renderer_ResourceAccess inputs[2];
	inputs[0].resourceIndex = pState->pFrameInfos[pState->frameIndex].vertexBufferIndex;
	inputs[0].accessFlags = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
	inputs[0].stageFlags = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;

	inputs[1].resourceIndex = pState->pFrameInfos[pState->frameIndex].uniformBufferIndex;
	inputs[1].accessFlags = VK_ACCESS_SHADER_READ_BIT;
	inputs[1].stageFlags = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;

	struct TRM_Renderer_ResourceAccess outputs[2];
	outputs[0].resourceIndex = pState->pSwapchainImageInfos[imageIndex].colorImageIndex;
	outputs[0].accessFlags = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	outputs[0].stageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	outputs[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	
	outputs[1].resourceIndex = pState->pSwapchainImageInfos[imageIndex].depthImageIndex;
	outputs[1].accessFlags = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	outputs[1].stageFlags = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	outputs[1].layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkClearValue clears[2];
	clears[0].color.float32[0] = 1.0f;
	clears[0].color.float32[1] = 0.0f;
	clears[0].color.float32[2] = 0.0f;
	clears[0].color.float32[3] = 1.0f;

	clears[1].depthStencil.depth = 1.0f;
	clears[1].depthStencil.stencil = 0;

	struct TRM_Renderer_Pass passes[2];
	passes[0].type = TRM_RENDERER_PASS_TYPE_DRAW;
	passes[0].inputCount = sizeof(inputs) / sizeof(inputs[0]);
	passes[0].pInputs = inputs;
	passes[0].outputCount = sizeof(outputs) / sizeof(outputs[0]);
	passes[0].pOutputs = outputs;
	passes[0].info.draw.width = pState->swapchainWidth;
	passes[0].info.draw.height = pState->swapchainHeight;
	passes[0].info.draw.vertexCount = 3;
	passes[0].info.draw.renderPass = pState->renderPass;
	passes[0].info.draw.framebuffer = pState->pSwapchainImageInfos[imageIndex].framebuffer;
	passes[0].info.draw.pipelineLayout = pState->graphicsPipelineLayout;
	passes[0].info.draw.descriptorSetCount = 1;
	passes[0].info.draw.pDescriptorSets = &pState->pFrameInfos[pState->frameIndex].descriptorSet;
	passes[0].info.draw.pipeline = pState->graphicsPipeline;
	passes[0].info.draw.attachmentCount = 2;
	passes[0].info.draw.pClearValues = clears;

	struct TRM_Renderer_ResourceAccess present;
	present.resourceIndex = pState->pSwapchainImageInfos[imageIndex].colorImageIndex;
	present.accessFlags = VK_ACCESS_NONE;
	present.stageFlags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	present.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	passes[1].type = TRM_RENDERER_PASS_TYPE_PRESENT;
	passes[1].inputCount = 1;
	passes[1].pInputs = &present;
	passes[1].outputCount = 0;
	passes[1].pOutputs = NULL;

	TRM_Renderer_fillCommandBuffer(
		(struct TRM_Renderer_Resource*)pState->resources.pData, 
		sizeof(passes) / sizeof(passes[0]), 
		passes, 
		pState->pFrameInfos[pState->frameIndex].commandBuffer);

	VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

	VkSubmitInfo submitInfo = {0};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = NULL;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &pState->pFrameInfos[pState->frameIndex].imageAvailableSemaphore;
	submitInfo.pWaitDstStageMask = &waitDstStageMask;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &pState->pFrameInfos[pState->frameIndex].commandBuffer;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &pState->pSwapchainImageInfos[imageIndex].imageRenderedSemaphore;

	if(vkQueueSubmit(pState->queue, 1, &submitInfo, pState->pFrameInfos[pState->frameIndex].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	VkPresentInfoKHR presentInfo = {0};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = NULL;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &pState->pSwapchainImageInfos[imageIndex].imageRenderedSemaphore;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &pState->swapchain;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = NULL;

	if(vkQueuePresentKHR(pState->queue, &presentInfo) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	pState->frameIndex = (pState->frameIndex + 1) % TRM_RENDERER_FRAME_COUNT;
}
