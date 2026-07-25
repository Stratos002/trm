#include "trm.h"
#include "trm_memory.h"
#include "trm_containers.h"

#define VOLK_IMPLEMENTATION
#include "volk.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// having 1 more frame in flight could hide lag spikes from the GPU, but I don't think it's worth the memory cost
#define TRM_BACKEND_FRAME_COUNT 2 
#define TRM_BACKEND_MAX_RESOURCE_COUNT 1024
#define TRM_BACKEND_MAX_PIPELINE_COUNT 64
#define TRM_MAX_DESCRIPTOR_SET_PER_FRAME_COUNT 32
#define TRM_MAX_FRAMEBUFFER_PER_FRAME_COUNT 8
#define TRM_MAX_MIP_PER_IMAGE_COUNT 16

// for later
/*
device local buffers : can be shared across all frames (vertex buffers, textures, depth), must sync access with barriers. 
					   (it's not worth having cross frame parallelism as there are usually enough intra frame parallelism)
host visible buffers : frequently written by the CPU, we must keep one copy per frame (uniform buffers)
occasionally updated buffers : rarely written by the CPU, we can have one device local copy + copy buffer (which could be recyclable) [for later]
*/

enum TRM_Backend_ResourceType
{
	TRM_BACKEND_RESOURCE_TYPE_BUFFER,
	TRM_BACKEND_RESOURCE_TYPE_BUFFER_INDIRECTION,
	TRM_BACKEND_RESOURCE_TYPE_IMAGE
};

struct TRM_Backend_HostVisibleBufferIndirectionInfo
{
	uint32_t buffers[TRM_BACKEND_FRAME_COUNT];
};

struct TRM_Backend_DeviceLocalBufferIndirectionInfo
{
	uint32_t buffer;
};

struct TRM_Backend_BufferIndirectionResourceInfo
{
	bool hostVisible;
	union
	{
		struct TRM_Backend_HostVisibleBufferIndirectionInfo hostVisible;
		struct TRM_Backend_DeviceLocalBufferIndirectionInfo deviceLocal;
	} info;
};

struct TRM_Backend_BufferResourceInfo
{
	VkBuffer buffer;
	VkDeviceMemory memory;
};

struct TRM_Backend_ImageResourceInfo
{
	VkImage image;
	uint32_t mipCount;
	VkImageView allMipsImageView;
	VkImageView singleMipImageViews[TRM_MAX_MIP_PER_IMAGE_COUNT];
	VkImageAspectFlags aspect;
	VkDeviceMemory memory;
	bool swapchainImage;
};

struct TRM_Backend_ResourceState
{
	VkAccessFlags access;
	VkPipelineStageFlags stage;
	VkImageLayout layout;
};

struct TRM_Backend_Resource
{
	enum TRM_Backend_ResourceType type;
	struct TRM_Backend_ResourceState state;
	bool toDelete;
	uint32_t lastFrameIndexNoWarp;
	union
	{
		struct TRM_Backend_BufferResourceInfo buffer;
		struct TRM_Backend_BufferIndirectionResourceInfo bufferIndirection;
		struct TRM_Backend_ImageResourceInfo image;
	} info;
};

struct TRM_Backend_ComputePipeline
{
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	uint32_t descriptorInfoCount;
	struct TRM_DescriptorInfo descriptorInfos[TRM_MAX_DESCRIPTOR_COUNT];
};

struct TRM_Backend_GraphicsPipeline
{
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkRenderPass renderPass;
	uint32_t descriptorInfoCount;
	struct TRM_DescriptorInfo descriptorInfos[TRM_MAX_DESCRIPTOR_COUNT];
};

struct TRM_Backend_Pipeline
{
	enum TRM_PipelineType type;
	union
	{
		struct TRM_Backend_ComputePipeline compute;
		struct TRM_Backend_GraphicsPipeline graphics;
	} info;
};

struct TRM_Backend_DispatchPassInfo
{
	uint32_t pipeline;
	uint32_t descriptorSet;
	uint32_t groupCountX;
	uint32_t groupCountY;
	uint32_t groupCountZ;
};

struct TRM_Backend_drawVertexPassInfo
{
	uint32_t pipeline;
	bool useVertexBuffer;
	uint32_t vertexCount;
	uint32_t width;
	uint32_t height;
	uint32_t descriptorSet;
	uint32_t framebuffer;
	uint32_t clearColorCount;
	VkClearValue clearColors[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	uint32_t vertexBuffer;
};

struct TRM_Backend_drawIndexedPassInfo
{
	uint32_t pipeline;
	uint32_t indexCount;
	uint32_t width;
	uint32_t height;
	uint32_t descriptorSet;
	uint32_t framebuffer;
	uint32_t clearColorCount;
	VkClearValue clearColors[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	uint32_t vertexBuffer;
	uint32_t indexBuffer;
};

struct TRM_Backend_drawVertexIndirectPassInfo
{
	uint32_t pipeline;
	bool useVertexBuffer;
	uint32_t drawCount;
	uint32_t width;
	uint32_t height;
	uint32_t descriptorSet;
	uint32_t framebuffer;
	uint32_t clearColorCount;
	VkClearValue clearColors[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	uint32_t vertexBuffer;
	uint32_t indirectCommandBuffer;
};

struct TRM_Backend_drawIndexedIndirectPassInfo
{
	uint32_t pipeline;
	uint32_t drawCount;
	uint32_t width;
	uint32_t height;
	uint32_t descriptorSet;
	uint32_t framebuffer;
	uint32_t clearColorCount;
	VkClearValue clearColors[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	uint32_t vertexBuffer;
	uint32_t indexBuffer;
	uint32_t indirectCommandBuffer;
};

struct TRM_Backend_ImageToImagePassInfo
{
	uint32_t width;
	uint32_t height;
	uint32_t srcImage;
	uint32_t dstImage;
};

struct TRM_Backend_BufferToImagePassInfo
{
	uint32_t width;
	uint32_t height;
	uint32_t srcBuffer;
	uint32_t dstImage;
};

struct TRM_Backend_BufferToBufferPassInfo
{
	uint32_t sizeInBytes;
	uint32_t srcBuffer;
	uint32_t dstBuffer;
};

struct TRM_Backend_BlitPassInfo
{
	uint32_t srcWidth;
	uint32_t srcHeight;
	uint32_t dstWidth;
	uint32_t dstHeight;
	uint32_t srcImage;
	uint32_t dstImage;
};

struct TRM_Backend_Binding
{
	uint32_t resource;
	uint32_t mip;
	VkImageLayout layout;
};

struct TRM_Backend_ExpectedResourceState
{
	uint32_t resource;
	struct TRM_Backend_ResourceState state;
};

struct TRM_Backend_Pass
{
	enum TRM_PassType type;
	struct TRM_DynamicArray expectedResourceStates;
	union
	{
		struct TRM_Backend_DispatchPassInfo dispatch;
		struct TRM_Backend_drawVertexPassInfo drawVertex;
		struct TRM_Backend_drawIndexedPassInfo drawIndexed;
		struct TRM_Backend_drawVertexIndirectPassInfo drawVertexIndirect;
		struct TRM_Backend_drawIndexedIndirectPassInfo drawIndexedIndirect;
		struct TRM_Backend_ImageToImagePassInfo imageToImageCopy;
		struct TRM_Backend_BufferToImagePassInfo bufferToImageCopy;
		struct TRM_Backend_BufferToBufferPassInfo bufferToBufferCopy;
		struct TRM_Backend_BlitPassInfo blit;
	} info;
};

struct TRM_Backend_FrameInfo
{
	VkCommandBuffer commandBuffer;
	VkFence commandBufferExecutedFence;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore timelineSemaphore;
	uint32_t descriptorSetCount;
	VkDescriptorSet descriptorSets[TRM_MAX_DESCRIPTOR_SET_PER_FRAME_COUNT];
	uint32_t framebufferCount;
	VkFramebuffer framebuffers[TRM_MAX_FRAMEBUFFER_PER_FRAME_COUNT];
};

struct TRM_Backend_SwapchainImageInfo
{
	VkSemaphore imageRenderedSemaphore;
	uint32_t colorImage;
};

struct TRM_Backend_State
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
	struct TRM_Backend_SwapchainImageInfo* pSwapchainImageInfos;
	struct TRM_Backend_FrameInfo* pFrameInfos;
	uint32_t frameIndexWarp;
	uint64_t frameIndexNoWarp;
	VkSampler globalSampler;
	struct TRM_Arena resourcePool;
	struct TRM_LinkedList resourceHandles;
	struct TRM_Arena pipelinePool;
};

static struct TRM_Backend_State* pState = NULL;

static void TRM_Backend_createInstance(const VkAllocationCallbacks* pAllocator, VkInstance* pInstance)
{
	VkApplicationInfo applicationInfo = {0};
	applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	applicationInfo.pNext = NULL;
	applicationInfo.pApplicationName = "TRM";
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pEngineName = "TRM";
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.apiVersion = VK_MAKE_API_VERSION(1, 2, 0, 0);

	const char* pValidationLayerName = "VK_LAYER_KHRONOS_validation";

	const char* pExtensionNames[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
	#if defined(TRM_PLATFORM_WINDOWS)
		VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
	#elif defined(TRM_PLATFORM_LINUX)
		VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
	#endif
	};

	VkInstanceCreateInfo instanceCreateInfo = {0};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.flags = 0;
	instanceCreateInfo.pApplicationInfo = &applicationInfo;
	instanceCreateInfo.enabledLayerCount = 1;
	instanceCreateInfo.ppEnabledLayerNames = &pValidationLayerName;
	instanceCreateInfo.enabledExtensionCount = sizeof(pExtensionNames) / sizeof(pExtensionNames[0]);
	instanceCreateInfo.ppEnabledExtensionNames = pExtensionNames;

	if(vkCreateInstance(&instanceCreateInfo, pAllocator, pInstance) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_pickPhysicalDevice(VkInstance instance, VkPhysicalDevice* pPhysicalDevice)
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

static void TRM_Backend_createSurface(
	const VkAllocationCallbacks* pAllocator,
	VkInstance instance,
	struct TRM_NativeWindow nativeWindow,
	VkSurfaceKHR* pSurface)
{
#if defined(TRM_PLATFORM_WINDOWS)
	VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {0};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.pNext = NULL;
	surfaceCreateInfo.flags = 0;
	surfaceCreateInfo.hinstance = nativeWindow.hinstance;
	surfaceCreateInfo.hwnd = nativeWindow.hwnd;

	if(vkCreateWin32SurfaceKHR(instance, &surfaceCreateInfo, pAllocator, pSurface) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

#elif defined(TRM_PLATFORM_LINUX)
	/*
	VkXcbSurfaceCreateInfoKHR createInfo = {
    .sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
    .connection = connection,
    .window = window
};

vkCreateXcbSurfaceKHR(
    instance,
    &createInfo,
    allocator,
    &surface);
	*/

	VkWaylandSurfaceCreateInfoKHR surfaceCreateInfo = {0};
	surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
	surfaceCreateInfo.pNext = NULL;
	surfaceCreateInfo.flags = 0;
	surfaceCreateInfo.display = nativeWindow.display;
	surfaceCreateInfo.surface = nativeWindow.surface;
	
	if(vkCreateWaylandSurfaceKHR(instance, &surfaceCreateInfo, pAllocator, pSurface) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
#endif
}

static void TRM_Backend_findQueueFamilyIndex(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, uint32_t* pQueueFamilyIndex)
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

static void TRM_Backend_createDevice(
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

	VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {0};
	timelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
	timelineFeatures.timelineSemaphore = VK_TRUE;

	VkPhysicalDeviceFeatures features = {0};
	features.multiDrawIndirect = VK_TRUE;

	VkDeviceCreateInfo deviceCreateInfo = {0};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext = NULL;
	deviceCreateInfo.pNext = &timelineFeatures;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
	deviceCreateInfo.enabledLayerCount = 0;
	deviceCreateInfo.ppEnabledLayerNames = NULL;
	deviceCreateInfo.enabledExtensionCount = sizeof(ppEnabledExtensionNames) / sizeof(ppEnabledExtensionNames[0]);
	deviceCreateInfo.ppEnabledExtensionNames = ppEnabledExtensionNames;
	deviceCreateInfo.pEnabledFeatures = &features;

	if(vkCreateDevice(physicalDevice, &deviceCreateInfo, pAllocator, pDevice) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_createSwapchain(
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
	swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

static void TRM_Backend_findMemoryTypeIndex(
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

static void TRM_Backend_allocateMemoryForBuffer(
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
	TRM_Backend_findMemoryTypeIndex(physicalDevice, memoryRequirements.memoryTypeBits, memoryPropertyFlags, &memoryTypeIndex);

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

static void TRM_Backend_allocateMemoryForImage(
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
	TRM_Backend_findMemoryTypeIndex(physicalDevice, memoryRequirements.memoryTypeBits, memoryPropertyFlags, &memoryTypeIndex);

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

static void TRM_Backend_createBuffer(
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

static void TRM_Backend_createImage(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	uint32_t width,
	uint32_t height,
	uint32_t mipCount,
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
	imageCreateInfo.mipLevels = mipCount;
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

static void TRM_Backend_createImageView(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkImage image,
	uint32_t baseMip,
	uint32_t mipCount,
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
	imageViewCreateInfo.subresourceRange.baseMipLevel = baseMip;
	imageViewCreateInfo.subresourceRange.levelCount = mipCount;
	imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	imageViewCreateInfo.subresourceRange.layerCount = 1;

	if(vkCreateImageView(device, &imageViewCreateInfo, pAllocator, pImageView) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_createSampler(const VkAllocationCallbacks* pAllocator, VkDevice device, VkSampler* pSampler)
{
	VkSamplerCreateInfo samplerCreateInfo = {0};
	samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCreateInfo.pNext = NULL;
	samplerCreateInfo.flags = 0;
	samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
	samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCreateInfo.mipLodBias = 0.0f;
	samplerCreateInfo.anisotropyEnable = VK_FALSE;
	samplerCreateInfo.maxAnisotropy = 0.0f;
	samplerCreateInfo.compareEnable = VK_FALSE;
	samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerCreateInfo.minLod = 0.0f;
	samplerCreateInfo.maxLod = (float)TRM_MAX_MIP_PER_IMAGE_COUNT;
	samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

	if(vkCreateSampler(device, &samplerCreateInfo, pAllocator, pSampler) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_createCommandPool(
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

static void TRM_Backend_createFence(const VkAllocationCallbacks* pAllocator, VkDevice device, VkFence* pFence)
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

static void TRM_Backend_createSemaphore(const VkAllocationCallbacks* pAllocator, VkDevice device, VkSemaphore* pSemaphore)
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

static void TRM_Backend_createTimelineSemaphore(const VkAllocationCallbacks* pAllocator, VkDevice device, VkSemaphore* pSemaphore)
{
	VkSemaphoreTypeCreateInfo timelineInfo = {0};
	timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	timelineInfo.pNext = NULL;
	timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	timelineInfo.initialValue = 0;

	VkSemaphoreCreateInfo semaphoreCreateInfo = {0};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = &timelineInfo;
	semaphoreCreateInfo.flags = 0;

	if(vkCreateSemaphore(device, &semaphoreCreateInfo, pAllocator, pSemaphore) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_allocateCommandBuffer(VkCommandPool commandPool, VkDevice device, VkCommandBuffer* pCommandBuffer)
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

static void TRM_Backend_createDescriptorPool(const VkAllocationCallbacks* pAllocator, VkDevice device, VkDescriptorPool* pDescriptorPool)
{
	VkDescriptorPoolSize uniformBufferDescriptorPoolSize = {0};
	uniformBufferDescriptorPoolSize.descriptorCount = 128;
	uniformBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

	VkDescriptorPoolSize storageBufferDescriptorPoolSize = {0};
	storageBufferDescriptorPoolSize.descriptorCount = 128;
	storageBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

	VkDescriptorPoolSize storageImageDescriptorPoolSize = {0};
	storageImageDescriptorPoolSize.descriptorCount = 128;
	storageImageDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

	VkDescriptorPoolSize combinedImageSamplerDescriptorPoolSize = {0};
	combinedImageSamplerDescriptorPoolSize.descriptorCount = 128;
	combinedImageSamplerDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

	VkDescriptorPoolSize descriptorPoolSizes[] = {
		uniformBufferDescriptorPoolSize,
		storageBufferDescriptorPoolSize,
		storageImageDescriptorPoolSize,
		combinedImageSamplerDescriptorPoolSize
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {0};
	descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolCreateInfo.pNext = NULL;
	descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	descriptorPoolCreateInfo.maxSets = 64;
	descriptorPoolCreateInfo.poolSizeCount = sizeof(descriptorPoolSizes) / sizeof(descriptorPoolSizes[0]);
	descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes;

	if(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, pAllocator, pDescriptorPool) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_allocateDescriptorSet(
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

static void TRM_Backend_createDescriptorSetLayout(
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

static void TRM_Backend_createPipelineLayout(
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

static void TRM_Backend_createShaderModule(
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

static void TRM_Backend_createComputePipeline(
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

static void TRM_Backend_createRenderPass(
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

static void TRM_Backend_createFramebuffer(
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

static void TRM_Backend_createGraphicsPipeline(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkShaderModule vertexShaderModule,
	VkShaderModule fragmentShaderModule,
	VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo,
	uint32_t colorAttachmentCount,
	VkPipelineLayout pipelineLayout,
	VkRenderPass renderPass,
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

	VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {0};
	viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateCreateInfo.pNext = NULL;
	viewportStateCreateInfo.flags = 0;
	viewportStateCreateInfo.viewportCount = 1;
	viewportStateCreateInfo.pViewports = NULL; // dynamic
	viewportStateCreateInfo.scissorCount = 1;
	viewportStateCreateInfo.pScissors = NULL; // dynamic

	VkDynamicState dynamicStates[2] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {0};
	dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicStateCreateInfo.pNext = NULL;
	dynamicStateCreateInfo.flags = 0;
	dynamicStateCreateInfo.dynamicStateCount = 2;
	dynamicStateCreateInfo.pDynamicStates = dynamicStates;

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

	VkPipelineColorBlendAttachmentState colorBlendAttachments[TRM_MAX_COLOR_OUTPUT_COUNT];

	for(uint32_t colorBlendAttachmentIndex = 0; colorBlendAttachmentIndex < colorAttachmentCount; ++colorBlendAttachmentIndex)
	{
		colorBlendAttachments[colorBlendAttachmentIndex].blendEnable = VK_FALSE;
		colorBlendAttachments[colorBlendAttachmentIndex].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachments[colorBlendAttachmentIndex].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachments[colorBlendAttachmentIndex].colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachments[colorBlendAttachmentIndex].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachments[colorBlendAttachmentIndex].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachments[colorBlendAttachmentIndex].alphaBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachments[colorBlendAttachmentIndex].colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
	}

	VkPipelineColorBlendStateCreateInfo colorBlendStateInfo = {0};
	colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendStateInfo.pNext = NULL;
	colorBlendStateInfo.flags = 0;
	colorBlendStateInfo.logicOpEnable = VK_FALSE;
	colorBlendStateInfo.logicOp = VK_LOGIC_OP_COPY;
	colorBlendStateInfo.attachmentCount = colorAttachmentCount;
	colorBlendStateInfo.pAttachments = colorBlendAttachments;
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
	graphicsPipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
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

static uint32_t TRM_Backend_translateResource(uint32_t handle, uint32_t swapchainImageIndex)
{
	if(handle == TRM_SWAPCHAIN_IMAGE)
	{
		return pState->pSwapchainImageInfos[swapchainImageIndex].colorImage;
	}
	
	struct TRM_Backend_Resource* pResource = NULL;
	TRM_Arena_get(handle, pState->resourcePool, (void**)&pResource);
	if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_BUFFER_INDIRECTION)
	{
		if(pResource->info.bufferIndirection.hostVisible)
		{
			return pResource->info.bufferIndirection.info.hostVisible.buffers[pState->frameIndexWarp];
		}
		return pResource->info.bufferIndirection.info.deviceLocal.buffer;
	}

	return handle;
}

// i'm not sure it covers all the cases, but we should be good for now
static void TRM_Backend_mergeLayouts(VkImageLayout a, VkImageLayout b, VkImageLayout* pResult)
{
	if(a == b)
	{
		*pResult = a;
		return;
	}

	if(a == VK_IMAGE_LAYOUT_GENERAL || b == VK_IMAGE_LAYOUT_GENERAL)
	{
		*pResult = VK_IMAGE_LAYOUT_GENERAL;
		return;
	}

	if(a == VK_IMAGE_LAYOUT_UNDEFINED)
	{
		*pResult = b;
		return;
	}

	if(b == VK_IMAGE_LAYOUT_UNDEFINED)
	{
		*pResult = a;
		return;
	}

	*pResult = VK_IMAGE_LAYOUT_GENERAL;
}

static VkAccessFlags TRM_Backend_convertAccessFlags(uint32_t flags)
{
	VkAccessFlags result = 0;
	if((flags & TRM_SHADER_ACCESS_FLAG_READ) != 0)
	{
		result |= VK_ACCESS_SHADER_READ_BIT;
	}

	if((flags & TRM_SHADER_ACCESS_FLAG_WRITE) != 0)
	{
		result |= VK_ACCESS_SHADER_WRITE_BIT;
	}
	return result;
}

static VkFormat TRM_Backend_convertFormat(enum TRM_Format format)
{
	if(format == TRM_SWAPCHAIN_IMAGE_FORMAT)
	{
		return pState->swapchainFormat;
	}

	switch(format)
	{
	case(TRM_FORMAT_D32_SFLOAT): return VK_FORMAT_D32_SFLOAT; break;
	case(TRM_FORMAT_R8_UNORM): return VK_FORMAT_R8_UNORM; break;
	case(TRM_FORMAT_R8_SNORM): return VK_FORMAT_R8_SNORM; break;
	case(TRM_FORMAT_R8_USCALED): return VK_FORMAT_R8_USCALED; break;
	case(TRM_FORMAT_R8_SSCALED): return VK_FORMAT_R8_SSCALED; break;
	case(TRM_FORMAT_R8_UINT): return VK_FORMAT_R8_UINT; break;
	case(TRM_FORMAT_R8_SINT): return VK_FORMAT_R8_SINT; break;
	case(TRM_FORMAT_R8_SRGB): return VK_FORMAT_R8_SRGB; break;
	case(TRM_FORMAT_R8G8_UNORM): return VK_FORMAT_R8G8_UNORM; break;
	case(TRM_FORMAT_R8G8_SNORM): return VK_FORMAT_R8G8_SNORM; break;
	case(TRM_FORMAT_R8G8_USCALED): return VK_FORMAT_R8G8_USCALED; break;
	case(TRM_FORMAT_R8G8_SSCALED): return VK_FORMAT_R8G8_SSCALED; break;
	case(TRM_FORMAT_R8G8_UINT): return VK_FORMAT_R8G8_UINT; break;
	case(TRM_FORMAT_R8G8_SINT): return VK_FORMAT_R8G8_SINT; break;
	case(TRM_FORMAT_R8G8_SRGB): return VK_FORMAT_R8G8_SRGB; break;
	case(TRM_FORMAT_R8G8B8_UNORM): return VK_FORMAT_R8G8B8_UNORM; break;
	case(TRM_FORMAT_R8G8B8_SNORM): return VK_FORMAT_R8G8B8_SNORM; break;
	case(TRM_FORMAT_R8G8B8_USCALED): return VK_FORMAT_R8G8B8_USCALED; break;
	case(TRM_FORMAT_R8G8B8_SSCALED): return VK_FORMAT_R8G8B8_SSCALED; break;
	case(TRM_FORMAT_R8G8B8_UINT): return VK_FORMAT_R8G8B8_UINT; break;
	case(TRM_FORMAT_R8G8B8_SINT): return VK_FORMAT_R8G8B8_SINT; break;
	case(TRM_FORMAT_R8G8B8_SRGB): return VK_FORMAT_R8G8B8_SRGB; break;
	case(TRM_FORMAT_B8G8R8_UNORM): return VK_FORMAT_B8G8R8_UNORM; break;
	case(TRM_FORMAT_B8G8R8_SNORM): return VK_FORMAT_B8G8R8_SNORM; break;
	case(TRM_FORMAT_B8G8R8_USCALED): return VK_FORMAT_B8G8R8_USCALED; break;
	case(TRM_FORMAT_B8G8R8_SSCALED): return VK_FORMAT_B8G8R8_SSCALED; break;
	case(TRM_FORMAT_B8G8R8_UINT): return VK_FORMAT_B8G8R8_UINT; break;
	case(TRM_FORMAT_B8G8R8_SINT): return VK_FORMAT_B8G8R8_SINT; break;
	case(TRM_FORMAT_B8G8R8_SRGB): return VK_FORMAT_B8G8R8_SRGB; break;
	case(TRM_FORMAT_R8G8B8A8_UNORM): return VK_FORMAT_R8G8B8A8_UNORM; break;
	case(TRM_FORMAT_R8G8B8A8_SNORM): return VK_FORMAT_R8G8B8A8_SNORM; break;
	case(TRM_FORMAT_R8G8B8A8_USCALED): return VK_FORMAT_R8G8B8A8_USCALED; break;
	case(TRM_FORMAT_R8G8B8A8_SSCALED): return VK_FORMAT_R8G8B8A8_SSCALED; break;
	case(TRM_FORMAT_R8G8B8A8_UINT): return VK_FORMAT_R8G8B8A8_UINT; break;
	case(TRM_FORMAT_R8G8B8A8_SINT): return VK_FORMAT_R8G8B8A8_SINT; break;
	case(TRM_FORMAT_R8G8B8A8_SRGB): return VK_FORMAT_R8G8B8A8_SRGB; break;
	case(TRM_FORMAT_B8G8R8A8_UNORM): return VK_FORMAT_B8G8R8A8_UNORM; break;
	case(TRM_FORMAT_B8G8R8A8_SNORM): return VK_FORMAT_B8G8R8A8_SNORM; break;
	case(TRM_FORMAT_B8G8R8A8_USCALED): return VK_FORMAT_B8G8R8A8_USCALED; break;
	case(TRM_FORMAT_B8G8R8A8_SSCALED): return VK_FORMAT_B8G8R8A8_SSCALED; break;
	case(TRM_FORMAT_B8G8R8A8_UINT): return VK_FORMAT_B8G8R8A8_UINT; break;
	case(TRM_FORMAT_B8G8R8A8_SINT): return VK_FORMAT_B8G8R8A8_SINT; break;
	case(TRM_FORMAT_B8G8R8A8_SRGB): return VK_FORMAT_B8G8R8A8_SRGB; break;
	case(TRM_FORMAT_R16G16_UNORM): return VK_FORMAT_R16G16_UNORM; break;
	case(TRM_FORMAT_R16G16_SNORM): return VK_FORMAT_R16G16_SNORM; break;
	case(TRM_FORMAT_R16G16_USCALED): return VK_FORMAT_R16G16_USCALED; break;
	case(TRM_FORMAT_R16G16_SSCALED): return VK_FORMAT_R16G16_SSCALED; break;
	case(TRM_FORMAT_R16G16_UINT): return VK_FORMAT_R16G16_UINT; break;
	case(TRM_FORMAT_R16G16_SINT): return VK_FORMAT_R16G16_SINT; break;
	case(TRM_FORMAT_R16G16_SFLOAT): return VK_FORMAT_R16G16_SFLOAT; break;
	case(TRM_FORMAT_R16G16B16_UNORM): return VK_FORMAT_R16G16B16_UNORM; break;
	case(TRM_FORMAT_R16G16B16_SNORM): return VK_FORMAT_R16G16B16_SNORM; break;
	case(TRM_FORMAT_R16G16B16_USCALED): return VK_FORMAT_R16G16B16_USCALED; break;
	case(TRM_FORMAT_R16G16B16_SSCALED): return VK_FORMAT_R16G16B16_SSCALED; break;
	case(TRM_FORMAT_R16G16B16_UINT): return VK_FORMAT_R16G16B16_UINT; break;
	case(TRM_FORMAT_R16G16B16_SINT): return VK_FORMAT_R16G16B16_SINT; break;
	case(TRM_FORMAT_R16G16B16_SFLOAT): return VK_FORMAT_R16G16B16_SFLOAT; break;
	case(TRM_FORMAT_R16G16B16A16_UNORM): return VK_FORMAT_R16G16B16A16_UNORM; break;
	case(TRM_FORMAT_R16G16B16A16_SNORM): return VK_FORMAT_R16G16B16A16_SNORM; break;
	case(TRM_FORMAT_R16G16B16A16_USCALED): return VK_FORMAT_R16G16B16A16_USCALED; break;
	case(TRM_FORMAT_R16G16B16A16_SSCALED): return VK_FORMAT_R16G16B16A16_SSCALED; break;
	case(TRM_FORMAT_R16G16B16A16_UINT): return VK_FORMAT_R16G16B16A16_UINT; break;
	case(TRM_FORMAT_R16G16B16A16_SINT): return VK_FORMAT_R16G16B16A16_SINT; break;
	case(TRM_FORMAT_R16G16B16A16_SFLOAT): return VK_FORMAT_R16G16B16A16_SFLOAT; break;
	case(TRM_FORMAT_R32_UINT): return VK_FORMAT_R32_UINT; break;
	case(TRM_FORMAT_R32_SINT): return VK_FORMAT_R32_SINT; break;
	case(TRM_FORMAT_R32_SFLOAT): return VK_FORMAT_R32_SFLOAT; break;
	case(TRM_FORMAT_R32G32_UINT): return VK_FORMAT_R32G32_UINT; break;
	case(TRM_FORMAT_R32G32_SINT): return VK_FORMAT_R32G32_SINT; break;
	case(TRM_FORMAT_R32G32_SFLOAT): return VK_FORMAT_R32G32_SFLOAT; break;
	case(TRM_FORMAT_R32G32B32_UINT): return VK_FORMAT_R32G32B32_UINT; break;
	case(TRM_FORMAT_R32G32B32_SINT): return VK_FORMAT_R32G32B32_SINT; break;
	case(TRM_FORMAT_R32G32B32_SFLOAT): return VK_FORMAT_R32G32B32_SFLOAT; break;
	case(TRM_FORMAT_R32G32B32A32_UINT): return VK_FORMAT_R32G32B32A32_UINT; break;
	case(TRM_FORMAT_R32G32B32A32_SINT): return VK_FORMAT_R32G32B32A32_SINT; break;
	case(TRM_FORMAT_R32G32B32A32_SFLOAT): return VK_FORMAT_R32G32B32A32_SFLOAT; break;
	default: exit(EXIT_FAILURE);
	}
}

static VkAttachmentLoadOp TRM_Backend_convertLoadOp(enum TRM_OutputLoadOp loadOp)
{
	switch(loadOp)
	{
	case(TRM_OUTPUT_LOAD_OP_LOAD): return VK_ATTACHMENT_LOAD_OP_LOAD; break;
	case(TRM_OUTPUT_LOAD_OP_CLEAR): return VK_ATTACHMENT_LOAD_OP_CLEAR; break;
	default: exit(EXIT_FAILURE);
	}
}


static VkDescriptorType TRM_Backend_convertDescriptorType(enum TRM_DescriptorType descriptorType)
{
	switch(descriptorType)
	{
	case TRM_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	case TRM_DESCRIPTOR_TYPE_STORAGE_BUFFER: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	case TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	case TRM_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	default: exit(EXIT_FAILURE);
	}
}

static VkBufferUsageFlagBits TRM_Backend_convertBufferUsage(enum TRM_BufferUsage bufferUsage)
{
	switch(bufferUsage)
	{
	case TRM_BUFFER_USAGE_UNIFORM: return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	case TRM_BUFFER_USAGE_STORAGE: return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	case TRM_BUFFER_USAGE_TRANSFER_SRC: return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	case TRM_BUFFER_USAGE_TRANSFER_DST: return VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	case TRM_BUFFER_USAGE_VERTEX: return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	case TRM_BUFFER_USAGE_INDEX: return VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	case TRM_BUFFER_USAGE_DRAW_INDIRECT: return VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
	default: exit(EXIT_FAILURE);
	}
}

static VkImageUsageFlagBits TRM_Backend_convertImageUsage(enum TRM_ImageUsage imageUsage)
{
	switch(imageUsage)
	{
	case TRM_IMAGE_USAGE_COLOR_ATTACHMENT: return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	case TRM_IMAGE_USAGE_DEPTH_ATTACHMENT: return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	case TRM_IMAGE_USAGE_SAMPLED: return VK_IMAGE_USAGE_SAMPLED_BIT;
	case TRM_IMAGE_USAGE_STORAGE: return VK_IMAGE_USAGE_STORAGE_BIT;
	case TRM_IMAGE_USAGE_TRANSFER_SRC: return VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	case TRM_IMAGE_USAGE_TRANSFER_DST: return VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	default: exit(EXIT_FAILURE);
	}
}

static void TRM_Backend_createResource(struct TRM_ResourceCreateInfo info, uint32_t* pHandle)
{
	if(info.type == TRM_RESOURCE_TYPE_BUFFER)
	{
		VkBufferUsageFlags bufferUsage = 0;
		for(uint32_t bit = 1; bit < TRM_BUFFER_USAGE_MAX; bit <<= 1)
		{
			if((info.info.buffer.usage & bit) != 0)
			{
				bufferUsage |= TRM_Backend_convertBufferUsage((info.info.buffer.usage & bit));
			}
		}

		struct TRM_Backend_Resource bufferIndirection = {0};
		bufferIndirection.type = TRM_BACKEND_RESOURCE_TYPE_BUFFER_INDIRECTION;
		bufferIndirection.info.bufferIndirection.hostVisible = info.info.buffer.hostVisible;
		bufferIndirection.toDelete = false;
		bufferIndirection.lastFrameIndexNoWarp = 1;

		if(info.info.buffer.hostVisible)
		{
			for(uint32_t frameIndexWarp = 0; frameIndexWarp < TRM_BACKEND_FRAME_COUNT; ++frameIndexWarp)
			{
				struct TRM_Backend_Resource resource = {0};
				resource.type = TRM_BACKEND_RESOURCE_TYPE_BUFFER;
				resource.state.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				resource.state.access = VK_ACCESS_NONE;

				TRM_Backend_createBuffer(
					pState->pAllocator,
					pState->device,
					info.info.buffer.sizeInBytes,
					bufferUsage,
					pState->queueFamilyIndex,
					&resource.info.buffer.buffer);

				TRM_Backend_allocateMemoryForBuffer(
					pState->pAllocator,
					pState->physicalDevice,
					pState->device,
					resource.info.buffer.buffer,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					&resource.info.buffer.memory);

				vkBindBufferMemory(
					pState->device,
					resource.info.buffer.buffer,
					resource.info.buffer.memory,
					0);

				uint32_t buffer = 0;
				TRM_Arena_add((void*)&resource, &pState->resourcePool, &buffer);
				bufferIndirection.info.bufferIndirection.info.hostVisible.buffers[frameIndexWarp] = buffer;
			}
		}
		else
		{
			struct TRM_Backend_Resource resource = {0};
			resource.type = TRM_BACKEND_RESOURCE_TYPE_BUFFER;
			resource.state.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			resource.state.access = VK_ACCESS_NONE;
			bufferIndirection.toDelete = false;
			bufferIndirection.lastFrameIndexNoWarp = 1;

			TRM_Backend_createBuffer(
				pState->pAllocator,
				pState->device,
				info.info.buffer.sizeInBytes,
				bufferUsage,
				pState->queueFamilyIndex,
				&resource.info.buffer.buffer);

			TRM_Backend_allocateMemoryForBuffer(
				pState->pAllocator,
				pState->physicalDevice,
				pState->device,
				resource.info.buffer.buffer,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				&resource.info.buffer.memory);

			vkBindBufferMemory(
				pState->device,
				resource.info.buffer.buffer,
				resource.info.buffer.memory,
				0);

			uint32_t buffer = 0;
			TRM_Arena_add((void*)&resource, &pState->resourcePool, &buffer);
			bufferIndirection.info.bufferIndirection.info.deviceLocal.buffer = buffer;
		}

		TRM_Arena_add((void*)&bufferIndirection, &pState->resourcePool, pHandle);
	}
	else
	{
		VkImageUsageFlags imageUsage = 0;
		for(uint32_t bit = 1; bit < TRM_IMAGE_USAGE_MAX; bit <<= 1)
		{
			if((info.info.image.usage & bit) != 0)
			{
				imageUsage |= TRM_Backend_convertImageUsage((info.info.image.usage & bit));
			}
		}

		// maybe we should expose that explicitly
		VkImageAspectFlags imageAspect = 0;
		if(info.info.image.usage & TRM_IMAGE_USAGE_COLOR_ATTACHMENT)
		{
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if(info.info.image.usage & TRM_IMAGE_USAGE_DEPTH_ATTACHMENT)
		{
			imageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
		}

		if(imageAspect == 0) // default
		{
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		struct TRM_Backend_Resource resource = {0};
		resource.type = TRM_BACKEND_RESOURCE_TYPE_IMAGE;
		resource.toDelete = false;
		resource.lastFrameIndexNoWarp = 1;
		resource.info.image.aspect = imageAspect;
		resource.info.image.swapchainImage = false;
		resource.info.image.mipCount = info.info.image.mipCount;
		resource.state.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		resource.state.access = VK_ACCESS_NONE;
		resource.state.layout = VK_IMAGE_LAYOUT_UNDEFINED;

		TRM_Backend_createImage(
			pState->pAllocator,
			pState->device,
			info.info.image.width,
			info.info.image.height,
			info.info.image.mipCount,
			TRM_Backend_convertFormat(info.info.image.format),
			resource.state.layout,
			imageUsage,
			pState->queueFamilyIndex,
			&resource.info.image.image);

		TRM_Backend_allocateMemoryForImage(
			pState->pAllocator,
			pState->physicalDevice,
			pState->device,
			resource.info.image.image,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&resource.info.image.memory);

		vkBindImageMemory(pState->device, resource.info.image.image, resource.info.image.memory, 0);

		TRM_Backend_createImageView(
			pState->pAllocator,
			pState->device,
			resource.info.image.image,
			0,
			resource.info.image.mipCount,
			TRM_Backend_convertFormat(info.info.image.format),
			resource.info.image.aspect,
			&resource.info.image.allMipsImageView);

		for(uint32_t i = 0; i < resource.info.image.mipCount; ++i)
		{
			TRM_Backend_createImageView(
				pState->pAllocator,
				pState->device,
				resource.info.image.image,
				i,
				1,
				TRM_Backend_convertFormat(info.info.image.format),
				resource.info.image.aspect,
				&resource.info.image.singleMipImageViews[i]);
		}

		TRM_Arena_add((void*)&resource, &pState->resourcePool, pHandle);
	}

}

static void TRM_Backend_destroyResource(uint32_t handle)
{
	struct TRM_Backend_Resource* pResource = NULL;
	TRM_Arena_get(handle, pState->resourcePool, (void**)&pResource);

	if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_BUFFER_INDIRECTION)
	{
		if(pResource->info.bufferIndirection.hostVisible)
		{
			for(uint32_t frameIndexWarp = 0; frameIndexWarp < TRM_BACKEND_FRAME_COUNT; ++frameIndexWarp)
			{
				struct TRM_Backend_Resource* pBuffer = NULL;
				TRM_Arena_get(pResource->info.bufferIndirection.info.hostVisible.buffers[frameIndexWarp], pState->resourcePool, (void**)&pBuffer);
				vkDestroyBuffer(pState->device, pBuffer->info.buffer.buffer, pState->pAllocator);
				vkFreeMemory(pState->device, pBuffer->info.buffer.memory, pState->pAllocator);
				TRM_Arena_remove(pResource->info.bufferIndirection.info.hostVisible.buffers[frameIndexWarp], &pState->resourcePool);
			}
		}
		else
		{
			struct TRM_Backend_Resource* pBuffer = NULL;
			TRM_Arena_get(pResource->info.bufferIndirection.info.deviceLocal.buffer, pState->resourcePool, (void**)&pBuffer);
			vkDestroyBuffer(pState->device, pBuffer->info.buffer.buffer, pState->pAllocator);
			vkFreeMemory(pState->device, pBuffer->info.buffer.memory, pState->pAllocator);
			TRM_Arena_remove(pResource->info.bufferIndirection.info.deviceLocal.buffer, &pState->resourcePool);
		}
	}
	else
	{
		if(!pResource->info.image.swapchainImage)
		{
			vkDestroyImage(pState->device, pResource->info.image.image, pState->pAllocator);
			vkFreeMemory(pState->device, pResource->info.image.memory, pState->pAllocator);

			for(uint32_t i = 0; i < pResource->info.image.mipCount; ++i)
			{
				vkDestroyImageView(pState->device, pResource->info.image.singleMipImageViews[i], pState->pAllocator);
			}
		}
		
		vkDestroyImageView(pState->device, pResource->info.image.allMipsImageView, pState->pAllocator);
	}

	TRM_Arena_remove(handle, &pState->resourcePool);
}

static void TRM_Backend_recreateSwapchain(uint32_t width, uint32_t height)
{
	vkDeviceWaitIdle(pState->device);

	if(pState->swapchain != VK_NULL_HANDLE)
	{
		for(uint32_t i = 0; i < pState->swapchainImageCount; ++i)
		{
			struct TRM_Backend_Resource* pResource = NULL;
			TRM_Arena_get(pState->pSwapchainImageInfos[i].colorImage, pState->resourcePool, (void**)&pResource);
			TRM_Backend_destroyResource(pState->pSwapchainImageInfos[i].colorImage);
			vkDestroySemaphore(pState->device, pState->pSwapchainImageInfos[i].imageRenderedSemaphore, pState->pAllocator);
		}

		vkDestroySwapchainKHR(pState->device, pState->swapchain, pState->pAllocator);

		TRM_Memory_deallocate(pState->pSwapchainImageInfos);
	}

	TRM_Backend_createSwapchain(
		pState->pAllocator,
		pState->physicalDevice,
		pState->device,
		pState->surface,
		width,
		height,
		false,
		pState->queueFamilyIndex,
		&pState->swapchain,
		&pState->swapchainFormat,
		&pState->swapchainWidth,
		&pState->swapchainHeight);

	VkImage* pSwapchainImages = NULL;
	vkGetSwapchainImagesKHR(pState->device, pState->swapchain, &pState->swapchainImageCount, NULL);
	TRM_Memory_allocate(sizeof(VkImage) * pState->swapchainImageCount, (void**)&pSwapchainImages);
	vkGetSwapchainImagesKHR(pState->device, pState->swapchain, &pState->swapchainImageCount, pSwapchainImages);

	TRM_Memory_allocate(
		sizeof(struct TRM_Backend_SwapchainImageInfo) * pState->swapchainImageCount,
		(void**)&pState->pSwapchainImageInfos);

	for(uint32_t i = 0; i < pState->swapchainImageCount; ++i)
	{
		struct TRM_Backend_Resource swapchainColorImage = {0};
		swapchainColorImage.type = TRM_BACKEND_RESOURCE_TYPE_IMAGE;
		swapchainColorImage.info.image.image = pSwapchainImages[i];

		TRM_Backend_createImageView(
			pState->pAllocator,
			pState->device,
			pSwapchainImages[i],
			0,
			1,
			pState->swapchainFormat,
			VK_IMAGE_ASPECT_COLOR_BIT,
			&swapchainColorImage.info.image.allMipsImageView);

		swapchainColorImage.info.image.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		swapchainColorImage.info.image.memory = VK_NULL_HANDLE;
		swapchainColorImage.info.image.swapchainImage = true;
		swapchainColorImage.info.image.mipCount = 1;
		swapchainColorImage.state.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		swapchainColorImage.state.access = VK_ACCESS_NONE;
		swapchainColorImage.state.layout = VK_IMAGE_LAYOUT_UNDEFINED;

		TRM_Arena_add(
			(void*)&swapchainColorImage,
			&pState->resourcePool,
			&pState->pSwapchainImageInfos[i].colorImage);

		TRM_Backend_createSemaphore(
			pState->pAllocator,
			pState->device,
			&pState->pSwapchainImageInfos[i].imageRenderedSemaphore);
	}

	TRM_Memory_deallocate(pSwapchainImages);
}

static void TRM_Backend_updateDescriptorSet(
	uint32_t descriptorCount, 
	const struct TRM_DescriptorInfo* pDescriptorInfos, 
	const struct TRM_Backend_Binding* pBindings,
	VkDescriptorSet descriptorSet,
	uint32_t swapchainImageIndex)
{
	VkWriteDescriptorSet writes[TRM_MAX_DESCRIPTOR_COUNT];
	VkDescriptorBufferInfo bufferInfos[TRM_MAX_DESCRIPTOR_COUNT];
	VkDescriptorImageInfo imageInfos[TRM_MAX_DESCRIPTOR_COUNT];

	for(uint32_t i = 0; i < descriptorCount; ++i)
	{
		VkDescriptorType descriptorType = TRM_Backend_convertDescriptorType(pDescriptorInfos[i].descriptorType);
		uint32_t resource = TRM_Backend_translateResource(pBindings[i].resource, swapchainImageIndex);
		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].pNext = NULL;
		writes[i].dstBinding = i;
		writes[i].dstArrayElement = 0;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = descriptorType;

		if(descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
		{
			bufferInfos[i].buffer = pResource->info.buffer.buffer;
			bufferInfos[i].offset = 0;
			bufferInfos[i].range = VK_WHOLE_SIZE;

			writes[i].pBufferInfo = &bufferInfos[i];
			writes[i].pImageInfo = NULL;
			writes[i].pTexelBufferView = NULL;

		}
		else if(descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
		{
			imageInfos[i].imageLayout = pBindings[i].layout;
			imageInfos[i].imageView = pResource->info.image.singleMipImageViews[pBindings[i].mip];
			imageInfos[i].sampler = VK_NULL_HANDLE;

			writes[i].pBufferInfo = NULL;
			writes[i].pImageInfo = &imageInfos[i];
		}
		else if(descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
		{
			imageInfos[i].imageLayout = pBindings[i].layout;
			imageInfos[i].imageView = pResource->info.image.allMipsImageView;
			imageInfos[i].sampler = pState->globalSampler;

			writes[i].pBufferInfo = NULL;
			writes[i].pImageInfo = &imageInfos[i];
		}
		else
		{
			exit(EXIT_FAILURE);
		}

		writes[i].dstSet = descriptorSet;
		writes[i].pTexelBufferView = NULL;
	}

	vkUpdateDescriptorSets(pState->device, descriptorCount, writes, 0, NULL);
}

static void TRM_Backend_addExpectedResourceState(struct TRM_Backend_ExpectedResourceState expectedResourceState, struct TRM_DynamicArray* pExpectedResourceStates)
{
	for(uint32_t i = 0; i < pExpectedResourceStates->elementCount; ++i)
	{
		struct TRM_Backend_ExpectedResourceState* pCurrentExpectedResourceState = NULL;
		TRM_DynamicArray_getPtrAt(i, pExpectedResourceStates, (void**)&pCurrentExpectedResourceState);

		if(expectedResourceState.resource == pCurrentExpectedResourceState->resource)
		{
			// merge states
			pCurrentExpectedResourceState->state.access |= expectedResourceState.state.access;
			TRM_Backend_mergeLayouts(pCurrentExpectedResourceState->state.layout, expectedResourceState.state.layout, &pCurrentExpectedResourceState->state.layout);
			pCurrentExpectedResourceState->state.stage |= expectedResourceState.state.stage;
			return;
		}
	}

	TRM_DynamicArray_push(&expectedResourceState, pExpectedResourceStates);
}

static struct TRM_Backend_Pass TRM_Backend_createDispatchPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_DISPATCH;
	backendPass.info.dispatch.pipeline = pass.info.dispatch.pipeline;
	backendPass.info.dispatch.groupCountX = pass.info.dispatch.groupCountX;
	backendPass.info.dispatch.groupCountY = pass.info.dispatch.groupCountY;
	backendPass.info.dispatch.groupCountZ = pass.info.dispatch.groupCountZ;

	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	// merge all resource states
	for(uint32_t i = 0; i < pass.info.dispatch.bindingCount; ++i)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.dispatch.bindings[i].resource, swapchainImageIndex);
		
		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pResource);
		struct TRM_Backend_Pipeline* pPipeline = NULL;
		TRM_Arena_get(backendPass.info.dispatch.pipeline, pState->pipelinePool, (void**)&pPipeline);

		VkAccessFlags accessFlags = TRM_Backend_convertAccessFlags(pPipeline->info.compute.descriptorInfos[i].resourceAccessFlags);

		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
		{
			layout =
				pPipeline->info.compute.descriptorInfos[i].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
				VK_IMAGE_LAYOUT_GENERAL :
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		
		expectedResourceState.state.layout = layout;
		expectedResourceState.state.access = accessFlags;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
	}

	// bindings
	uint32_t bindingCount = pass.info.dispatch.bindingCount;
	struct TRM_Backend_Binding bindings[TRM_MAX_DESCRIPTOR_COUNT];
	
	for(uint32_t i = 0; i < bindingCount; ++i)
	{
		struct TRM_Backend_Binding binding = {0};
		binding.resource = TRM_Backend_translateResource(pass.info.dispatch.bindings[i].resource, swapchainImageIndex);
		binding.mip = pass.info.dispatch.bindings[i].mip;

		// find merged resource state
		for(uint32_t y = 0; y < backendPass.expectedResourceStates.elementCount; ++y)
		{
			struct TRM_Backend_ExpectedResourceState mergedResourceState;
			TRM_DynamicArray_at(y, &backendPass.expectedResourceStates, &mergedResourceState);
			if(mergedResourceState.resource == binding.resource)
			{
				binding.layout = mergedResourceState.state.layout;
				break;
			}
		}
		bindings[i] = binding;
	}

	// update descriptor set
	struct TRM_Backend_Pipeline* pPipeline = NULL;
	TRM_Arena_get(backendPass.info.dispatch.pipeline, pState->pipelinePool, (void**)&pPipeline);

	VkDescriptorSet* pDescriptorSet =
		&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount];

	TRM_Backend_allocateDescriptorSet(
		pState->device,
		pState->descriptorPool,
		pPipeline->info.compute.descriptorSetLayout,
		pDescriptorSet);

	backendPass.info.dispatch.descriptorSet = pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount;

	pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount += 1;

	TRM_Backend_updateDescriptorSet(
		pPipeline->info.compute.descriptorInfoCount,
		pPipeline->info.compute.descriptorInfos,
		bindings,
		*pDescriptorSet,
		swapchainImageIndex);
	
	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createDrawVertexPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_DRAW_VERTEX;
	backendPass.info.drawVertex.pipeline = pass.info.drawVertex.pipeline;
	backendPass.info.drawVertex.width = pass.info.drawVertex.width;
	backendPass.info.drawVertex.height = pass.info.drawVertex.height;
	backendPass.info.drawVertex.useVertexBuffer = pass.info.drawVertex.useVertexBuffer;
	backendPass.info.drawVertex.vertexCount = pass.info.drawVertex.vertexCount;
	backendPass.info.drawVertex.clearColorCount = pass.info.drawVertex.colorOutputImageCount + 1;
	
	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);


	struct TRM_Backend_Pipeline* pPipeline = NULL;
	TRM_Arena_get(backendPass.info.drawVertex.pipeline, pState->pipelinePool, (void**)&pPipeline);

	// gather all expected resource states + bindings
	for(uint32_t i = 0; i < pass.info.drawVertex.bindingCount; ++i)
	{
		const uint32_t resource = TRM_Backend_translateResource(pass.info.drawVertex.bindings[i].resource, swapchainImageIndex);

		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = resource;

		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

		VkAccessFlags accessFlags = TRM_Backend_convertAccessFlags(pPipeline->info.graphics.descriptorInfos[i].resourceAccessFlags);

		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
		{
			layout =
				pPipeline->info.graphics.descriptorInfos[i].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
				VK_IMAGE_LAYOUT_GENERAL :
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		expectedResourceState.state.layout = layout;
		expectedResourceState.state.access = accessFlags;
		expectedResourceState.state.stage = (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
	}

	// vertex buffer
	if(pass.info.drawVertex.useVertexBuffer)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertex.vertexBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawVertex.vertexBuffer = expectedResourceState.resource;
	}

	VkImageView attachments[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	
	// color attachments
	for(uint32_t i = 0; i < pass.info.drawVertex.colorOutputImageCount; ++i)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertex.colorOutputImages[i], swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pColorImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pColorImage);
		attachments[i] = pColorImage->info.image.singleMipImageViews[0];
	}

	// depth attachment
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertex.depthOutputImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pDepthImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pDepthImage);
		attachments[pass.info.drawVertex.colorOutputImageCount] = pDepthImage->info.image.singleMipImageViews[0];
	}

	// clear colors
	{
		for(uint32_t i = 0; i < pass.info.drawVertex.colorOutputImageCount; ++i)
		{
			TRM_Memory_memcpy(
				sizeof(float) * 4,
				pass.info.drawVertex.clearColors[i].color,
				backendPass.info.drawVertex.clearColors[i].color.float32);
		}
		backendPass.info.drawVertex.clearColors[pass.info.drawVertex.colorOutputImageCount].depthStencil.depth = 1.0f;
		backendPass.info.drawVertex.clearColors[pass.info.drawVertex.colorOutputImageCount].depthStencil.stencil = 0;
	}
	
	// framebuffer
	TRM_Backend_createFramebuffer(
		pState->pAllocator,
		pState->device,
		pPipeline->info.graphics.renderPass,
		pass.info.drawVertex.colorOutputImageCount + 1,
		attachments,
		backendPass.info.drawVertex.width,
		backendPass.info.drawVertex.height,
		&pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pState->pFrameInfos[pState->frameIndexWarp].framebufferCount]);


	backendPass.info.drawVertex.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebufferCount;
	pState->pFrameInfos[pState->frameIndexWarp].framebufferCount += 1;

	// bindings
	uint32_t bindingCount = pass.info.drawVertex.bindingCount;
	struct TRM_Backend_Binding bindings[TRM_MAX_DESCRIPTOR_COUNT];

	for(uint32_t i = 0; i < bindingCount; ++i)
	{
		struct TRM_Backend_Binding binding = {0};
		binding.resource = TRM_Backend_translateResource(pass.info.drawVertex.bindings[i].resource, swapchainImageIndex);
		binding.mip = pass.info.drawVertex.bindings[i].mip;

		// find merged resource state
		for(uint32_t y = 0; y < backendPass.expectedResourceStates.elementCount; ++y)
		{
			struct TRM_Backend_ExpectedResourceState mergedResourceState;
			TRM_DynamicArray_at(y, &backendPass.expectedResourceStates, &mergedResourceState);
			if(mergedResourceState.resource == binding.resource)
			{
				binding.layout = mergedResourceState.state.layout;
				break;
			}
		}
		bindings[i] = binding;
	}

	// update descriptor set
	VkDescriptorSet* pDescriptorSet =
		&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount];
			
	TRM_Backend_allocateDescriptorSet(
		pState->device,
		pState->descriptorPool,
		pPipeline->info.graphics.descriptorSetLayout,
		pDescriptorSet);
			
	backendPass.info.drawVertex.descriptorSet = pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount;
			
	pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount += 1;

	TRM_Backend_updateDescriptorSet(
		pPipeline->info.graphics.descriptorInfoCount, 
		pPipeline->info.graphics.descriptorInfos, 
		bindings,
		*pDescriptorSet,
		swapchainImageIndex);
	
	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createDrawIndexedPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_DRAW_INDEXED;
	backendPass.info.drawIndexed.pipeline = pass.info.drawIndexed.pipeline;
	backendPass.info.drawIndexed.width = pass.info.drawIndexed.width;
	backendPass.info.drawIndexed.height = pass.info.drawIndexed.height;
	backendPass.info.drawIndexed.indexCount = pass.info.drawIndexed.indexCount;
	backendPass.info.drawIndexed.clearColorCount = pass.info.drawIndexed.colorOutputImageCount + 1;

	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	struct TRM_Backend_Pipeline* pPipeline = NULL;
	TRM_Arena_get(backendPass.info.drawIndexed.pipeline, pState->pipelinePool, (void**)&pPipeline);

	// bindings
	for(uint32_t i = 0; i < pass.info.drawIndexed.bindingCount; ++i)
	{
		const uint32_t resource = TRM_Backend_translateResource(pass.info.drawIndexed.bindings[i].resource, swapchainImageIndex);

		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = resource;

		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

		VkAccessFlags accessFlags = TRM_Backend_convertAccessFlags(pPipeline->info.graphics.descriptorInfos[i].resourceAccessFlags);

		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
		{
			layout =
				pPipeline->info.graphics.descriptorInfos[i].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
				VK_IMAGE_LAYOUT_GENERAL :
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		expectedResourceState.state.layout = layout;
		expectedResourceState.state.access = accessFlags;
		expectedResourceState.state.stage = (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
	}

	// vertex buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexed.vertexBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawIndexed.vertexBuffer = expectedResourceState.resource;
	}

	// index buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexed.indexBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_INDEX_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawIndexed.indexBuffer = expectedResourceState.resource;
	}

	VkImageView attachments[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	
	// color attachments
	for(uint32_t i = 0; i < pass.info.drawIndexed.colorOutputImageCount; ++i)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexed.colorOutputImages[i], swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pColorImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pColorImage);
		attachments[i] = pColorImage->info.image.singleMipImageViews[0];
	}

	// depth attachment
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexed.depthOutputImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pDepthImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pDepthImage);
		attachments[pass.info.drawIndexed.colorOutputImageCount] = pDepthImage->info.image.singleMipImageViews[0];
	}

	// clear colors
	{
		for(uint32_t i = 0; i < pass.info.drawIndexed.colorOutputImageCount; ++i)
		{
			TRM_Memory_memcpy(
				sizeof(float) * 4,
				pass.info.drawIndexed.clearColors[i].color,
				backendPass.info.drawIndexed.clearColors[i].color.float32);
		}
		backendPass.info.drawIndexed.clearColors[pass.info.drawIndexed.colorOutputImageCount].depthStencil.depth = 1.0f;
		backendPass.info.drawIndexed.clearColors[pass.info.drawIndexed.colorOutputImageCount].depthStencil.stencil = 0;
	}

	// framebuffer
	TRM_Backend_createFramebuffer(
		pState->pAllocator,
		pState->device,
		pPipeline->info.graphics.renderPass,
		pass.info.drawIndexed.colorOutputImageCount + 1,
		attachments,
		backendPass.info.drawIndexed.width,
		backendPass.info.drawIndexed.height,
		&pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pState->pFrameInfos[pState->frameIndexWarp].framebufferCount]);

	backendPass.info.drawIndexed.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebufferCount;
	pState->pFrameInfos[pState->frameIndexWarp].framebufferCount += 1;

	// bindings
	uint32_t bindingCount = pass.info.drawIndexed.bindingCount;
	struct TRM_Backend_Binding bindings[TRM_MAX_DESCRIPTOR_COUNT];

	for(uint32_t i = 0; i < bindingCount; ++i)
	{
		struct TRM_Backend_Binding binding = {0};
		binding.resource = TRM_Backend_translateResource(pass.info.drawIndexed.bindings[i].resource, swapchainImageIndex);
		binding.mip = pass.info.drawIndexed.bindings[i].mip;

		// find merged resource state
		for(uint32_t y = 0; y < backendPass.expectedResourceStates.elementCount; ++y)
		{
			struct TRM_Backend_ExpectedResourceState mergedResourceState;
			TRM_DynamicArray_at(y, &backendPass.expectedResourceStates, &mergedResourceState);
			if(mergedResourceState.resource == binding.resource)
			{
				binding.layout = mergedResourceState.state.layout;
				break;
			}
		}
		bindings[i] = binding;
	}

	// update descriptor set
	VkDescriptorSet* pDescriptorSet =
		&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount];

	TRM_Backend_allocateDescriptorSet(
		pState->device,
		pState->descriptorPool,
		pPipeline->info.graphics.descriptorSetLayout,
		pDescriptorSet);

	backendPass.info.drawIndexed.descriptorSet = pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount;

	pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount += 1;

	TRM_Backend_updateDescriptorSet(
		pPipeline->info.graphics.descriptorInfoCount,
		pPipeline->info.graphics.descriptorInfos,
		bindings,
		*pDescriptorSet,
		swapchainImageIndex);

	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createDrawVertexIndirectPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_DRAW_VERTEX_INDIRECT;
	backendPass.info.drawVertexIndirect.pipeline = pass.info.drawVertexIndirect.pipeline;
	backendPass.info.drawVertexIndirect.width = pass.info.drawVertexIndirect.width;
	backendPass.info.drawVertexIndirect.height = pass.info.drawVertexIndirect.height;
	backendPass.info.drawVertexIndirect.useVertexBuffer = pass.info.drawVertexIndirect.useVertexBuffer;
	backendPass.info.drawVertexIndirect.drawCount = pass.info.drawVertexIndirect.drawCount;
	backendPass.info.drawVertexIndirect.clearColorCount = pass.info.drawVertexIndirect.colorOutputImageCount + 1;

	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);


	struct TRM_Backend_Pipeline* pPipeline = NULL;
	TRM_Arena_get(backendPass.info.drawVertexIndirect.pipeline, pState->pipelinePool, (void**)&pPipeline);

	// bindings
	for(uint32_t i = 0; i < pass.info.drawVertexIndirect.bindingCount; ++i)
	{
		const uint32_t resource = TRM_Backend_translateResource(pass.info.drawVertexIndirect.bindings[i].resource, swapchainImageIndex);

		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = resource;

		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

		VkAccessFlags accessFlags = TRM_Backend_convertAccessFlags(pPipeline->info.graphics.descriptorInfos[i].resourceAccessFlags);

		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
		{
			layout =
				pPipeline->info.graphics.descriptorInfos[i].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
				VK_IMAGE_LAYOUT_GENERAL :
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		expectedResourceState.state.layout = layout;
		expectedResourceState.state.access = accessFlags;
		expectedResourceState.state.stage = (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
	}

	// indirect command buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertexIndirect.buffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawVertexIndirect.indirectCommandBuffer = expectedResourceState.resource;
	}

	// vertex buffer
	if(pass.info.drawVertexIndirect.useVertexBuffer)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertexIndirect.vertexBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawVertexIndirect.vertexBuffer = expectedResourceState.resource;
	}

	VkImageView attachments[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	
	// color attachments
	for(uint32_t i = 0; i < pass.info.drawVertexIndirect.colorOutputImageCount; ++i)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertexIndirect.colorOutputImages[i], swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pColorImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pColorImage);
		attachments[i] = pColorImage->info.image.singleMipImageViews[0];
	}

	// depth attachment
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawVertexIndirect.depthOutputImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pDepthImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pDepthImage);
		attachments[pass.info.drawVertexIndirect.colorOutputImageCount] = pDepthImage->info.image.singleMipImageViews[0];
	}

	// clear colors
	{
		for(uint32_t i = 0; i < pass.info.drawVertexIndirect.colorOutputImageCount; ++i)
		{
			TRM_Memory_memcpy(
				sizeof(float) * 4,
				pass.info.drawVertexIndirect.clearColors[i].color,
				backendPass.info.drawVertexIndirect.clearColors[i].color.float32);
		}
		backendPass.info.drawVertexIndirect.clearColors[pass.info.drawVertexIndirect.colorOutputImageCount].depthStencil.depth = 1.0f;
		backendPass.info.drawVertexIndirect.clearColors[pass.info.drawVertexIndirect.colorOutputImageCount].depthStencil.stencil = 0;
	}

	// framebuffer
	TRM_Backend_createFramebuffer(
		pState->pAllocator,
		pState->device,
		pPipeline->info.graphics.renderPass,
		pass.info.drawVertexIndirect.colorOutputImageCount + 1,
		attachments,
		backendPass.info.drawVertexIndirect.width,
		backendPass.info.drawVertexIndirect.height,
		&pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pState->pFrameInfos[pState->frameIndexWarp].framebufferCount]);

	backendPass.info.drawVertexIndirect.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebufferCount;
	pState->pFrameInfos[pState->frameIndexWarp].framebufferCount += 1;

	// bindings
	uint32_t bindingCount = pass.info.drawVertexIndirect.bindingCount;
	struct TRM_Backend_Binding bindings[TRM_MAX_DESCRIPTOR_COUNT];

	for(uint32_t i = 0; i < bindingCount; ++i)
	{
		struct TRM_Backend_Binding binding = {0};
		binding.resource = TRM_Backend_translateResource(pass.info.drawVertexIndirect.bindings[i].resource, swapchainImageIndex);
		binding.mip = pass.info.drawVertexIndirect.bindings[i].mip;

		// find merged resource state
		for(uint32_t y = 0; y < backendPass.expectedResourceStates.elementCount; ++y)
		{
			struct TRM_Backend_ExpectedResourceState mergedResourceState;
			TRM_DynamicArray_at(y, &backendPass.expectedResourceStates, &mergedResourceState);
			if(mergedResourceState.resource == binding.resource)
			{
				binding.layout = mergedResourceState.state.layout;
				break;
			}
		}
		bindings[i] = binding;
	}

	// update descriptor set
	VkDescriptorSet* pDescriptorSet =
		&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount];

	TRM_Backend_allocateDescriptorSet(
		pState->device,
		pState->descriptorPool,
		pPipeline->info.graphics.descriptorSetLayout,
		pDescriptorSet);

	backendPass.info.drawVertexIndirect.descriptorSet = pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount;

	pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount += 1;

	TRM_Backend_updateDescriptorSet(
		pPipeline->info.graphics.descriptorInfoCount,
		pPipeline->info.graphics.descriptorInfos,
		bindings,
		*pDescriptorSet,
		swapchainImageIndex);

	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createDrawIndexedIndirectPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_DRAW_INDEXED_INDIRECT;
	backendPass.info.drawIndexedIndirect.pipeline = pass.info.drawIndexedIndirect.pipeline;
	backendPass.info.drawIndexedIndirect.width = pass.info.drawIndexedIndirect.width;
	backendPass.info.drawIndexedIndirect.height = pass.info.drawIndexedIndirect.height;
	backendPass.info.drawIndexedIndirect.drawCount = pass.info.drawIndexedIndirect.drawCount;
	backendPass.info.drawIndexedIndirect.clearColorCount = pass.info.drawIndexedIndirect.colorOutputImageCount + 1;

	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	struct TRM_Backend_Pipeline* pPipeline = NULL;
	TRM_Arena_get(backendPass.info.drawIndexedIndirect.pipeline, pState->pipelinePool, (void**)&pPipeline);

	// bindings
	for(uint32_t i = 0; i < pass.info.drawIndexedIndirect.bindingCount; ++i)
	{
		const uint32_t resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.bindings[i].resource, swapchainImageIndex);

		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = resource;

		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

		VkAccessFlags accessFlags = TRM_Backend_convertAccessFlags(pPipeline->info.graphics.descriptorInfos[i].resourceAccessFlags);

		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
		{
			layout =
				pPipeline->info.graphics.descriptorInfos[i].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
				VK_IMAGE_LAYOUT_GENERAL :
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		expectedResourceState.state.layout = layout;
		expectedResourceState.state.access = accessFlags;
		expectedResourceState.state.stage = (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
	}

	// indirect command buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.buffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawIndexedIndirect.indirectCommandBuffer = expectedResourceState.resource;
	}

	// vertex buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.vertexBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawIndexedIndirect.vertexBuffer = expectedResourceState.resource;
	}

	// index buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.indexBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_INDEX_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.drawIndexedIndirect.indexBuffer = expectedResourceState.resource;
	}

	VkImageView attachments[TRM_MAX_COLOR_OUTPUT_COUNT + 1];
	
	// color attachments
	for(uint32_t i = 0; i < pass.info.drawIndexedIndirect.colorOutputImageCount; ++i)
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.colorOutputImages[i], swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pColorImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pColorImage);
		attachments[i] = pColorImage->info.image.singleMipImageViews[0];
	}

	// depth attachment
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.depthOutputImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);

		struct TRM_Backend_Resource* pDepthImage = NULL;
		TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pDepthImage);
		attachments[pass.info.drawIndexedIndirect.colorOutputImageCount] = pDepthImage->info.image.singleMipImageViews[0];
	}

	// clear colors
	{
		for(uint32_t i = 0; i < pass.info.drawIndexedIndirect.colorOutputImageCount; ++i)
		{
			TRM_Memory_memcpy(
				sizeof(float) * 4,
				pass.info.drawIndexedIndirect.clearColors[i].color,
				backendPass.info.drawIndexedIndirect.clearColors[i].color.float32);
		}
		backendPass.info.drawIndexedIndirect.clearColors[pass.info.drawIndexedIndirect.colorOutputImageCount].depthStencil.depth = 1.0f;
		backendPass.info.drawIndexedIndirect.clearColors[pass.info.drawIndexedIndirect.colorOutputImageCount].depthStencil.stencil = 0;
	}

	// framebuffer
	TRM_Backend_createFramebuffer(
		pState->pAllocator,
		pState->device,
		pPipeline->info.graphics.renderPass,
		pass.info.drawIndexedIndirect.colorOutputImageCount + 1,
		attachments,
		backendPass.info.drawIndexedIndirect.width,
		backendPass.info.drawIndexedIndirect.height,
		&pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pState->pFrameInfos[pState->frameIndexWarp].framebufferCount]);

	backendPass.info.drawIndexedIndirect.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebufferCount;
	pState->pFrameInfos[pState->frameIndexWarp].framebufferCount += 1;

	// bindings
	uint32_t bindingCount = pass.info.drawIndexedIndirect.bindingCount;
	struct TRM_Backend_Binding bindings[TRM_MAX_DESCRIPTOR_COUNT];

	for(uint32_t i = 0; i < bindingCount; ++i)
	{
		struct TRM_Backend_Binding binding = {0};
		binding.resource = TRM_Backend_translateResource(pass.info.drawIndexedIndirect.bindings[i].resource, swapchainImageIndex);
		binding.mip = pass.info.drawIndexedIndirect.bindings[i].mip;

		// find merged resource state
		for(uint32_t y = 0; y < backendPass.expectedResourceStates.elementCount; ++y)
		{
			struct TRM_Backend_ExpectedResourceState mergedResourceState;
			TRM_DynamicArray_at(y, &backendPass.expectedResourceStates, &mergedResourceState);
			if(mergedResourceState.resource == binding.resource)
			{
				binding.layout = mergedResourceState.state.layout;
				break;
			}
		}
		bindings[i] = binding;
	}

	// update descriptor set
	VkDescriptorSet* pDescriptorSet =
		&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount];

	TRM_Backend_allocateDescriptorSet(
		pState->device,
		pState->descriptorPool,
		pPipeline->info.graphics.descriptorSetLayout,
		pDescriptorSet);

	backendPass.info.drawIndexedIndirect.descriptorSet = pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount;

	pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount += 1;

	TRM_Backend_updateDescriptorSet(
		pPipeline->info.graphics.descriptorInfoCount,
		pPipeline->info.graphics.descriptorInfos,
		bindings,
		*pDescriptorSet,
		swapchainImageIndex);

	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createImageToImageCopyPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_IMAGE_TO_IMAGE_COPY;
	backendPass.info.imageToImageCopy.width = pass.info.imageToImageCopy.width;
	backendPass.info.imageToImageCopy.height = pass.info.imageToImageCopy.height;

	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	// src image
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.imageToImageCopy.srcImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.imageToImageCopy.srcImage = expectedResourceState.resource;
	}

	// dst image
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.imageToImageCopy.dstImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.imageToImageCopy.dstImage = expectedResourceState.resource;
	}
	
	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createBufferToImageCopyPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_BUFFER_TO_IMAGE_COPY;
	backendPass.info.bufferToImageCopy.width = pass.info.bufferToImageCopy.width;
	backendPass.info.bufferToImageCopy.height = pass.info.bufferToImageCopy.height;

	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	// src buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.bufferToImageCopy.srcBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.bufferToImageCopy.srcBuffer = expectedResourceState.resource;
	}

	// dst image
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.bufferToImageCopy.dstImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.bufferToImageCopy.dstImage = expectedResourceState.resource;
	}
	
	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createBufferToBufferCopyPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_BUFFER_TO_BUFFER_COPY;
	backendPass.info.bufferToBufferCopy.sizeInBytes = pass.info.bufferToBufferCopy.sizeInBytes;
	
	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	// src buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.bufferToBufferCopy.srcBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.bufferToBufferCopy.srcBuffer = expectedResourceState.resource;
	}

	// dst buffer
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.bufferToBufferCopy.dstBuffer, swapchainImageIndex);
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.bufferToBufferCopy.dstBuffer = expectedResourceState.resource;
	}
	
	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createBlitPass(struct TRM_Pass pass, uint32_t swapchainImageIndex)
{		
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_BLIT;
	backendPass.info.blit.srcWidth = pass.info.blit.srcWidth;
	backendPass.info.blit.srcHeight = pass.info.blit.srcHeight;
	backendPass.info.blit.dstWidth = pass.info.blit.dstWidth;
	backendPass.info.blit.dstHeight = pass.info.blit.dstHeight;
	
	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	// src image
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.blit.srcImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_READ_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.blit.srcImage = expectedResourceState.resource;
	}

	// dst image
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pass.info.blit.dstImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		expectedResourceState.state.access = VK_ACCESS_TRANSFER_WRITE_BIT;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
		backendPass.info.blit.dstImage = expectedResourceState.resource;
	}
	
	return backendPass;
}

static struct TRM_Backend_Pass TRM_Backend_createPresentPass(uint32_t swapchainImageIndex)
{			
	struct TRM_Backend_Pass backendPass = {0};
	backendPass.type = TRM_PASS_TYPE_PRESENT;
	
	TRM_DynamicArray_create(sizeof(struct TRM_Backend_ExpectedResourceState), &backendPass.expectedResourceStates);

	// present image
	{
		struct TRM_Backend_ExpectedResourceState expectedResourceState = {0};
		expectedResourceState.resource = TRM_Backend_translateResource(pState->pSwapchainImageInfos[swapchainImageIndex].colorImage, swapchainImageIndex);
		expectedResourceState.state.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		expectedResourceState.state.access = 0;
		expectedResourceState.state.stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		TRM_Backend_addExpectedResourceState(expectedResourceState, &backendPass.expectedResourceStates);
	}
	
	return backendPass;
}

static void TRM_Backend_createPasses(uint32_t passCount, const struct TRM_Pass* pPasses, struct TRM_Backend_Pass* pBackendPasses, uint32_t swapchainImageIndex)
{
	for(uint32_t i = 0; i < passCount; ++i)
	{
		const struct TRM_Pass pass = pPasses[i];
		switch(pass.type)
		{
			case TRM_PASS_TYPE_DISPATCH : 				pBackendPasses[i] = TRM_Backend_createDispatchPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_DRAW_VERTEX : 			pBackendPasses[i] = TRM_Backend_createDrawVertexPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_DRAW_INDEXED : 			pBackendPasses[i] = TRM_Backend_createDrawIndexedPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_DRAW_VERTEX_INDIRECT : 	pBackendPasses[i] = TRM_Backend_createDrawVertexIndirectPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_DRAW_INDEXED_INDIRECT : 	pBackendPasses[i] = TRM_Backend_createDrawIndexedIndirectPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_IMAGE_TO_IMAGE_COPY : 	pBackendPasses[i] = TRM_Backend_createImageToImageCopyPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_BUFFER_TO_IMAGE_COPY : 	pBackendPasses[i] = TRM_Backend_createBufferToImageCopyPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_BUFFER_TO_BUFFER_COPY : 	pBackendPasses[i] = TRM_Backend_createBufferToBufferCopyPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_BLIT : 					pBackendPasses[i] = TRM_Backend_createBlitPass(pass, swapchainImageIndex); break;
			case TRM_PASS_TYPE_PRESENT : 				pBackendPasses[i] = TRM_Backend_createPresentPass(swapchainImageIndex); break;
			default : exit(EXIT_FAILURE);
		}
	}
}

static void TRM_Backend_fillCommandBuffer(
	uint32_t passCount,
	struct TRM_Backend_Pass* pPasses,
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
		const struct TRM_Backend_Pass* pPass = &pPasses[passIndex];

		VkPipelineStageFlags srcStageFlags = 0;
		VkPipelineStageFlags dstStageFlags = 0;
		struct TRM_DynamicArray bufferMemoryBarriers;
		struct TRM_DynamicArray imageMemoryBarriers;

		TRM_DynamicArray_create(sizeof(VkBufferMemoryBarrier), &bufferMemoryBarriers);
		TRM_DynamicArray_create(sizeof(VkImageMemoryBarrier), &imageMemoryBarriers);

		// retrieves all barriers (memory + execution) for this pass for each resource
		bool needBarrier = false;
		for(uint32_t i = 0; i < pPass->expectedResourceStates.elementCount; ++i)
		{
			struct TRM_Backend_ExpectedResourceState expectedResourceState;
			TRM_DynamicArray_at(i, &pPass->expectedResourceStates, &expectedResourceState);

			// update resource use
			struct TRM_Backend_Resource* pResource = NULL;
			TRM_Arena_get(expectedResourceState.resource, pState->resourcePool, (void**)&pResource);
			pResource->lastFrameIndexNoWarp = pState->frameIndexWarp;

			struct TRM_Backend_ResourceState* pPreviousResourceState = &pResource->state;
			struct TRM_Backend_ResourceState* pCurrentResourceState = &expectedResourceState.state;

			const uint32_t lastWasWrite = (pPreviousResourceState->access & (
				VK_ACCESS_SHADER_WRITE_BIT |
				VK_ACCESS_TRANSFER_WRITE_BIT |
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));

			const uint32_t currentIsWrite = (pCurrentResourceState->access & (
				VK_ACCESS_SHADER_WRITE_BIT |
				VK_ACCESS_TRANSFER_WRITE_BIT |
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));

			const bool readAfterWrite = (lastWasWrite && !currentIsWrite);
			const bool writeAfterRead = (!lastWasWrite && currentIsWrite);
			const bool writeAfterWrite = (lastWasWrite && currentIsWrite);
			const bool layoutChanged = (pPreviousResourceState->layout != pCurrentResourceState->layout);

			if(layoutChanged || readAfterWrite || writeAfterRead || writeAfterWrite)
			{
				needBarrier = true;

				srcStageFlags |= pPreviousResourceState->stage;
				dstStageFlags |= pCurrentResourceState->stage;

				if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
				{
					VkImageMemoryBarrier imageMemoryBarrier = {0};
					imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					imageMemoryBarrier.pNext = NULL;

					if(readAfterWrite || writeAfterWrite)
					{
						imageMemoryBarrier.srcAccessMask = pPreviousResourceState->access;
						imageMemoryBarrier.dstAccessMask = pCurrentResourceState->access;
					}
					else
					{
						imageMemoryBarrier.srcAccessMask = VK_ACCESS_NONE;
						imageMemoryBarrier.dstAccessMask = VK_ACCESS_NONE;
					}

					imageMemoryBarrier.oldLayout = pPreviousResourceState->layout;
					imageMemoryBarrier.newLayout = pCurrentResourceState->layout;
					imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					imageMemoryBarrier.image = pResource->info.image.image;
					imageMemoryBarrier.subresourceRange.aspectMask = pResource->info.image.aspect;
					imageMemoryBarrier.subresourceRange.layerCount = 1;
					imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
					imageMemoryBarrier.subresourceRange.levelCount = pResource->info.image.mipCount; // all mips are transitionned
					imageMemoryBarrier.subresourceRange.baseMipLevel = 0;

					TRM_DynamicArray_push(&imageMemoryBarrier, &imageMemoryBarriers);
				}
				else if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_BUFFER)
				{
					VkBufferMemoryBarrier bufferMemoryBarrier = {0};
					bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
					bufferMemoryBarrier.pNext = NULL;

					if(readAfterWrite || writeAfterWrite)
					{
						bufferMemoryBarrier.srcAccessMask = pPreviousResourceState->access;
						bufferMemoryBarrier.dstAccessMask = pCurrentResourceState->access;
					}
					else
					{
						bufferMemoryBarrier.srcAccessMask = VK_ACCESS_NONE;
						bufferMemoryBarrier.dstAccessMask = VK_ACCESS_NONE;
					}

					bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					bufferMemoryBarrier.buffer = pResource->info.buffer.buffer;
					bufferMemoryBarrier.offset = 0;
					bufferMemoryBarrier.size = VK_WHOLE_SIZE;

					TRM_DynamicArray_push(&bufferMemoryBarrier, &bufferMemoryBarriers);
				}

				*pPreviousResourceState = *pCurrentResourceState;
			}
		}

		// emit commands
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
		}

		TRM_DynamicArray_destroy(&bufferMemoryBarriers);
		TRM_DynamicArray_destroy(&imageMemoryBarriers);

		switch(pPass->type)
		{
		case TRM_PASS_TYPE_DISPATCH:
		{
			struct TRM_Backend_Pipeline* pPipeline = NULL;
			TRM_Arena_get(pPass->info.dispatch.pipeline, pState->pipelinePool, (void**)&pPipeline);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_COMPUTE,
				pPipeline->info.compute.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pPass->info.dispatch.descriptorSet],
				0,
				NULL);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pPipeline->info.compute.pipeline);

			vkCmdDispatch(
				commandBuffer,
				pPass->info.dispatch.groupCountX,
				pPass->info.dispatch.groupCountY,
				pPass->info.dispatch.groupCountZ);
			break;
		}
		case TRM_PASS_TYPE_DRAW_VERTEX:
		{
			struct TRM_Backend_Pipeline* pPipeline = NULL;
			TRM_Arena_get(pPass->info.drawVertex.pipeline, pState->pipelinePool, (void**)&pPipeline);

			VkRenderPassBeginInfo renderPassBeginInfo = {0};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.pNext = NULL;
			renderPassBeginInfo.renderPass = pPipeline->info.graphics.renderPass;
			renderPassBeginInfo.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pPass->info.drawVertex.framebuffer];
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = pPass->info.drawVertex.width;
			renderPassBeginInfo.renderArea.extent.height = pPass->info.drawVertex.height;
			renderPassBeginInfo.clearValueCount = pPass->info.drawVertex.clearColorCount;
			renderPassBeginInfo.pClearValues = pPass->info.drawVertex.clearColors;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPipeline->info.graphics.pipeline);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pPipeline->info.graphics.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pPass->info.drawVertex.descriptorSet],
				0,
				NULL);

			VkViewport viewport = {0};
			viewport.x = 0;
			viewport.y = (float)pPass->info.drawVertex.height;
			viewport.width = (float)pPass->info.drawVertex.width;
			viewport.height = -(float)pPass->info.drawVertex.height;
			viewport.minDepth = 0;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {0};
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			scissor.extent.width = pPass->info.drawVertex.width;
			scissor.extent.height = pPass->info.drawVertex.height;

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			if(pPass->info.drawVertex.useVertexBuffer)
			{
				struct TRM_Backend_Resource* pVertexBuffer = NULL;
				TRM_Arena_get(pPass->info.drawVertex.vertexBuffer, pState->resourcePool, (void**)&pVertexBuffer);
				VkDeviceSize offset = 0;
				VkBuffer vertexBuffer = pVertexBuffer->info.buffer.buffer;
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
			}
			vkCmdDraw(commandBuffer, pPass->info.drawVertex.vertexCount, 1, 0, 0);

			vkCmdEndRenderPass(commandBuffer);
			break;
		}
		case TRM_PASS_TYPE_DRAW_INDEXED:
		{
			struct TRM_Backend_Pipeline* pPipeline = NULL;
			TRM_Arena_get(pPass->info.drawIndexed.pipeline, pState->pipelinePool, (void**)&pPipeline);

			VkRenderPassBeginInfo renderPassBeginInfo = {0};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.pNext = NULL;
			renderPassBeginInfo.renderPass = pPipeline->info.graphics.renderPass;
			renderPassBeginInfo.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pPass->info.drawIndexed.framebuffer];
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = pPass->info.drawIndexed.width;
			renderPassBeginInfo.renderArea.extent.height = pPass->info.drawIndexed.height;
			renderPassBeginInfo.clearValueCount = pPass->info.drawIndexed.clearColorCount;
			renderPassBeginInfo.pClearValues = pPass->info.drawIndexed.clearColors;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPipeline->info.graphics.pipeline);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pPipeline->info.graphics.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pPass->info.drawIndexed.descriptorSet],
				0,
				NULL);

			VkViewport viewport = {0};
			viewport.x = 0;
			viewport.y = (float)pPass->info.drawIndexed.height;
			viewport.width = (float)pPass->info.drawIndexed.width;
			viewport.height = -(float)pPass->info.drawIndexed.height;
			viewport.minDepth = 0;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {0};
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			scissor.extent.width = pPass->info.drawIndexed.width;
			scissor.extent.height = pPass->info.drawIndexed.height;

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			struct TRM_Backend_Resource* pVertexBuffer = NULL;
			TRM_Arena_get(pPass->info.drawIndexed.vertexBuffer, pState->resourcePool, (void**)&pVertexBuffer);

			struct TRM_Backend_Resource* pIndexBuffer = NULL;
			TRM_Arena_get(pPass->info.drawIndexed.indexBuffer, pState->resourcePool, (void**)&pIndexBuffer);

			VkDeviceSize offset = 0;
			VkBuffer vertexBuffer = pVertexBuffer->info.buffer.buffer;
			VkBuffer indexBuffer = pIndexBuffer->info.buffer.buffer;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
			vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(commandBuffer, pPass->info.drawIndexed.indexCount, 1, 0, 0, 0);

			vkCmdEndRenderPass(commandBuffer);
			break;
		}
		case TRM_PASS_TYPE_DRAW_VERTEX_INDIRECT:
		{
			struct TRM_Backend_Pipeline* pPipeline = NULL;
			TRM_Arena_get(pPass->info.drawVertexIndirect.pipeline, pState->pipelinePool, (void**)&pPipeline);

			VkRenderPassBeginInfo renderPassBeginInfo = {0};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.pNext = NULL;
			renderPassBeginInfo.renderPass = pPipeline->info.graphics.renderPass;
			renderPassBeginInfo.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pPass->info.drawVertexIndirect.framebuffer];
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = pPass->info.drawVertexIndirect.width;
			renderPassBeginInfo.renderArea.extent.height = pPass->info.drawVertexIndirect.height;
			renderPassBeginInfo.clearValueCount = pPass->info.drawVertexIndirect.clearColorCount;
			renderPassBeginInfo.pClearValues = pPass->info.drawVertexIndirect.clearColors;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPipeline->info.graphics.pipeline);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pPipeline->info.graphics.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pPass->info.drawVertexIndirect.descriptorSet],
				0,
				NULL);

			VkViewport viewport = {0};
			viewport.x = 0;
			viewport.y = (float)pPass->info.drawVertexIndirect.height;
			viewport.width = (float)pPass->info.drawVertexIndirect.width;
			viewport.height = -(float)pPass->info.drawVertexIndirect.height;
			viewport.minDepth = 0;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {0};
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			scissor.extent.width = pPass->info.drawVertexIndirect.width;
			scissor.extent.height = pPass->info.drawVertexIndirect.height;

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			struct TRM_Backend_Resource* pBuffer = NULL;
			TRM_Arena_get(pPass->info.drawVertexIndirect.indirectCommandBuffer, pState->resourcePool, (void**)&pBuffer);

			if(pPass->info.drawVertexIndirect.useVertexBuffer)
			{
				struct TRM_Backend_Resource* pVertexBuffer = NULL;
				TRM_Arena_get(pPass->info.drawVertexIndirect.vertexBuffer, pState->resourcePool, (void**)&pVertexBuffer);
				VkDeviceSize offset = 0;
				VkBuffer vertexBuffer = pVertexBuffer->info.buffer.buffer;
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
			}

			vkCmdDrawIndirect(commandBuffer, pBuffer->info.buffer.buffer, 0, pPass->info.drawVertexIndirect.drawCount, sizeof(VkDrawIndirectCommand));

			vkCmdEndRenderPass(commandBuffer);
			break;
		}
		case TRM_PASS_TYPE_DRAW_INDEXED_INDIRECT:
		{
			struct TRM_Backend_Pipeline* pPipeline = NULL;
			TRM_Arena_get(pPass->info.drawIndexedIndirect.pipeline, pState->pipelinePool, (void**)&pPipeline);

			VkRenderPassBeginInfo renderPassBeginInfo = {0};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.pNext = NULL;
			renderPassBeginInfo.renderPass = pPipeline->info.graphics.renderPass;
			renderPassBeginInfo.framebuffer = pState->pFrameInfos[pState->frameIndexWarp].framebuffers[pPass->info.drawIndexed.framebuffer];
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = pPass->info.drawIndexed.width;
			renderPassBeginInfo.renderArea.extent.height = pPass->info.drawIndexed.height;
			renderPassBeginInfo.clearValueCount = pPass->info.drawIndexed.clearColorCount;
			renderPassBeginInfo.pClearValues = pPass->info.drawIndexed.clearColors;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPipeline->info.graphics.pipeline);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pPipeline->info.graphics.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndexWarp].descriptorSets[pPass->info.drawIndexed.descriptorSet],
				0,
				NULL);

			VkViewport viewport = {0};
			viewport.x = 0;
			viewport.y = (float)pPass->info.drawIndexed.height;
			viewport.width = (float)pPass->info.drawIndexed.width;
			viewport.height = -(float)pPass->info.drawIndexed.height;
			viewport.minDepth = 0;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {0};
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			scissor.extent.width = pPass->info.drawIndexed.width;
			scissor.extent.height = pPass->info.drawIndexed.height;

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			struct TRM_Backend_Resource* pBuffer = NULL;
			TRM_Arena_get(pPass->info.drawIndexedIndirect.indirectCommandBuffer, pState->resourcePool, (void**)&pBuffer);

			struct TRM_Backend_Resource* pVertexBuffer = NULL;
			TRM_Arena_get(pPass->info.drawIndexedIndirect.vertexBuffer, pState->resourcePool, (void**)&pVertexBuffer);

			struct TRM_Backend_Resource* pIndexBuffer = NULL;
			TRM_Arena_get(pPass->info.drawIndexedIndirect.indexBuffer, pState->resourcePool, (void**)&pIndexBuffer);

			VkDeviceSize offset = 0;
			VkBuffer vertexBuffer = pVertexBuffer->info.buffer.buffer;
			VkBuffer indexBuffer = pIndexBuffer->info.buffer.buffer;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
			vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdDrawIndexedIndirect(commandBuffer, pBuffer->info.buffer.buffer, 0, pPass->info.drawIndexedIndirect.drawCount, sizeof(VkDrawIndexedIndirectCommand));

			vkCmdEndRenderPass(commandBuffer);
			break;
		}
		case TRM_PASS_TYPE_IMAGE_TO_IMAGE_COPY:
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPass->info.imageToImageCopy.srcImage, pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPass->info.imageToImageCopy.dstImage, pState->resourcePool, (void**)&pOutputResource);

			VkImageCopy imageCopy = {0};
			imageCopy.srcSubresource.aspectMask = pInputResource->info.image.aspect;
			imageCopy.srcSubresource.baseArrayLayer = 0;
			imageCopy.srcSubresource.layerCount = 1;
			imageCopy.srcSubresource.mipLevel = 0;
			imageCopy.srcOffset.x = 0;
			imageCopy.srcOffset.y = 0;
			imageCopy.srcOffset.z = 0;
			imageCopy.dstSubresource.aspectMask = pOutputResource->info.image.aspect;
			imageCopy.dstSubresource.baseArrayLayer = 0;
			imageCopy.dstSubresource.layerCount = 1;
			imageCopy.dstSubresource.mipLevel = 0;
			imageCopy.dstOffset.x = 0;
			imageCopy.dstOffset.y = 0;
			imageCopy.dstOffset.z = 0;
			imageCopy.extent.width = pPass->info.imageToImageCopy.width;
			imageCopy.extent.height = pPass->info.imageToImageCopy.height;
			imageCopy.extent.depth = 1;

			vkCmdCopyImage(
				commandBuffer,
				pInputResource->info.image.image,
				pInputResource->state.layout,
				pOutputResource->info.image.image,
				pOutputResource->state.layout,
				1,
				&imageCopy);
			break;
		}
		case TRM_PASS_TYPE_BUFFER_TO_IMAGE_COPY:
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPass->info.bufferToImageCopy.srcBuffer, pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPass->info.bufferToImageCopy.dstImage, pState->resourcePool, (void**)&pOutputResource);

			VkBufferImageCopy bufferImageCopy = {0};
			bufferImageCopy.bufferOffset = 0;
			bufferImageCopy.bufferRowLength = 0;
			bufferImageCopy.bufferImageHeight = 0;
			bufferImageCopy.imageSubresource.aspectMask = pOutputResource->info.image.aspect;
			bufferImageCopy.imageSubresource.baseArrayLayer = 0;
			bufferImageCopy.imageSubresource.layerCount = 1;
			bufferImageCopy.imageSubresource.mipLevel = 0;
			bufferImageCopy.imageOffset.x = 0;
			bufferImageCopy.imageOffset.y = 0;
			bufferImageCopy.imageOffset.z = 0;
			bufferImageCopy.imageExtent.width = pPass->info.bufferToImageCopy.width;
			bufferImageCopy.imageExtent.height = pPass->info.bufferToImageCopy.height;
			bufferImageCopy.imageExtent.depth = 1;

			vkCmdCopyBufferToImage(
				commandBuffer,
				pInputResource->info.buffer.buffer,
				pOutputResource->info.image.image,
				pOutputResource->state.layout,
				1,
				&bufferImageCopy);
			break;
		}
		case TRM_PASS_TYPE_BUFFER_TO_BUFFER_COPY:
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPass->info.bufferToBufferCopy.srcBuffer, pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPass->info.bufferToBufferCopy.dstBuffer, pState->resourcePool, (void**)&pOutputResource);

			VkBufferCopy bufferCopy = {0};
			bufferCopy.srcOffset = 0;
			bufferCopy.dstOffset = 0;
			bufferCopy.size = pPass->info.bufferToBufferCopy.sizeInBytes;

			vkCmdCopyBuffer(
				commandBuffer,
				pInputResource->info.buffer.buffer,
				pOutputResource->info.buffer.buffer,
				1,
				&bufferCopy);
			break;
		}
		case TRM_PASS_TYPE_BLIT:
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPass->info.blit.srcImage, pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPass->info.blit.dstImage, pState->resourcePool, (void**)&pOutputResource);

			VkImageBlit blit = {0};
			blit.srcSubresource.aspectMask = pInputResource->info.image.aspect;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.srcSubresource.mipLevel = 0;
			blit.srcOffsets[0].x = 0;
			blit.srcOffsets[0].y = 0;
			blit.srcOffsets[0].z = 0;
			blit.srcOffsets[1].x = (int32_t)pPasses[passIndex].info.blit.srcWidth;
			blit.srcOffsets[1].y = (int32_t)pPasses[passIndex].info.blit.srcHeight;
			blit.srcOffsets[1].z = 1;
			blit.dstSubresource.aspectMask = pOutputResource->info.image.aspect;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;
			blit.dstSubresource.mipLevel = 0;
			blit.dstOffsets[0].x = 0;
			blit.dstOffsets[0].y = 0;
			blit.dstOffsets[0].z = 0;
			blit.dstOffsets[1].x = (int32_t)pPasses[passIndex].info.blit.dstWidth;
			blit.dstOffsets[1].y = (int32_t)pPasses[passIndex].info.blit.dstHeight;
			blit.dstOffsets[1].z = 1;

			vkCmdBlitImage(
				commandBuffer,
				pInputResource->info.image.image,
				pInputResource->state.layout,
				pOutputResource->info.image.image,
				pOutputResource->state.layout,
				1,
				&blit,
				VK_FILTER_LINEAR);
			break;
		}
		case TRM_PASS_TYPE_PRESENT: break;
		default: exit(EXIT_FAILURE);
		}
	}

	if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

void TRM_start(struct TRM_NativeWindow nativeWindow)
{
	if(pState != NULL)
	{
		exit(EXIT_FAILURE);
	}

	TRM_Memory_start();

	TRM_Memory_allocate(sizeof(struct TRM_Backend_State), (void**)&pState);
	TRM_Memory_memzero(sizeof(struct TRM_Backend_State), pState);

	if(volkInitialize() != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	pState->pAllocator = NULL;

	TRM_Backend_createInstance(pState->pAllocator, &pState->instance);
	volkLoadInstance(pState->instance);
	TRM_Backend_pickPhysicalDevice(pState->instance, &pState->physicalDevice);
	TRM_Backend_createSurface(pState->pAllocator, pState->instance, nativeWindow, &pState->surface);
	TRM_Backend_findQueueFamilyIndex(pState->physicalDevice, pState->surface, &pState->queueFamilyIndex);
	TRM_Backend_createDevice(pState->pAllocator, pState->physicalDevice, pState->queueFamilyIndex, &pState->device);
	vkGetDeviceQueue(pState->device, pState->queueFamilyIndex, 0, &pState->queue);
	TRM_Backend_createCommandPool(pState->pAllocator, pState->device, pState->queueFamilyIndex, &pState->commandPool);
	TRM_Backend_createDescriptorPool(pState->pAllocator, pState->device, &pState->descriptorPool);

	pState->swapchain = VK_NULL_HANDLE;

	TRM_Arena_create(sizeof(struct TRM_Backend_Resource), TRM_BACKEND_MAX_RESOURCE_COUNT, &pState->resourcePool);
	TRM_LinkedList_create(sizeof(uint32_t), &pState->resourceHandles);
	TRM_Arena_create(sizeof(struct TRM_Backend_Pass), TRM_BACKEND_MAX_PIPELINE_COUNT, &pState->pipelinePool);

	TRM_Memory_allocate(sizeof(struct TRM_Backend_FrameInfo) * TRM_BACKEND_FRAME_COUNT, (void**)&pState->pFrameInfos);
	TRM_Backend_createSampler(pState->pAllocator, pState->device, &pState->globalSampler);

	for(uint32_t frameIndexWarp = 0; frameIndexWarp < TRM_BACKEND_FRAME_COUNT; ++frameIndexWarp)
	{
		TRM_Backend_allocateCommandBuffer(pState->commandPool, pState->device, &pState->pFrameInfos[frameIndexWarp].commandBuffer);
		TRM_Backend_createFence(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndexWarp].commandBufferExecutedFence);
		TRM_Backend_createSemaphore(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndexWarp].imageAvailableSemaphore);
		TRM_Backend_createTimelineSemaphore(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndexWarp].timelineSemaphore);
		pState->pFrameInfos[frameIndexWarp].descriptorSetCount = 0;
		pState->pFrameInfos[frameIndexWarp].framebufferCount = 0;
	}

	pState->frameIndexWarp = 1; // we start at 1 to avoid issues with the timeline semaphores default value (0)
	pState->frameIndexNoWarp = 1;
}

void TRM_terminate(void)
{
	if(pState != NULL)
	{
		if(vkDeviceWaitIdle(pState->device) != VK_SUCCESS)
		{
			exit(EXIT_FAILURE);
		}

		struct TRM_LinkedList_Node* pResourceNode = pState->resourceHandles.pFirstNode;
		while(pResourceNode != NULL)
		{
			const uint32_t handle = *(uint32_t*)pResourceNode->pData;
			struct TRM_Backend_Resource* pResource = NULL;
			TRM_Arena_get(handle, pState->resourcePool, (void**)&pResource);
			TRM_Backend_destroyResource(handle);
			pResourceNode = pResourceNode->pNextNode;
		}
		TRM_LinkedList_destroy(&pState->resourceHandles);

		for(uint32_t frameIndexWarp = 0; frameIndexWarp < TRM_BACKEND_FRAME_COUNT; ++frameIndexWarp)
		{
			vkDestroySemaphore(pState->device, pState->pFrameInfos[frameIndexWarp].timelineSemaphore, pState->pAllocator);
			vkDestroySemaphore(pState->device, pState->pFrameInfos[frameIndexWarp].imageAvailableSemaphore, pState->pAllocator);
			vkDestroyFence(pState->device, pState->pFrameInfos[frameIndexWarp].commandBufferExecutedFence, pState->pAllocator);

			for(uint32_t framebufferIndex = 0; framebufferIndex < pState->pFrameInfos[frameIndexWarp].framebufferCount; ++framebufferIndex)
			{
				vkDestroyFramebuffer(pState->device, pState->pFrameInfos[frameIndexWarp].framebuffers[framebufferIndex], pState->pAllocator);
			}
		}
		TRM_Memory_deallocate(pState->pFrameInfos);

		if(pState->swapchain != VK_NULL_HANDLE)
		{
			for(uint32_t swapchainImageIndex = 0; swapchainImageIndex < pState->swapchainImageCount; ++swapchainImageIndex)
			{
				struct TRM_Backend_Resource* pSwapchainColorImage;
				TRM_Arena_get(pState->pSwapchainImageInfos[swapchainImageIndex].colorImage, pState->resourcePool, (void**)&pSwapchainColorImage);

				vkDestroyImageView(pState->device, pSwapchainColorImage->info.image.allMipsImageView, pState->pAllocator);
				vkFreeMemory(pState->device, pSwapchainColorImage->info.image.memory, pState->pAllocator);
				vkDestroySemaphore(pState->device, pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore, pState->pAllocator);
			}
			TRM_Memory_deallocate(pState->pSwapchainImageInfos);
		}
		
		vkDestroySampler(pState->device, pState->globalSampler, pState->pAllocator);

		TRM_Arena_destroy(&pState->resourcePool);
		TRM_Arena_destroy(&pState->pipelinePool);
		
		vkDestroySwapchainKHR(pState->device, pState->swapchain, pState->pAllocator);
		vkDestroyDescriptorPool(pState->device, pState->descriptorPool, pState->pAllocator);
		vkDestroyCommandPool(pState->device, pState->commandPool, pState->pAllocator);
		vkDestroyDevice(pState->device, pState->pAllocator);
		vkDestroySurfaceKHR(pState->instance, pState->surface, pState->pAllocator);
		vkDestroyInstance(pState->instance, pState->pAllocator);

		TRM_Memory_deallocate(pState);
		pState = NULL;

		TRM_Memory_terminate();
	}
}

void TRM_beginFrame(void)
{
	if(vkWaitForFences(pState->device, 1, &pState->pFrameInfos[pState->frameIndexWarp].commandBufferExecutedFence, VK_FALSE, UINT64_MAX) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	for(uint32_t framebufferIndex = 0; framebufferIndex < pState->pFrameInfos[pState->frameIndexWarp].framebufferCount; ++framebufferIndex)
	{
		vkDestroyFramebuffer(pState->device, pState->pFrameInfos[pState->frameIndexWarp].framebuffers[framebufferIndex], pState->pAllocator);
	}
	pState->pFrameInfos[pState->frameIndexWarp].framebufferCount = 0;
	
	if(pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount > 0)
	{
		vkFreeDescriptorSets(
			pState->device, 
			pState->descriptorPool, 
			pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount,
			pState->pFrameInfos[pState->frameIndexWarp].descriptorSets);
		pState->pFrameInfos[pState->frameIndexWarp].descriptorSetCount = 0;
	}

	uint64_t completedFrameIndexNoWarp = 0;
	vkGetSemaphoreCounterValue(pState->device, pState->pFrameInfos[pState->frameIndexWarp].timelineSemaphore, &completedFrameIndexNoWarp);

	struct TRM_LinkedList_Node* pResourceNode = pState->resourceHandles.pFirstNode;
	while(pResourceNode != NULL)
	{
		const uint32_t handle = *(uint32_t*)pResourceNode->pData;
		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(handle, pState->resourcePool, (void**)&pResource);
		if(pResource->toDelete && pResource->lastFrameIndexNoWarp <= completedFrameIndexNoWarp)
		{
			TRM_Backend_destroyResource(handle);
			struct TRM_LinkedList_Node* pNextNode = pResourceNode->pNextNode;
			TRM_LinkedList_delete(pResourceNode, &pState->resourceHandles);
			pResourceNode = pNextNode;
		}
		else
		{
			pResourceNode = pResourceNode->pNextNode;
		}
	}
}

void TRM_endFrame(uint32_t passCount, struct TRM_Pass* pPasss, uint32_t windowWidth, uint32_t windowHeight)
{
	if(windowWidth != pState->swapchainWidth || windowHeight != pState->swapchainHeight)
	{
		TRM_Backend_recreateSwapchain(windowWidth, windowHeight);
	}

	uint32_t swapchainImageIndex = 0;
	VkResult acquireNextImageResult = vkAcquireNextImageKHR(
		pState->device,
		pState->swapchain,
		UINT64_MAX,
		pState->pFrameInfos[pState->frameIndexWarp].imageAvailableSemaphore,
		VK_NULL_HANDLE,
		&swapchainImageIndex);
		
	if(acquireNextImageResult != VK_SUCCESS && 
		acquireNextImageResult != VK_ERROR_OUT_OF_DATE_KHR && 
		acquireNextImageResult != VK_SUBOPTIMAL_KHR)
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_Backend_Pass* pBackendPasses = NULL;
	TRM_Memory_allocate(sizeof(struct TRM_Backend_Pass) * passCount, (void**)&pBackendPasses);
	TRM_Memory_memzero(sizeof(struct TRM_Backend_Pass) * passCount, pBackendPasses);
	
	TRM_Backend_createPasses(passCount, pPasss, pBackendPasses, swapchainImageIndex);

	TRM_Backend_fillCommandBuffer(passCount, pBackendPasses, pState->pFrameInfos[pState->frameIndexWarp].commandBuffer);

	for(uint32_t passIndex = 0; passIndex < passCount; ++passIndex)
	{
		TRM_DynamicArray_destroy(&pBackendPasses[passIndex].expectedResourceStates);
	}
	TRM_Memory_deallocate(pBackendPasses);

	VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

	VkSemaphore signalSemaphores[2] = {
		pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore,
		pState->pFrameInfos[pState->frameIndexWarp].timelineSemaphore
	};

	uint64_t signalValues[2] =
	{
		0, // ignored for binary semaphore
		pState->frameIndexNoWarp + 1
	};

	VkTimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = {0};
	timelineSemaphoreSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timelineSemaphoreSubmitInfo.pNext = NULL;
	timelineSemaphoreSubmitInfo.waitSemaphoreValueCount = 0;
	timelineSemaphoreSubmitInfo.pWaitSemaphoreValues = NULL;
	timelineSemaphoreSubmitInfo.signalSemaphoreValueCount = 2;
	timelineSemaphoreSubmitInfo.pSignalSemaphoreValues = signalValues;

	VkSubmitInfo submitInfo = {0};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = &timelineSemaphoreSubmitInfo;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &pState->pFrameInfos[pState->frameIndexWarp].imageAvailableSemaphore;
	submitInfo.pWaitDstStageMask = &waitDstStageMask;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &pState->pFrameInfos[pState->frameIndexWarp].commandBuffer;
	submitInfo.signalSemaphoreCount = 2;
	submitInfo.pSignalSemaphores = signalSemaphores;

	if(vkResetFences(pState->device, 1, &pState->pFrameInfos[pState->frameIndexWarp].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	if(vkQueueSubmit(pState->queue, 1, &submitInfo, pState->pFrameInfos[pState->frameIndexWarp].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	VkPresentInfoKHR presentInfo = {0};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = NULL;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &pState->swapchain;
	presentInfo.pImageIndices = &swapchainImageIndex;
	presentInfo.pResults = NULL;
	
	VkResult presentResult = vkQueuePresentKHR(pState->queue, &presentInfo);
	if(presentResult != VK_SUCCESS && presentResult != VK_ERROR_OUT_OF_DATE_KHR && presentResult != VK_SUBOPTIMAL_KHR)
	{
		exit(EXIT_FAILURE);
	}

	pState->frameIndexWarp = (pState->frameIndexWarp + 1) % TRM_BACKEND_FRAME_COUNT;
	pState->frameIndexNoWarp += 1;
}

void TRM_createResource(struct TRM_ResourceCreateInfo info, uint32_t* pHandle)
{
	TRM_Backend_createResource(info, pHandle);
	TRM_LinkedList_push(pHandle, &pState->resourceHandles);
}

void TRM_destroyResource(uint32_t handle)
{
	struct TRM_Backend_Resource* pResource = NULL;
	TRM_Arena_get(handle, pState->resourcePool, (void**)&pResource);
	pResource->toDelete = true; // resource destruction is differed
}

void TRM_writeBuffer(uint32_t sizeInBytes, const void* pData, uint32_t handle)
{
	struct TRM_Backend_Resource* pBufferIndirection = NULL;
	TRM_Arena_get(handle, pState->resourcePool, (void**)&pBufferIndirection);

	uint32_t buffer = pBufferIndirection->info.bufferIndirection.info.hostVisible.buffers[pState->frameIndexWarp];

	struct TRM_Backend_Resource* pResource = NULL;
	TRM_Arena_get(buffer, pState->resourcePool, (void**)&pResource);

	void* pMappedMemory = NULL;
	vkMapMemory(pState->device, pResource->info.buffer.memory, 0, sizeInBytes, 0, &pMappedMemory);
	TRM_Memory_memcpy(sizeInBytes, pData, pMappedMemory);
	vkUnmapMemory(pState->device, pResource->info.buffer.memory);
}

void TRM_createComputePipeline(struct TRM_ComputePipelineCreateInfo info, uint32_t* pHandle)
{
	struct TRM_Backend_Pipeline pipeline = {0};
	
	pipeline.type = TRM_PIPELINE_TYPE_COMPUTE;
	pipeline.info.compute.descriptorInfoCount = info.descriptorInfoCount;
	TRM_Memory_memcpy(
		sizeof(struct TRM_DescriptorInfo) * info.descriptorInfoCount, 
		info.descriptorInfos, 
		pipeline.info.compute.descriptorInfos);

	VkDescriptorSetLayoutBinding bindings[TRM_MAX_DESCRIPTOR_COUNT];
	
	for(uint32_t descriptorInfoIndex = 0; descriptorInfoIndex < info.descriptorInfoCount; ++descriptorInfoIndex)
	{
		VkDescriptorType descriptorType = TRM_Backend_convertDescriptorType(info.descriptorInfos[descriptorInfoIndex].descriptorType);

		bindings[descriptorInfoIndex].binding = descriptorInfoIndex;
		bindings[descriptorInfoIndex].descriptorCount = 1;
		bindings[descriptorInfoIndex].descriptorType = descriptorType;
		bindings[descriptorInfoIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		bindings[descriptorInfoIndex].pImmutableSamplers = NULL;
	}

	TRM_Backend_createDescriptorSetLayout(
		pState->pAllocator, 
		pState->device, 
		info.descriptorInfoCount, 
		bindings, 
		&pipeline.info.compute.descriptorSetLayout);

	TRM_Backend_createPipelineLayout(
		pState->pAllocator, 
		pState->device, 
		1, 
		&pipeline.info.compute.descriptorSetLayout, 
		&pipeline.info.compute.pipelineLayout);

	VkShaderModule shaderModule;
	TRM_Backend_createShaderModule(pState->pAllocator, pState->device, info.codeSize, info.pCode, &shaderModule);

	TRM_Backend_createComputePipeline(
		pState->pAllocator, 
		pState->device, 
		shaderModule, 
		pipeline.info.compute.pipelineLayout, 
		&pipeline.info.compute.pipeline);

	vkDestroyShaderModule(pState->device, shaderModule, pState->pAllocator);

	TRM_Arena_add(&pipeline, &pState->pipelinePool, pHandle);
}

void TRM_createGraphicsPipeline(struct TRM_GraphicsPipelineCreateInfo info, uint32_t* pHandle)
{
	struct TRM_Backend_Pipeline pipeline = {0};
	
	pipeline.type = TRM_PIPELINE_TYPE_GRAPHICS;
	pipeline.info.graphics.descriptorInfoCount = info.descriptorInfoCount;
	TRM_Memory_memcpy(
		sizeof(struct TRM_DescriptorInfo) * info.descriptorInfoCount, 
		info.descriptorInfos, 
		pipeline.info.graphics.descriptorInfos);

	VkDescriptorSetLayoutBinding bindings[TRM_MAX_DESCRIPTOR_COUNT];
	
	for(uint32_t descriptorInfoIndex = 0; descriptorInfoIndex < info.descriptorInfoCount; ++descriptorInfoIndex)
	{
		VkDescriptorType descriptorType = TRM_Backend_convertDescriptorType(info.descriptorInfos[descriptorInfoIndex].descriptorType);

		bindings[descriptorInfoIndex].binding = descriptorInfoIndex;
		bindings[descriptorInfoIndex].descriptorCount = 1;
		bindings[descriptorInfoIndex].descriptorType = descriptorType;
		bindings[descriptorInfoIndex].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings[descriptorInfoIndex].pImmutableSamplers = NULL;
	}

	TRM_Backend_createDescriptorSetLayout(
		pState->pAllocator, 
		pState->device, 
		info.descriptorInfoCount, 
		bindings, 
		&pipeline.info.graphics.descriptorSetLayout);

	TRM_Backend_createPipelineLayout(
		pState->pAllocator, 
		pState->device, 
		1, 
		&pipeline.info.graphics.descriptorSetLayout, 
		&pipeline.info.graphics.pipelineLayout);

	VkShaderModule vertexShaderModule;
	TRM_Backend_createShaderModule(pState->pAllocator, pState->device, info.vertexCodeSize, (uint32_t*)info.pVertexCode, &vertexShaderModule);

	VkShaderModule fragmentShaderModule;
	TRM_Backend_createShaderModule(pState->pAllocator, pState->device, info.fragmentCodeSize, (uint32_t*)info.pFragmentCode, &fragmentShaderModule);

	VkVertexInputAttributeDescription vertexAttributeDescriptions[TRM_MAX_VERTEX_ATTRIBUTE_COUNT];

	for(uint32_t i = 0; i < info.vertexAttributeDescriptionCount; ++i)
	{
		vertexAttributeDescriptions[i].binding = 0;
		vertexAttributeDescriptions[i].format = TRM_Backend_convertFormat(info.vertexAttributeDescriptions[i].format);
		vertexAttributeDescriptions[i].location = info.vertexAttributeDescriptions[i].shaderLocation;
		vertexAttributeDescriptions[i].offset = info.vertexAttributeDescriptions[i].offset;
	}

	VkVertexInputBindingDescription vertexBindingDescription = {0};
	vertexBindingDescription.binding = 0;
	vertexBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	vertexBindingDescription.stride = info.vertexStride;

	VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {0};
	vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputStateCreateInfo.pNext = NULL;
	vertexInputStateCreateInfo.flags = 0;
	vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
	vertexInputStateCreateInfo.pVertexBindingDescriptions = &vertexBindingDescription;
	vertexInputStateCreateInfo.vertexAttributeDescriptionCount = info.vertexAttributeDescriptionCount;
	vertexInputStateCreateInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions;

	VkAttachmentDescription attachmentDescriptions[TRM_MAX_COLOR_OUTPUT_COUNT + 1];

	for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < info.colorOutputImageCount; ++colorOutputImageIndex)
	{
		attachmentDescriptions[colorOutputImageIndex].flags = 0;
		attachmentDescriptions[colorOutputImageIndex].format = TRM_Backend_convertFormat(info.colorOutputImageInfos[colorOutputImageIndex].format);
		attachmentDescriptions[colorOutputImageIndex].samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescriptions[colorOutputImageIndex].loadOp = TRM_Backend_convertLoadOp(info.colorOutputImageInfos[colorOutputImageIndex].loadOp);
		attachmentDescriptions[colorOutputImageIndex].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescriptions[colorOutputImageIndex].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescriptions[colorOutputImageIndex].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescriptions[colorOutputImageIndex].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachmentDescriptions[colorOutputImageIndex].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}

	attachmentDescriptions[info.colorOutputImageCount].flags = 0;
	attachmentDescriptions[info.colorOutputImageCount].format = TRM_Backend_convertFormat(info.depthOutputInfo.format);
	attachmentDescriptions[info.colorOutputImageCount].samples = VK_SAMPLE_COUNT_1_BIT;
	attachmentDescriptions[info.colorOutputImageCount].loadOp = TRM_Backend_convertLoadOp(info.depthOutputInfo.loadOp);
	attachmentDescriptions[info.colorOutputImageCount].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachmentDescriptions[info.colorOutputImageCount].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachmentDescriptions[info.colorOutputImageCount].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachmentDescriptions[info.colorOutputImageCount].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachmentDescriptions[info.colorOutputImageCount].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachmentReferences[TRM_MAX_COLOR_OUTPUT_COUNT];

	for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < info.colorOutputImageCount; ++colorOutputImageIndex)
	{
		colorAttachmentReferences[colorOutputImageIndex].attachment = colorOutputImageIndex;
		colorAttachmentReferences[colorOutputImageIndex].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}	

	VkAttachmentReference depthAttachmentReference;
	depthAttachmentReference.attachment = info.colorOutputImageCount;
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {0};
	subpassDescription.flags = 0;
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = NULL;
	subpassDescription.colorAttachmentCount = info.colorOutputImageCount;
	subpassDescription.pColorAttachments = colorAttachmentReferences;
	subpassDescription.pResolveAttachments = NULL;
	subpassDescription.pDepthStencilAttachment = &depthAttachmentReference;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = NULL;

	TRM_Backend_createRenderPass(
		pState->pAllocator,
		pState->device,
		info.colorOutputImageCount + 1,
		attachmentDescriptions,
		subpassDescription,
		&pipeline.info.graphics.renderPass);

	TRM_Backend_createGraphicsPipeline(
		pState->pAllocator,
		pState->device,
		vertexShaderModule,
		fragmentShaderModule,
		vertexInputStateCreateInfo,
		info.colorOutputImageCount,
		pipeline.info.graphics.pipelineLayout,
		pipeline.info.graphics.renderPass,
		&pipeline.info.graphics.pipeline);

	vkDestroyShaderModule(pState->device, vertexShaderModule, pState->pAllocator);
	vkDestroyShaderModule(pState->device, fragmentShaderModule, pState->pAllocator);

	TRM_Arena_add(&pipeline, &pState->pipelinePool, pHandle);
}

void TRM_destroyPipeline(uint32_t handle)
{
	if(vkDeviceWaitIdle(pState->device) != VK_SUCCESS) // this shouldn't be necessary
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_Backend_Pipeline* pPipeline = NULL;
	TRM_Arena_get(handle, pState->pipelinePool, (void**)&pPipeline);
	
	if(pPipeline->type == TRM_PIPELINE_TYPE_COMPUTE)
	{
		vkDestroyDescriptorSetLayout(pState->device, pPipeline->info.compute.descriptorSetLayout, pState->pAllocator);
		vkDestroyPipelineLayout(pState->device, pPipeline->info.compute.pipelineLayout, pState->pAllocator);
		vkDestroyPipeline(pState->device, pPipeline->info.compute.pipeline, pState->pAllocator);
	}
	else
	{
		vkDestroyDescriptorSetLayout(pState->device, pPipeline->info.graphics.descriptorSetLayout, pState->pAllocator);
		vkDestroyPipelineLayout(pState->device, pPipeline->info.graphics.pipelineLayout, pState->pAllocator);
		vkDestroyPipeline(pState->device, pPipeline->info.graphics.pipeline, pState->pAllocator);
		vkDestroyRenderPass(pState->device, pPipeline->info.graphics.renderPass, pState->pAllocator);
	}
}

uint32_t TRM_getDrawIndirectCommandSize(void)
{
	return sizeof(VkDrawIndirectCommand);
}

uint32_t TRM_getDrawIndexedIndirectCommandSize(void)
{
	return sizeof(VkDrawIndexedIndirectCommand);
}
