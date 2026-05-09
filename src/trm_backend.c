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
#define TRM_BACKEND_MAX_RESOURCE_COUNT 64
#define TRM_BACKEND_MAX_PASS_COUNT 64
#define TRM_MAX_DESCRIPTOR_SET_PER_FRAME_COUNT 32
#define TRM_MAX_FRAMEBUFFER_PER_FRAME_COUNT 8

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
	VkImageView imageView;
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
	uint32_t lastUsedSubmitionIndex;
	union
	{
		struct TRM_Backend_BufferResourceInfo buffer;
		struct TRM_Backend_BufferIndirectionResourceInfo bufferIndirection;
		struct TRM_Backend_ImageResourceInfo image;
	} info;
};

struct TRM_Backend_DispatchPass
{
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	uint32_t descriptorInfoCount;
	struct TRM_DescriptorInfo* pDescriptorInfos;
};

struct TRM_Backend_DrawPass
{
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkRenderPass renderPass;
	uint32_t descriptorInfoCount;
	struct TRM_DescriptorInfo* pDescriptorInfos;
};

struct TRM_Backend_Pass
{
	enum TRM_PassType type;
	union
	{
		struct TRM_Backend_DispatchPass dispatch;
		struct TRM_Backend_DrawPass draw;
	} info;
};

struct TRM_Backend_DispatchPassInstanceInfo
{
	uint32_t pass;
	uint32_t descriptorSet;
	uint32_t groupCountX;
	uint32_t groupCountY;
	uint32_t groupCountZ;
};

struct TRM_Backend_drawPassInstanceInfo
{
	uint32_t width;
	uint32_t height;
	uint32_t pass;
	uint32_t descriptorSet;
	uint32_t framebuffer;
	uint32_t vertexCount;
	uint32_t clearColorCount;
	VkClearValue* pClearColors;
};

struct TRM_Backend_ImageToImagePassInstanceInfo
{
	uint32_t width;
	uint32_t height;
};

struct TRM_Backend_BufferToImagePassInstanceInfo
{
	uint32_t width;
	uint32_t height;
};

struct TRM_Backend_BufferToBufferPassInstanceInfo
{
	uint32_t sizeInBytes;
};

struct TRM_Backend_BlitPassInstanceInfo
{
	uint32_t srcWidth;
	uint32_t srcHeight;
	uint32_t dstWidth;
	uint32_t dstHeight;
};

struct TRM_Backend_PassInstance
{
	enum TRM_PassType type;
	uint32_t bindingCount;
	uint32_t* pBindings;
	struct TRM_Backend_ResourceState* pResourceStates;
	union
	{
		struct TRM_Backend_DispatchPassInstanceInfo dispatch;
		struct TRM_Backend_drawPassInstanceInfo draw;
		struct TRM_Backend_ImageToImagePassInstanceInfo imageToImageCopy;
		struct TRM_Backend_BufferToImagePassInstanceInfo bufferToImageCopy;
		struct TRM_Backend_BufferToBufferPassInstanceInfo bufferToBufferCopy;
		struct TRM_Backend_BlitPassInstanceInfo blit;
	} info;
};

struct TRM_Backend_FrameInfo
{
	VkCommandBuffer commandBuffer;
	VkFence commandBufferExecutedFence;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore timelineSemaphore;
	uint32_t descriptorSetCount;
	VkDescriptorSet descriptorSets[TRM_MAX_DESCRIPTOR_SET_PER_FRAME_COUNT]; // descriptor sets are created/destroyed each frame (BAD)
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
	uint32_t frameIndex;
	uint64_t submitionIndex;
	VkSampler globalSampler;
	struct TRM_Arena resourcePool;
	struct TRM_LinkedList resourceHandles;
	struct TRM_Arena passPool;
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
		exit(EXIT_FAILURE;)
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
	deviceCreateInfo.pEnabledFeatures = NULL;

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

static void TRM_Backend_createImageView(
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
	samplerCreateInfo.maxLod = 0.0f;
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
	uniformBufferDescriptorPoolSize.descriptorCount = 10;
	uniformBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

	VkDescriptorPoolSize storageBufferDescriptorPoolSize = {0};
	storageBufferDescriptorPoolSize.descriptorCount = 10;
	storageBufferDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

	VkDescriptorPoolSize storageImageDescriptorPoolSize = {0};
	storageImageDescriptorPoolSize.descriptorCount = 10;
	storageImageDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

	VkDescriptorPoolSize combinedImageSamplerDescriptorPoolSize = {0};
	combinedImageSamplerDescriptorPoolSize.descriptorCount = 10;
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
	descriptorPoolCreateInfo.maxSets = 10;
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

	VkPipelineColorBlendAttachmentState* pColorBlendAttachments = NULL;
	TRM_Memory_allocate(sizeof(VkPipelineColorBlendAttachmentState) * colorAttachmentCount, (void**)&pColorBlendAttachments);

	for(uint32_t colorBlendAttachmentIndex = 0; colorBlendAttachmentIndex < colorAttachmentCount; ++colorBlendAttachmentIndex)
	{
		pColorBlendAttachments[colorBlendAttachmentIndex].blendEnable = VK_FALSE;
		pColorBlendAttachments[colorBlendAttachmentIndex].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		pColorBlendAttachments[colorBlendAttachmentIndex].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		pColorBlendAttachments[colorBlendAttachmentIndex].colorBlendOp = VK_BLEND_OP_ADD;
		pColorBlendAttachments[colorBlendAttachmentIndex].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		pColorBlendAttachments[colorBlendAttachmentIndex].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		pColorBlendAttachments[colorBlendAttachmentIndex].alphaBlendOp = VK_BLEND_OP_ADD;
		pColorBlendAttachments[colorBlendAttachmentIndex].colorWriteMask =
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
	colorBlendStateInfo.pAttachments = pColorBlendAttachments;
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

	TRM_Memory_deallocate(pColorBlendAttachments);
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
			return pResource->info.bufferIndirection.info.hostVisible.buffers[pState->frameIndex];
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

static void TRM_Backend_createResource(struct TRM_ResourceCreateInfo info, uint32_t* pHandle)
{
	if(info.type == TRM_RESOURCE_TYPE_BUFFER)
	{
		VkBufferUsageFlags bufferUsage = 0;

		if((info.info.buffer.usage & TRM_BUFFER_USAGE_UNIFORM) != 0)
		{
			bufferUsage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		}

		if((info.info.buffer.usage & TRM_BUFFER_USAGE_STORAGE) != 0)
		{
			bufferUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		}

		if((info.info.buffer.usage & TRM_BUFFER_USAGE_TRANSFER_SRC) != 0)
		{
			bufferUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		}

		if((info.info.buffer.usage & TRM_BUFFER_USAGE_TRANSFER_DST) != 0)
		{
			bufferUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		}

		if((info.info.buffer.usage & TRM_BUFFER_USAGE_VERTEX) != 0)
		{
			bufferUsage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		}

		struct TRM_Backend_Resource bufferIndirection = {0};
		bufferIndirection.type = TRM_BACKEND_RESOURCE_TYPE_BUFFER_INDIRECTION;
		bufferIndirection.info.bufferIndirection.hostVisible = info.info.buffer.hostVisible;
		bufferIndirection.toDelete = false;
		bufferIndirection.lastUsedSubmitionIndex = 0;

		if(info.info.buffer.hostVisible)
		{
			for(uint32_t frameIndex = 0; frameIndex < TRM_BACKEND_FRAME_COUNT; ++frameIndex)
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
				bufferIndirection.info.bufferIndirection.info.hostVisible.buffers[frameIndex] = buffer;
			}
		}
		else
		{
			struct TRM_Backend_Resource resource = {0};
			resource.type = TRM_BACKEND_RESOURCE_TYPE_BUFFER;
			resource.state.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			resource.state.access = VK_ACCESS_NONE;
			bufferIndirection.toDelete = false;
			bufferIndirection.lastUsedSubmitionIndex = 0;

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
		VkImageAspectFlags imageAspect = 0;
		VkImageUsageFlags imageUsage = 0;

		if((info.info.image.usage & TRM_IMAGE_USAGE_COLOR_ATTACHMENT) != 0)
		{
			imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if((info.info.image.usage & TRM_IMAGE_USAGE_DEPTH_ATTACHMENT) != 0)
		{
			imageUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
			imageAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
		}

		if((info.info.image.usage & TRM_IMAGE_USAGE_SAMPLED) != 0)
		{
			imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if((info.info.image.usage & TRM_IMAGE_USAGE_STORAGE) != 0)
		{
			imageUsage |= VK_IMAGE_USAGE_STORAGE_BIT;
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if((info.info.image.usage & TRM_IMAGE_USAGE_TRANSFER_SRC) != 0)
		{
			imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if((info.info.image.usage & TRM_IMAGE_USAGE_TRANSFER_DST) != 0)
		{
			imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			imageAspect |= VK_IMAGE_ASPECT_COLOR_BIT;
		}

		struct TRM_Backend_Resource resource = {0};
		resource.type = TRM_BACKEND_RESOURCE_TYPE_IMAGE;
		resource.toDelete = false;
		resource.lastUsedSubmitionIndex = 0;
		resource.info.image.aspect = imageAspect;
		resource.info.image.swapchainImage = false;
		resource.state.stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		resource.state.access = VK_ACCESS_NONE;
		resource.state.layout = VK_IMAGE_LAYOUT_UNDEFINED;

		TRM_Backend_createImage(
			pState->pAllocator,
			pState->device,
			info.info.image.width,
			info.info.image.height,
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
			TRM_Backend_convertFormat(info.info.image.format),
			resource.info.image.aspect,
			&resource.info.image.imageView);

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
			for(uint32_t frameIndex = 0; frameIndex < TRM_BACKEND_FRAME_COUNT; ++frameIndex)
			{
				struct TRM_Backend_Resource* pBuffer = NULL;
				TRM_Arena_get(pResource->info.bufferIndirection.info.hostVisible.buffers[frameIndex], pState->resourcePool, (void**)&pBuffer);
				vkDestroyBuffer(pState->device, pBuffer->info.buffer.buffer, pState->pAllocator);
				vkFreeMemory(pState->device, pBuffer->info.buffer.memory, pState->pAllocator);
				TRM_Arena_remove(pResource->info.bufferIndirection.info.hostVisible.buffers[frameIndex], &pState->resourcePool);
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
		}
		vkDestroyImageView(pState->device, pResource->info.image.imageView, pState->pAllocator);
	}

	TRM_Arena_remove(handle, &pState->resourcePool);
}

static void TRM_Backend_RecreateSwapchain(uint32_t width, uint32_t height)
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
			pState->swapchainFormat,
			VK_IMAGE_ASPECT_COLOR_BIT,
			&swapchainColorImage.info.image.imageView);

		swapchainColorImage.info.image.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		swapchainColorImage.info.image.memory = VK_NULL_HANDLE;
		swapchainColorImage.info.image.swapchainImage = true;
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

// converts an API PassInstance into internal an PassInstance, it prepares command parameters and expected resource states for this pass instance
static void TRM_Backend_createPassInstances(
	uint32_t swapchainImageIndex, 
	uint32_t passInstanceCount, 
	struct TRM_PassInstance* pPassInstances, 
	struct TRM_Backend_PassInstance* pBackendPassInstances)
{
	for(uint32_t passInstanceIndex = 0; passInstanceIndex < passInstanceCount; ++passInstanceIndex)
	{
		struct TRM_PassInstance* pPassInstance = &pPassInstances[passInstanceIndex];
		struct TRM_Backend_PassInstance* pBackendPassInstance = &pBackendPassInstances[passInstanceIndex];

		TRM_Memory_allocate(
			sizeof(struct TRM_Backend_ResourceState) * TRM_BACKEND_MAX_RESOURCE_COUNT,
			(void**)&pBackendPassInstance->pResourceStates);
		TRM_Memory_memzero(
			sizeof(struct TRM_Backend_ResourceState) * TRM_BACKEND_MAX_RESOURCE_COUNT,
			pBackendPassInstance->pResourceStates);

		pBackendPassInstance->type = pPassInstance->type;

		switch(pPassInstance->type)
		{
		case TRM_PASS_TYPE_DISPATCH:
		{
			pBackendPassInstance->bindingCount = pPassInstance->info.dispatch.bindingCount;
			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			for(uint32_t bindingIndex = 0; bindingIndex < pBackendPassInstance->bindingCount; ++bindingIndex)
			{
				pBackendPassInstance->pBindings[bindingIndex] = 
					TRM_Backend_translateResource(pPassInstance->info.dispatch.pBindings[bindingIndex], swapchainImageIndex);
			}

			pBackendPassInstance->info.dispatch.pass = pPassInstance->info.dispatch.pass;
			pBackendPassInstance->info.dispatch.groupCountX = pPassInstance->info.dispatch.groupCountX; // what if we want to dispatch on a swapchain image sized buffer ?
			pBackendPassInstance->info.dispatch.groupCountY = pPassInstance->info.dispatch.groupCountY;
			pBackendPassInstance->info.dispatch.groupCountZ = pPassInstance->info.dispatch.groupCountZ;

			for(uint32_t bindingIndex = 0; bindingIndex < pBackendPassInstance->bindingCount; ++bindingIndex)
			{
				uint32_t resource = pBackendPassInstance->pBindings[bindingIndex];
				struct TRM_Backend_Resource* pResource = NULL;
				TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);
				struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[resource];
				struct TRM_Backend_Pass* pPass = NULL;
				TRM_Arena_get(pBackendPassInstance->info.dispatch.pass, pState->passPool, (void**)&pPass);

				VkAccessFlags accessFlags =
					(pPass->info.dispatch.pDescriptorInfos[bindingIndex].resourceAccessFlags & TRM_SHADER_ACCESS_FLAG_READ) != 0 ?
					VK_ACCESS_SHADER_READ_BIT :
					0;
				accessFlags |=
					((pPass->info.dispatch.pDescriptorInfos[bindingIndex].resourceAccessFlags & TRM_SHADER_ACCESS_FLAG_WRITE) != 0 ?
						VK_ACCESS_SHADER_WRITE_BIT :
						0);

				pResourceState->access |= accessFlags;
				VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
				if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
				{
					layout =
						pPass->info.dispatch.pDescriptorInfos[bindingIndex].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
						VK_IMAGE_LAYOUT_GENERAL :
						VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
				}
				pResourceState->stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			}
			break;
		}
		case TRM_PASS_TYPE_DRAW:
		{
			// 1 vertex buffer + x color attachments + 1 depth attachment + x additional bindings
			pBackendPassInstance->bindingCount = 1 + pPassInstance->info.draw.colorOutputImageCount + 1 + pPassInstance->info.draw.bindingCount;
			
			const uint32_t vertexBufferOffset = 0;
			const uint32_t colorAttachmentsOffset = vertexBufferOffset + 1;
			const uint32_t depthAttachmentOffset = colorAttachmentsOffset + pPassInstance->info.draw.colorOutputImageCount;
			const uint32_t additionalBindingsOffset = depthAttachmentOffset + 1;

			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			pBackendPassInstance->pBindings[vertexBufferOffset] =
				TRM_Backend_translateResource(pPassInstance->info.draw.vertexBuffer, swapchainImageIndex);

			for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < pPassInstance->info.draw.colorOutputImageCount; ++colorOutputImageIndex)
			{
				pBackendPassInstance->pBindings[colorAttachmentsOffset + colorOutputImageIndex] =
					TRM_Backend_translateResource(pPassInstance->info.draw.pColorOutputImages[colorOutputImageIndex], swapchainImageIndex);
			}

			pBackendPassInstance->pBindings[depthAttachmentOffset] =
				TRM_Backend_translateResource(pPassInstance->info.draw.depthOutputImage, swapchainImageIndex);

			for(uint32_t additionalBindingIndex = 0; additionalBindingIndex < pPassInstance->info.draw.bindingCount; ++additionalBindingIndex)
			{
				pBackendPassInstance->pBindings[additionalBindingsOffset + additionalBindingIndex] =
					TRM_Backend_translateResource(pPassInstance->info.draw.pBindings[additionalBindingIndex], swapchainImageIndex);
			}

			pBackendPassInstance->info.draw.pass = pPassInstance->info.draw.pass; // translate
			pBackendPassInstance->info.draw.width = pPassInstance->info.draw.width;
			pBackendPassInstance->info.draw.height = pPassInstance->info.draw.height;
			pBackendPassInstance->info.draw.vertexCount = pPassInstance->info.draw.vertexCount;
			pBackendPassInstance->info.draw.clearColorCount = pPassInstance->info.draw.colorOutputImageCount + 1;
			
			TRM_Memory_allocate(sizeof(VkClearColorValue) * (pPassInstance->info.draw.colorOutputImageCount + 1), (void**)&pBackendPassInstance->info.draw.pClearColors);
			for(uint32_t clearColorIndex = 0; clearColorIndex < pPassInstance->info.draw.colorOutputImageCount; ++clearColorIndex)
			{
				pBackendPassInstance->info.draw.pClearColors[clearColorIndex].color.float32[0] = pPassInstance->info.draw.pClearColors[clearColorIndex].color[0];
				pBackendPassInstance->info.draw.pClearColors[clearColorIndex].color.float32[1] = pPassInstance->info.draw.pClearColors[clearColorIndex].color[1];
				pBackendPassInstance->info.draw.pClearColors[clearColorIndex].color.float32[2] = pPassInstance->info.draw.pClearColors[clearColorIndex].color[2];
				pBackendPassInstance->info.draw.pClearColors[clearColorIndex].color.float32[3] = pPassInstance->info.draw.pClearColors[clearColorIndex].color[3];
			}

			pBackendPassInstance->info.draw.pClearColors[pPassInstance->info.draw.colorOutputImageCount].depthStencil.depth = 1.0f;
			pBackendPassInstance->info.draw.pClearColors[pPassInstance->info.draw.colorOutputImageCount].depthStencil.stencil = 0;

			VkImageLayout layout;
			struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[pBackendPassInstance->pBindings[vertexBufferOffset]];
			pResourceState->access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;

			for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < pPassInstance->info.draw.colorOutputImageCount; ++colorOutputImageIndex)
			{
				pResourceState = &pBackendPassInstance->pResourceStates[pBackendPassInstance->pBindings[colorAttachmentsOffset + colorOutputImageIndex]];
				pResourceState->access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				pResourceState->stage |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
				layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
			}

			pResourceState = &pBackendPassInstance->pResourceStates[pBackendPassInstance->pBindings[depthAttachmentOffset]];
			pResourceState->access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);

			struct TRM_Backend_Pass* pPass = NULL;
			TRM_Arena_get(pBackendPassInstance->info.draw.pass, pState->passPool, (void**)&pPass);

			for(uint32_t additionalBindingIndex = 0; additionalBindingIndex < pPassInstance->info.draw.bindingCount; ++additionalBindingIndex)
			{
				uint32_t resource = pBackendPassInstance->pBindings[additionalBindingsOffset + additionalBindingIndex];
				struct TRM_Backend_Resource* pResource = NULL;
				TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

				pResourceState = &pBackendPassInstance->pResourceStates[resource];

				VkAccessFlags accessFlags =
					(pPass->info.draw.pDescriptorInfos[additionalBindingIndex].resourceAccessFlags & TRM_SHADER_ACCESS_FLAG_READ) != 0 ?
					VK_ACCESS_SHADER_READ_BIT :
					0;
				accessFlags |=
					((pPass->info.draw.pDescriptorInfos[additionalBindingIndex].resourceAccessFlags & TRM_SHADER_ACCESS_FLAG_WRITE) != 0 ?
						VK_ACCESS_SHADER_WRITE_BIT :
						0);

				pResourceState->access |= accessFlags;
				layout = VK_IMAGE_LAYOUT_UNDEFINED;
				if(pResource->type == TRM_BACKEND_RESOURCE_TYPE_IMAGE)
				{
					layout = pPass->info.draw.pDescriptorInfos[additionalBindingIndex].descriptorType == TRM_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
						VK_IMAGE_LAYOUT_GENERAL :
						VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
				}
				pResourceState->stage |= (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
			}

			// framebuffer creation - feels out of place
			VkImageView* pAttachments = NULL;
			TRM_Memory_allocate(sizeof(VkImageView) * (pPassInstance->info.draw.colorOutputImageCount + 1), (void**)&pAttachments);

			for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < pPassInstance->info.draw.colorOutputImageCount; ++colorOutputImageIndex)
			{
				struct TRM_Backend_Resource* pColorImage = NULL;
				TRM_Arena_get(pBackendPassInstance->pBindings[colorAttachmentsOffset + colorOutputImageIndex], pState->resourcePool, (void**)&pColorImage);
				pAttachments[colorOutputImageIndex] = pColorImage->info.image.imageView;
			}

			struct TRM_Backend_Resource* pDepthImage = NULL;
			TRM_Arena_get(pBackendPassInstance->pBindings[depthAttachmentOffset], pState->resourcePool, (void**)&pDepthImage);
			pAttachments[pPassInstance->info.draw.colorOutputImageCount] = pDepthImage->info.image.imageView;

			TRM_Backend_createFramebuffer(
				pState->pAllocator,
				pState->device,
				pPass->info.draw.renderPass,
				pPassInstance->info.draw.colorOutputImageCount + 1,
				pAttachments,
				pBackendPassInstance->info.draw.width,
				pBackendPassInstance->info.draw.height,
				&pState->pFrameInfos[pState->frameIndex].framebuffers[pState->pFrameInfos[pState->frameIndex].framebufferCount]);

			TRM_Memory_deallocate(pAttachments);

			pBackendPassInstance->info.draw.framebuffer = pState->pFrameInfos[pState->frameIndex].framebufferCount;
			pState->pFrameInfos[pState->frameIndex].framebufferCount += 1;
			break;
		}
		case TRM_PASS_TYPE_IMAGE_TO_IMAGE_COPY:
		{
			pBackendPassInstance->bindingCount = 2;
			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			pBackendPassInstance->pBindings[0] = 
				TRM_Backend_translateResource(pPassInstance->info.imageToImageCopy.srcImage, swapchainImageIndex);
			pBackendPassInstance->pBindings[1] = 
				TRM_Backend_translateResource(pPassInstance->info.imageToImageCopy.dstImage, swapchainImageIndex);

			pBackendPassInstance->info.imageToImageCopy.width = pPassInstance->info.imageToImageCopy.width;
			pBackendPassInstance->info.imageToImageCopy.height = pPassInstance->info.imageToImageCopy.height;

			uint32_t srcImage = pBackendPassInstance->pBindings[0];
			uint32_t dstImage = pBackendPassInstance->pBindings[1];

			VkImageLayout layout;
			struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[srcImage];
			pResourceState->access |= VK_ACCESS_TRANSFER_READ_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
			layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);

			pResourceState = &pBackendPassInstance->pResourceStates[dstImage];
			pResourceState->access |= VK_ACCESS_TRANSFER_WRITE_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
			layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
			break;
		}
		case TRM_PASS_TYPE_BUFFER_TO_IMAGE_COPY:
		{
			pBackendPassInstance->bindingCount = 2;
			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			pBackendPassInstance->pBindings[0] = 
				TRM_Backend_translateResource(pPassInstance->info.bufferToImageCopy.srcBuffer, swapchainImageIndex);
			pBackendPassInstance->pBindings[1] = 
				TRM_Backend_translateResource(pPassInstance->info.bufferToImageCopy.dstImage, swapchainImageIndex);

			pBackendPassInstance->info.bufferToImageCopy.width = pPassInstance->info.bufferToImageCopy.width;
			pBackendPassInstance->info.bufferToImageCopy.height = pPassInstance->info.bufferToImageCopy.height;

			uint32_t srcBuffer = pBackendPassInstance->pBindings[0];
			uint32_t dstImage = pBackendPassInstance->pBindings[1];

			struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[srcBuffer];
			pResourceState->access |= VK_ACCESS_TRANSFER_READ_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

			pResourceState = &pBackendPassInstance->pResourceStates[dstImage];
			pResourceState->access |= VK_ACCESS_TRANSFER_WRITE_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
			break;
		}
		case TRM_PASS_TYPE_BUFFER_TO_BUFFER_COPY:
		{
			pBackendPassInstance->bindingCount = 2;
			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			pBackendPassInstance->pBindings[0] = 
				TRM_Backend_translateResource(pPassInstance->info.bufferToBufferCopy.srcBuffer, swapchainImageIndex);
			pBackendPassInstance->pBindings[1] = 
				TRM_Backend_translateResource(pPassInstance->info.bufferToBufferCopy.dstBuffer, swapchainImageIndex);

			pBackendPassInstance->info.bufferToBufferCopy.sizeInBytes = pPassInstance->info.bufferToBufferCopy.sizeInBytes;

			uint32_t srcBuffer = pBackendPassInstance->pBindings[0];
			uint32_t dstBuffer = pBackendPassInstance->pBindings[1];

			struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[srcBuffer];
			pResourceState->access |= VK_ACCESS_TRANSFER_READ_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;

			pResourceState = &pBackendPassInstance->pResourceStates[dstBuffer];
			pResourceState->access |= VK_ACCESS_TRANSFER_WRITE_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
			break;
		}
		case TRM_PASS_TYPE_BLIT:
		{
			pBackendPassInstance->bindingCount = 2;
			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			pBackendPassInstance->pBindings[0] =
				TRM_Backend_translateResource(pPassInstance->info.blit.srcImage, swapchainImageIndex);
			pBackendPassInstance->pBindings[1] =
				TRM_Backend_translateResource(pPassInstance->info.blit.dstImage, swapchainImageIndex);

			pBackendPassInstance->info.blit.srcWidth = pPassInstance->info.blit.srcWidth;
			pBackendPassInstance->info.blit.srcHeight = pPassInstance->info.blit.srcHeight;
			pBackendPassInstance->info.blit.dstWidth = pPassInstance->info.blit.dstWidth;
			pBackendPassInstance->info.blit.dstHeight = pPassInstance->info.blit.dstHeight;

			uint32_t srcImage = pBackendPassInstance->pBindings[0];
			uint32_t dstImage = pBackendPassInstance->pBindings[1];

			VkImageLayout layout;
			struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[srcImage];
			pResourceState->access |= VK_ACCESS_TRANSFER_READ_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
			layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);

			pResourceState = &pBackendPassInstance->pResourceStates[dstImage];
			pResourceState->access |= VK_ACCESS_TRANSFER_WRITE_BIT;
			pResourceState->stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
			layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
			break;
		}
		case TRM_PASS_TYPE_PRESENT:
		{
			pBackendPassInstance->bindingCount = 1;
			TRM_Memory_allocate(sizeof(uint32_t) * pBackendPassInstance->bindingCount, (void**)&pBackendPassInstance->pBindings);

			pBackendPassInstance->pBindings[0] = 
				TRM_Backend_translateResource(pState->pSwapchainImageInfos[swapchainImageIndex].colorImage, swapchainImageIndex);

			uint32_t colorImage = pBackendPassInstance->pBindings[0];

			struct TRM_Backend_ResourceState* pResourceState = &pBackendPassInstance->pResourceStates[colorImage];
			pResourceState->stage |= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
			VkImageLayout layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			TRM_Backend_mergeLayouts(layout, pResourceState->layout, &pResourceState->layout);
			break;
		}
		default: break;
		}

		for(uint32_t bindingIndex = 0; bindingIndex < pBackendPassInstance->bindingCount; ++bindingIndex)
		{
			struct TRM_Backend_Resource* pResource = NULL;
			TRM_Arena_get(pBackendPassInstance->pBindings[bindingIndex], pState->resourcePool, (void**)&pResource);
			pResource->lastUsedSubmitionIndex = pState->frameIndex;
		}
	}
}

// update descriptor sets
static void TRM_Backend_updateDescriptorSets(uint32_t passInstanceCount, struct TRM_Backend_PassInstance* pPassInstances)
{
	for(uint32_t passInstanceIndex = 0; passInstanceIndex < passInstanceCount; ++passInstanceIndex)
	{
		if(pPassInstances[passInstanceIndex].type == TRM_PASS_TYPE_DISPATCH || 
			pPassInstances[passInstanceIndex].type == TRM_PASS_TYPE_DRAW)
		{
			struct TRM_Backend_PassInstance* pPassInstance = &pPassInstances[passInstanceIndex];
			const uint32_t bindingOffset = pPassInstance->type == TRM_PASS_TYPE_DRAW ? 3 : 0;
			
			VkWriteDescriptorSet* pWrites = NULL;
			VkDescriptorBufferInfo* pBufferInfos = NULL;
			VkDescriptorImageInfo* pImageInfos = NULL;
			TRM_Memory_allocate(sizeof(VkWriteDescriptorSet) * pPassInstance->bindingCount, (void**)&pWrites);
			TRM_Memory_allocate(sizeof(VkDescriptorBufferInfo) * pPassInstance->bindingCount, (void**)&pBufferInfos);
			TRM_Memory_allocate(sizeof(VkDescriptorImageInfo) * pPassInstance->bindingCount, (void**)&pImageInfos);
			
			VkDescriptorSet* pDescriptorSet = NULL;
			struct TRM_Backend_Pass* pPass = NULL;
			if(pPassInstance->type == TRM_PASS_TYPE_DISPATCH)
			{
				TRM_Arena_get(pPassInstance->info.dispatch.pass, pState->passPool, (void**)&pPass);
				pDescriptorSet = 
					&pState->pFrameInfos[pState->frameIndex].descriptorSets[pState->pFrameInfos[pState->frameIndex].descriptorSetCount];
				TRM_Backend_allocateDescriptorSet(
					pState->device, 
					pState->descriptorPool, 
					pPass->info.dispatch.descriptorSetLayout, 
					pDescriptorSet);
				pPassInstance->info.dispatch.descriptorSet = pState->pFrameInfos[pState->frameIndex].descriptorSetCount;
			}
			else
			{
				TRM_Arena_get(pPassInstance->info.draw.pass, pState->passPool, (void**)&pPass);
				pDescriptorSet = 
					&pState->pFrameInfos[pState->frameIndex].descriptorSets[pState->pFrameInfos[pState->frameIndex].descriptorSetCount];
				TRM_Backend_allocateDescriptorSet(
					pState->device, 
					pState->descriptorPool, 
					pPass->info.draw.descriptorSetLayout, 
					pDescriptorSet);
				pPassInstance->info.draw.descriptorSet = pState->pFrameInfos[pState->frameIndex].descriptorSetCount;
			}
			pState->pFrameInfos[pState->frameIndex].descriptorSetCount += 1;
			
			for(uint32_t bindingIndex = 0; bindingIndex < pPassInstance->bindingCount - bindingOffset; ++bindingIndex)
			{
				VkDescriptorType descriptorType;
				const uint32_t resource = pPassInstance->pBindings[bindingIndex + bindingOffset];
				struct TRM_Backend_Resource* pResource = NULL;
				TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

				descriptorType = pPassInstance->type == TRM_PASS_TYPE_DISPATCH ?
					TRM_Backend_convertDescriptorType(pPass->info.dispatch.pDescriptorInfos[bindingIndex].descriptorType) :
					TRM_Backend_convertDescriptorType(pPass->info.draw.pDescriptorInfos[bindingIndex].descriptorType);

				struct TRM_Backend_ResourceState* pResourceState = &pPassInstance->pResourceStates[resource];

				pWrites[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				pWrites[bindingIndex].pNext = NULL;
				pWrites[bindingIndex].dstBinding = bindingIndex;
				pWrites[bindingIndex].dstArrayElement = 0;
				pWrites[bindingIndex].descriptorCount = 1;
				pWrites[bindingIndex].descriptorType = descriptorType;

				if(descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
				{
					pBufferInfos[bindingIndex].buffer = pResource->info.buffer.buffer;
					pBufferInfos[bindingIndex].offset = 0;
					pBufferInfos[bindingIndex].range = VK_WHOLE_SIZE;

					pWrites[bindingIndex].pBufferInfo = &pBufferInfos[bindingIndex];
					pWrites[bindingIndex].pImageInfo = NULL;
					pWrites[bindingIndex].pTexelBufferView = NULL;

				}
				else if(descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
				{
					pImageInfos[bindingIndex].imageLayout = pResourceState->layout;
					pImageInfos[bindingIndex].imageView = pResource->info.image.imageView;
					pImageInfos[bindingIndex].sampler = VK_NULL_HANDLE;

					pWrites[bindingIndex].pBufferInfo = NULL;
					pWrites[bindingIndex].pImageInfo = &pImageInfos[bindingIndex];
				}
				else if(descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
				{
					pImageInfos[bindingIndex].imageLayout = pResourceState->layout;
					pImageInfos[bindingIndex].imageView = pResource->info.image.imageView;
					pImageInfos[bindingIndex].sampler = pState->globalSampler;

					pWrites[bindingIndex].pBufferInfo = NULL;
					pWrites[bindingIndex].pImageInfo = &pImageInfos[bindingIndex];
				}
				else
				{
					exit(EXIT_FAILURE);
				}

				pWrites[bindingIndex].dstSet = *pDescriptorSet;
				pWrites[bindingIndex].pTexelBufferView = NULL;
			}

			vkUpdateDescriptorSets(pState->device, pPassInstance->bindingCount - bindingOffset, pWrites, 0, NULL);

			TRM_Memory_deallocate(pBufferInfos);
			TRM_Memory_deallocate(pImageInfos);
			TRM_Memory_deallocate(pWrites);
		}
	}
}

static void TRM_Backend_fillCommandBuffer(
	uint32_t passInstanceCount,
	struct TRM_Backend_PassInstance* pPassInstances,
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

	for(uint32_t passInstanceIndex = 0; passInstanceIndex < passInstanceCount; ++passInstanceIndex)
	{
		VkPipelineStageFlags srcStageFlags = 0;
		VkPipelineStageFlags dstStageFlags = 0;
		struct TRM_DynamicArray bufferMemoryBarriers;
		struct TRM_DynamicArray imageMemoryBarriers;

		TRM_DynamicArray_create(sizeof(VkBufferMemoryBarrier), &bufferMemoryBarriers);
		TRM_DynamicArray_create(sizeof(VkImageMemoryBarrier), &imageMemoryBarriers);

		const struct TRM_Backend_PassInstance* pPassInstance = &pPassInstances[passInstanceIndex];

		// retrieves all barriers (memory + execution) for this pass for each resource
		bool needBarrier = false;
		for(uint32_t bindingIndex = 0; bindingIndex < pPassInstance->bindingCount; ++bindingIndex)
		{
			const uint32_t resource = pPassInstance->pBindings[bindingIndex];
			struct TRM_Backend_Resource* pResource = NULL;
			TRM_Arena_get(resource, pState->resourcePool, (void**)&pResource);

			struct TRM_Backend_ResourceState* pPreviousResourceState = &pResource->state;
			struct TRM_Backend_ResourceState* pCurrentResourceState = &pPassInstance->pResourceStates[resource];

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
					imageMemoryBarrier.subresourceRange.levelCount = 1;
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

		if(pPassInstance->type == TRM_PASS_TYPE_DISPATCH)
		{
			struct TRM_Backend_Pass* pPass = NULL;
			TRM_Arena_get(pPassInstance->info.dispatch.pass, pState->passPool, (void**)&pPass);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_COMPUTE,
				pPass->info.dispatch.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndex].descriptorSets[pPassInstance->info.dispatch.descriptorSet],
				0,
				NULL);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pPass->info.dispatch.pipeline);

			vkCmdDispatch(
				commandBuffer,
				pPassInstance->info.dispatch.groupCountX,
				pPassInstance->info.dispatch.groupCountY,
				pPassInstance->info.dispatch.groupCountZ);
		}
		else if(pPassInstance->type == TRM_PASS_TYPE_DRAW)
		{
			struct TRM_Backend_Pass* pPass = NULL;
			TRM_Arena_get(pPassInstance->info.draw.pass, pState->passPool, (void**)&pPass);

			VkRenderPassBeginInfo renderPassBeginInfo = {0};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.pNext = NULL;
			renderPassBeginInfo.renderPass = pPass->info.draw.renderPass;
			renderPassBeginInfo.framebuffer = pState->pFrameInfos[pState->frameIndex].framebuffers[pPassInstance->info.draw.framebuffer];
			renderPassBeginInfo.renderArea.offset.x = 0;
			renderPassBeginInfo.renderArea.offset.y = 0;
			renderPassBeginInfo.renderArea.extent.width = pPassInstance->info.draw.width;
			renderPassBeginInfo.renderArea.extent.height = pPassInstance->info.draw.height;
			renderPassBeginInfo.clearValueCount = pPassInstance->info.draw.clearColorCount;
			renderPassBeginInfo.pClearValues = pPassInstance->info.draw.pClearColors;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[0], pState->resourcePool, (void**)&pInputResource);

			VkBuffer vertexBuffer = pInputResource->info.buffer.buffer;
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPass->info.draw.pipeline);

			vkCmdBindDescriptorSets(
				commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				pPass->info.draw.pipelineLayout,
				0,
				1,
				&pState->pFrameInfos[pState->frameIndex].descriptorSets[pPassInstance->info.draw.descriptorSet],
				0,
				NULL);

			VkViewport viewport = {0};
			viewport.x = 0;
			viewport.y = (float)pPassInstance->info.draw.height;
			viewport.width = (float)pPassInstance->info.draw.width;
			viewport.height = -(float)pPassInstance->info.draw.height;
			viewport.minDepth = 0;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor = {0};
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			scissor.extent.width = pPassInstance->info.draw.width;
			scissor.extent.height = pPassInstance->info.draw.height;

			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			vkCmdDraw(commandBuffer, pPassInstance->info.draw.vertexCount, 1, 0, 0);

			vkCmdEndRenderPass(commandBuffer);
		}
		else if(pPassInstance->type == TRM_PASS_TYPE_IMAGE_TO_IMAGE_COPY)
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[0], pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[1], pState->resourcePool, (void**)&pOutputResource);

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
			imageCopy.extent.width = pPassInstance->info.imageToImageCopy.width;
			imageCopy.extent.height = pPassInstance->info.imageToImageCopy.height;
			imageCopy.extent.depth = 1;

			vkCmdCopyImage(
				commandBuffer,
				pInputResource->info.image.image,
				pInputResource->state.layout,
				pOutputResource->info.image.image,
				pOutputResource->state.layout,
				1,
				&imageCopy);
		}
		else if(pPassInstances[passInstanceIndex].type == TRM_PASS_TYPE_BUFFER_TO_IMAGE_COPY)
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[0], pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[1], pState->resourcePool, (void**)&pOutputResource);

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
			bufferImageCopy.imageExtent.width = pPassInstance->info.bufferToImageCopy.width;
			bufferImageCopy.imageExtent.height = pPassInstance->info.bufferToImageCopy.height;
			bufferImageCopy.imageExtent.depth = 1;

			vkCmdCopyBufferToImage(
				commandBuffer,
				pInputResource->info.buffer.buffer,
				pOutputResource->info.image.image,
				pOutputResource->state.layout,
				1,
				&bufferImageCopy);
		}

		else if(pPassInstances[passInstanceIndex].type == TRM_PASS_TYPE_BUFFER_TO_BUFFER_COPY)
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[0], pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[1], pState->resourcePool, (void**)&pOutputResource);

			VkBufferCopy bufferCopy = {0};
			bufferCopy.srcOffset = 0;
			bufferCopy.dstOffset = 0;
			bufferCopy.size = pPassInstance->info.bufferToBufferCopy.sizeInBytes;

			vkCmdCopyBuffer(
				commandBuffer,
				pInputResource->info.buffer.buffer,
				pOutputResource->info.buffer.buffer,
				1,
				&bufferCopy);
		}
		else if(pPassInstances[passInstanceIndex].type == TRM_PASS_TYPE_BLIT)
		{
			struct TRM_Backend_Resource* pInputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[0], pState->resourcePool, (void**)&pInputResource);

			struct TRM_Backend_Resource* pOutputResource = NULL;
			TRM_Arena_get(pPassInstance->pBindings[1], pState->resourcePool, (void**)&pOutputResource);

			VkImageBlit blit = {0};
			blit.srcSubresource.aspectMask = pInputResource->info.image.aspect;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.srcSubresource.mipLevel = 0;
			blit.srcOffsets[0].x = 0;
			blit.srcOffsets[0].y = 0;
			blit.srcOffsets[0].z = 0;
			blit.srcOffsets[1].x = (int32_t)pPassInstances[passInstanceIndex].info.blit.srcWidth;
			blit.srcOffsets[1].y = (int32_t)pPassInstances[passInstanceIndex].info.blit.srcHeight;
			blit.srcOffsets[1].z = 1;
			blit.dstSubresource.aspectMask = pOutputResource->info.image.aspect;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;
			blit.dstSubresource.mipLevel = 0;
			blit.dstOffsets[0].x = 0;
			blit.dstOffsets[0].y = 0;
			blit.dstOffsets[0].z = 0;
			blit.dstOffsets[1].x = (int32_t)pPassInstances[passInstanceIndex].info.blit.dstWidth;
			blit.dstOffsets[1].y = (int32_t)pPassInstances[passInstanceIndex].info.blit.dstHeight;
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
	TRM_Arena_create(sizeof(struct TRM_Backend_Pass), TRM_BACKEND_MAX_PASS_COUNT, &pState->passPool);

	TRM_Memory_allocate(sizeof(struct TRM_Backend_FrameInfo) * TRM_BACKEND_FRAME_COUNT, (void**)&pState->pFrameInfos);
	TRM_Backend_createSampler(pState->pAllocator, pState->device, &pState->globalSampler);

	for(uint32_t frameIndex = 0; frameIndex < TRM_BACKEND_FRAME_COUNT; ++frameIndex)
	{
		TRM_Backend_allocateCommandBuffer(pState->commandPool, pState->device, &pState->pFrameInfos[frameIndex].commandBuffer);
		TRM_Backend_createFence(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndex].commandBufferExecutedFence);
		TRM_Backend_createSemaphore(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndex].imageAvailableSemaphore);
		TRM_Backend_createTimelineSemaphore(pState->pAllocator, pState->device, &pState->pFrameInfos[frameIndex].timelineSemaphore);
		pState->pFrameInfos[frameIndex].descriptorSetCount = 0;
		pState->pFrameInfos[frameIndex].framebufferCount = 0;
	}

	pState->frameIndex = 0;
	pState->submitionIndex = 0;
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

		for(uint32_t frameIndex = 0; frameIndex < TRM_BACKEND_FRAME_COUNT; ++frameIndex)
		{
			vkDestroySemaphore(pState->device, pState->pFrameInfos[frameIndex].timelineSemaphore, pState->pAllocator);
			vkDestroySemaphore(pState->device, pState->pFrameInfos[frameIndex].imageAvailableSemaphore, pState->pAllocator);
			vkDestroyFence(pState->device, pState->pFrameInfos[frameIndex].commandBufferExecutedFence, pState->pAllocator);

			for(uint32_t framebufferIndex = 0; framebufferIndex < pState->pFrameInfos[frameIndex].framebufferCount; ++framebufferIndex)
			{
				vkDestroyFramebuffer(pState->device, pState->pFrameInfos[frameIndex].framebuffers[framebufferIndex], pState->pAllocator);
			}
		}
		TRM_Memory_deallocate(pState->pFrameInfos);

		if(pState->swapchain != VK_NULL_HANDLE)
		{
			for(uint32_t swapchainImageIndex = 0; swapchainImageIndex < pState->swapchainImageCount; ++swapchainImageIndex)
			{
				struct TRM_Backend_Resource* pSwapchainColorImage;
				TRM_Arena_get(pState->pSwapchainImageInfos[swapchainImageIndex].colorImage, pState->resourcePool, (void**)&pSwapchainColorImage);

				vkDestroyImageView(pState->device, pSwapchainColorImage->info.image.imageView, pState->pAllocator);
				vkFreeMemory(pState->device, pSwapchainColorImage->info.image.memory, pState->pAllocator);
				vkDestroySemaphore(pState->device, pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore, pState->pAllocator);
			}
			TRM_Memory_deallocate(pState->pSwapchainImageInfos);
		}
		
		vkDestroySampler(pState->device, pState->globalSampler, pState->pAllocator);

		TRM_Arena_destroy(&pState->resourcePool);
		TRM_Arena_destroy(&pState->passPool);
		
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

void TRM_beginFrame(void)
{
	if(vkWaitForFences(pState->device, 1, &pState->pFrameInfos[pState->frameIndex].commandBufferExecutedFence, VK_FALSE, UINT64_MAX) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	for(uint32_t framebufferIndex = 0; framebufferIndex < pState->pFrameInfos[pState->frameIndex].framebufferCount; ++framebufferIndex)
	{
		vkDestroyFramebuffer(pState->device, pState->pFrameInfos[pState->frameIndex].framebuffers[framebufferIndex], pState->pAllocator);
	}
	pState->pFrameInfos[pState->frameIndex].framebufferCount = 0;
	
	if(pState->pFrameInfos[pState->frameIndex].descriptorSetCount > 0)
	{
		vkFreeDescriptorSets(
			pState->device, 
			pState->descriptorPool, 
			pState->pFrameInfos[pState->frameIndex].descriptorSetCount, 
			pState->pFrameInfos[pState->frameIndex].descriptorSets);
		pState->pFrameInfos[pState->frameIndex].descriptorSetCount = 0;
	}

	uint64_t completedSubmitionIndex = 0;
	vkGetSemaphoreCounterValue(pState->device, pState->pFrameInfos[pState->frameIndex].timelineSemaphore, &completedSubmitionIndex);

	struct TRM_LinkedList_Node* pResourceNode = pState->resourceHandles.pFirstNode;
	while(pResourceNode != NULL)
	{
		const uint32_t handle = *(uint32_t*)pResourceNode->pData;
		struct TRM_Backend_Resource* pResource = NULL;
		TRM_Arena_get(handle, pState->resourcePool, (void**)&pResource);
		if(pResource->toDelete && pResource->lastUsedSubmitionIndex <= completedSubmitionIndex)
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

void TRM_endFrame(uint32_t passInstanceCount, struct TRM_PassInstance* pPassInstances, uint32_t windowWidth, uint32_t windowHeight)
{
	if(windowWidth != pState->swapchainWidth || windowHeight != pState->swapchainHeight)
	{
		TRM_Backend_RecreateSwapchain(windowWidth, windowHeight);
		return;
	}

	uint32_t swapchainImageIndex = 0;
	VkResult acquireNextImageResult = vkAcquireNextImageKHR(
		pState->device,
		pState->swapchain,
		UINT64_MAX,
		pState->pFrameInfos[pState->frameIndex].imageAvailableSemaphore,
		VK_NULL_HANDLE,
		&swapchainImageIndex);
		
	if(acquireNextImageResult != VK_SUCCESS && 
		acquireNextImageResult != VK_ERROR_OUT_OF_DATE_KHR && 
		acquireNextImageResult != VK_SUBOPTIMAL_KHR)
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_Backend_PassInstance* pBackendPassInstances = NULL;
	TRM_Memory_allocate(sizeof(struct TRM_Backend_PassInstance) * passInstanceCount, (void**)&pBackendPassInstances);
	TRM_Memory_memzero(sizeof(struct TRM_Backend_PassInstance) * passInstanceCount, pBackendPassInstances);
	
	TRM_Backend_createPassInstances(swapchainImageIndex, passInstanceCount, pPassInstances, pBackendPassInstances);

	TRM_Backend_updateDescriptorSets(passInstanceCount, pBackendPassInstances);

	TRM_Backend_fillCommandBuffer(passInstanceCount, pBackendPassInstances, pState->pFrameInfos[pState->frameIndex].commandBuffer);

	for(uint32_t passInstanceIndex = 0; passInstanceIndex < passInstanceCount; ++passInstanceIndex)
	{
		TRM_Memory_deallocate(pBackendPassInstances[passInstanceIndex].pBindings);
		TRM_Memory_deallocate(pBackendPassInstances[passInstanceIndex].pResourceStates);

		if(pBackendPassInstances[passInstanceIndex].type == TRM_PASS_TYPE_DRAW)
		{
			TRM_Memory_deallocate(pBackendPassInstances[passInstanceIndex].info.draw.pClearColors);
		}
	}
	TRM_Memory_deallocate(pBackendPassInstances);

	VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

	VkSemaphore signalSemaphores[2] = {
		pState->pSwapchainImageInfos[swapchainImageIndex].imageRenderedSemaphore,
		pState->pFrameInfos[pState->frameIndex].timelineSemaphore
	};

	uint64_t signalValues[2] =
	{
		0, // ignored for binary semaphore
		pState->submitionIndex + 1
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
	submitInfo.pWaitSemaphores = &pState->pFrameInfos[pState->frameIndex].imageAvailableSemaphore;
	submitInfo.pWaitDstStageMask = &waitDstStageMask;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &pState->pFrameInfos[pState->frameIndex].commandBuffer;
	submitInfo.signalSemaphoreCount = 2;
	submitInfo.pSignalSemaphores = signalSemaphores;

	if(vkResetFences(pState->device, 1, &pState->pFrameInfos[pState->frameIndex].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	if(vkQueueSubmit(pState->queue, 1, &submitInfo, pState->pFrameInfos[pState->frameIndex].commandBufferExecutedFence) != VK_SUCCESS)
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

	pState->frameIndex = (pState->frameIndex + 1) % TRM_BACKEND_FRAME_COUNT;
	pState->submitionIndex += 1;
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

	uint32_t buffer = pBufferIndirection->info.bufferIndirection.info.hostVisible.buffers[pState->frameIndex];

	struct TRM_Backend_Resource* pResource = NULL;
	TRM_Arena_get(buffer, pState->resourcePool, (void**)&pResource);

	void* pMappedMemory = NULL;
	vkMapMemory(pState->device, pResource->info.buffer.memory, 0, sizeInBytes, 0, &pMappedMemory);
	TRM_Memory_memcpy(sizeInBytes, pData, pMappedMemory);
	vkUnmapMemory(pState->device, pResource->info.buffer.memory);
}

void TRM_createDispatchPass(struct TRM_DispatchPassCreateInfo info, uint32_t* pHandle)
{
	struct TRM_Backend_Pass pass = {0};
	
	pass.type = TRM_PASS_TYPE_DISPATCH;
	pass.info.dispatch.descriptorInfoCount = info.descriptorInfoCount;
	TRM_Memory_allocate(sizeof(struct TRM_DescriptorInfo) * info.descriptorInfoCount, (void**)&pass.info.dispatch.pDescriptorInfos);
	TRM_Memory_memcpy(
		sizeof(struct TRM_DescriptorInfo) * info.descriptorInfoCount, 
		info.pDescriptorInfos, 
		pass.info.dispatch.pDescriptorInfos);

	VkDescriptorSetLayoutBinding* pBindings = NULL;
	TRM_Memory_allocate(sizeof(VkDescriptorSetLayoutBinding) * info.descriptorInfoCount, (void**)&pBindings);
	
	for(uint32_t descriptorInfoIndex = 0; descriptorInfoIndex < info.descriptorInfoCount; ++descriptorInfoIndex)
	{
		VkDescriptorType descriptorType = TRM_Backend_convertDescriptorType(info.pDescriptorInfos[descriptorInfoIndex].descriptorType);

		pBindings[descriptorInfoIndex].binding = descriptorInfoIndex;
		pBindings[descriptorInfoIndex].descriptorCount = 1;
		pBindings[descriptorInfoIndex].descriptorType = descriptorType;
		pBindings[descriptorInfoIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pBindings[descriptorInfoIndex].pImmutableSamplers = NULL;
	}

	TRM_Backend_createDescriptorSetLayout(
		pState->pAllocator, 
		pState->device, 
		info.descriptorInfoCount, 
		pBindings, 
		&pass.info.dispatch.descriptorSetLayout);
	TRM_Memory_deallocate(pBindings);

	TRM_Backend_createPipelineLayout(
		pState->pAllocator, 
		pState->device, 
		1, 
		&pass.info.dispatch.descriptorSetLayout, 
		&pass.info.dispatch.pipelineLayout);

	VkShaderModule shaderModule;
	TRM_Backend_createShaderModule(pState->pAllocator, pState->device, info.codeSize, info.pCode, &shaderModule);

	TRM_Backend_createComputePipeline(
		pState->pAllocator, 
		pState->device, 
		shaderModule, 
		pass.info.dispatch.pipelineLayout, 
		&pass.info.dispatch.pipeline);

	vkDestroyShaderModule(pState->device, shaderModule, pState->pAllocator);

	TRM_Arena_add(&pass, &pState->passPool, pHandle);
}

void TRM_createDrawPass(struct TRM_DrawPassCreateInfo info, uint32_t* pHandle)
{
	struct TRM_Backend_Pass pass = {0};
	
	pass.type = TRM_PASS_TYPE_DRAW;
	pass.info.draw.descriptorInfoCount = info.descriptorInfoCount;
	TRM_Memory_allocate(sizeof(struct TRM_DescriptorInfo) * info.descriptorInfoCount, (void**)&pass.info.draw.pDescriptorInfos);
	TRM_Memory_memcpy(
		sizeof(struct TRM_DescriptorInfo) * info.descriptorInfoCount, 
		info.pDescriptorInfos, 
		pass.info.draw.pDescriptorInfos);

	VkDescriptorSetLayoutBinding* pBindings = NULL;
	TRM_Memory_allocate(sizeof(VkDescriptorSetLayoutBinding) * info.descriptorInfoCount, (void**)&pBindings);
	for(uint32_t descriptorInfoIndex = 0; descriptorInfoIndex < info.descriptorInfoCount; ++descriptorInfoIndex)
	{
		VkDescriptorType descriptorType = TRM_Backend_convertDescriptorType(info.pDescriptorInfos[descriptorInfoIndex].descriptorType);

		pBindings[descriptorInfoIndex].binding = descriptorInfoIndex;
		pBindings[descriptorInfoIndex].descriptorCount = 1;
		pBindings[descriptorInfoIndex].descriptorType = descriptorType;
		pBindings[descriptorInfoIndex].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		pBindings[descriptorInfoIndex].pImmutableSamplers = NULL;
	}

	TRM_Backend_createDescriptorSetLayout(
		pState->pAllocator, 
		pState->device, 
		info.descriptorInfoCount, 
		pBindings, 
		&pass.info.draw.descriptorSetLayout);
	TRM_Memory_deallocate(pBindings);

	TRM_Backend_createPipelineLayout(
		pState->pAllocator, 
		pState->device, 
		1, 
		&pass.info.draw.descriptorSetLayout, 
		&pass.info.draw.pipelineLayout);

	VkShaderModule vertexShaderModule;
	TRM_Backend_createShaderModule(pState->pAllocator, pState->device, info.vertexCodeSize, (uint32_t*)info.pVertexCode, &vertexShaderModule);

	VkShaderModule fragmentShaderModule;
	TRM_Backend_createShaderModule(pState->pAllocator, pState->device, info.fragmentCodeSize, (uint32_t*)info.pFragmentCode, &fragmentShaderModule);

	VkVertexInputAttributeDescription* pVertexAttributeDescriptions = NULL;
	TRM_Memory_allocate(sizeof(VkVertexInputAttributeDescription) * info.vertexAttributeDescriptionCount, (void**)&pVertexAttributeDescriptions);

	for(uint32_t i = 0; i < info.vertexAttributeDescriptionCount; ++i)
	{
		pVertexAttributeDescriptions[i].binding = 0;
		pVertexAttributeDescriptions[i].format = TRM_Backend_convertFormat(info.pVertexAttributeDescriptions[i].format);
		pVertexAttributeDescriptions[i].location = info.pVertexAttributeDescriptions[i].shaderLocation;
		pVertexAttributeDescriptions[i].offset = info.pVertexAttributeDescriptions[i].offset;
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
	vertexInputStateCreateInfo.pVertexAttributeDescriptions = pVertexAttributeDescriptions;

	VkAttachmentDescription* pAttachmentDescriptions = NULL;
	TRM_Memory_allocate(sizeof(VkAttachmentDescription) * (info.colorOutputImageCount + 1), (void**)&pAttachmentDescriptions);

	for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < info.colorOutputImageCount; ++colorOutputImageIndex)
	{
		pAttachmentDescriptions[colorOutputImageIndex].flags = 0;
		pAttachmentDescriptions[colorOutputImageIndex].format = TRM_Backend_convertFormat(info.pColorOutputImageFormats[colorOutputImageIndex]);
		pAttachmentDescriptions[colorOutputImageIndex].samples = VK_SAMPLE_COUNT_1_BIT;
		pAttachmentDescriptions[colorOutputImageIndex].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		pAttachmentDescriptions[colorOutputImageIndex].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		pAttachmentDescriptions[colorOutputImageIndex].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		pAttachmentDescriptions[colorOutputImageIndex].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		pAttachmentDescriptions[colorOutputImageIndex].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		pAttachmentDescriptions[colorOutputImageIndex].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}

	pAttachmentDescriptions[info.colorOutputImageCount].flags = 0;
	pAttachmentDescriptions[info.colorOutputImageCount].format = TRM_Backend_convertFormat(info.depthOutputFormat);
	pAttachmentDescriptions[info.colorOutputImageCount].samples = VK_SAMPLE_COUNT_1_BIT;
	pAttachmentDescriptions[info.colorOutputImageCount].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	pAttachmentDescriptions[info.colorOutputImageCount].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	pAttachmentDescriptions[info.colorOutputImageCount].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	pAttachmentDescriptions[info.colorOutputImageCount].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	pAttachmentDescriptions[info.colorOutputImageCount].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	pAttachmentDescriptions[info.colorOutputImageCount].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference* pColorAttachmentReferences = NULL;
	TRM_Memory_allocate(sizeof(VkAttachmentReference) * info.colorOutputImageCount, (void**)&pColorAttachmentReferences);

	for(uint32_t colorOutputImageIndex = 0; colorOutputImageIndex < info.colorOutputImageCount; ++colorOutputImageIndex)
	{
		pColorAttachmentReferences[colorOutputImageIndex].attachment = colorOutputImageIndex;
		pColorAttachmentReferences[colorOutputImageIndex].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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
	subpassDescription.pColorAttachments = pColorAttachmentReferences;
	subpassDescription.pResolveAttachments = NULL;
	subpassDescription.pDepthStencilAttachment = &depthAttachmentReference;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = NULL;

	TRM_Backend_createRenderPass(
		pState->pAllocator,
		pState->device,
		info.colorOutputImageCount + 1,
		pAttachmentDescriptions,
		subpassDescription,
		&pass.info.draw.renderPass);

	TRM_Backend_createGraphicsPipeline(
		pState->pAllocator,
		pState->device,
		vertexShaderModule,
		fragmentShaderModule,
		vertexInputStateCreateInfo,
		info.colorOutputImageCount,
		pass.info.draw.pipelineLayout,
		pass.info.draw.renderPass,
		&pass.info.draw.pipeline);

	vkDestroyShaderModule(pState->device, vertexShaderModule, pState->pAllocator);
	vkDestroyShaderModule(pState->device, fragmentShaderModule, pState->pAllocator);

	TRM_Memory_deallocate(pVertexAttributeDescriptions);
	TRM_Memory_deallocate(pAttachmentDescriptions);
	TRM_Memory_deallocate(pColorAttachmentReferences);

	TRM_Arena_add(&pass, &pState->passPool, pHandle);
}

void TRM_destroyPass(uint32_t handle)
{
	if(vkDeviceWaitIdle(pState->device) != VK_SUCCESS) // this shouldn't be necessary
	{
		exit(EXIT_FAILURE);
	}

	struct TRM_Backend_Pass* pPass = NULL;
	TRM_Arena_get(handle, pState->passPool, (void**)&pPass);
	
	if(pPass->type == TRM_PASS_TYPE_DISPATCH)
	{
		vkDestroyDescriptorSetLayout(pState->device, pPass->info.dispatch.descriptorSetLayout, pState->pAllocator);
		vkDestroyPipelineLayout(pState->device, pPass->info.dispatch.pipelineLayout, pState->pAllocator);
		vkDestroyPipeline(pState->device, pPass->info.dispatch.pipeline, pState->pAllocator);
		TRM_Memory_deallocate(pPass->info.dispatch.pDescriptorInfos);
	}
	else
	{
		vkDestroyDescriptorSetLayout(pState->device, pPass->info.draw.descriptorSetLayout, pState->pAllocator);
		vkDestroyPipelineLayout(pState->device, pPass->info.draw.pipelineLayout, pState->pAllocator);
		vkDestroyPipeline(pState->device, pPass->info.draw.pipeline, pState->pAllocator);
		vkDestroyRenderPass(pState->device, pPass->info.draw.renderPass, pState->pAllocator);
		TRM_Memory_deallocate(pPass->info.draw.pDescriptorInfos);
	}
}
