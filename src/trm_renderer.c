#include "trm_renderer.h"
#include "trm_memory.h"

#define VOLK_IMPLEMENTATION
#include "volk.h"

#include <stdlib.h>
#include <stdio.h>

struct TRM_Renderer_FrameInFlight
{
	VkCommandBuffer commandBuffer;
	VkFence commandBufferExecutedFence;
	VkSemaphore imageAvailableSemaphore;
	uint32_t imageIndex;
	VkDescriptorSet descriptorSet;
};

struct TRM_Renderer_SwapchainImage
{
	VkImage image;
	VkImageView imageView;
	VkSemaphore imageRenderedSemaphore;
	VkBool32 transitionned;
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
	uint32_t swapchainImageCount;
	struct TRM_Renderer_SwapchainImage* pSwapchainImages;
	struct TRM_Renderer_FrameInFlight* pFramesInFlight;
	uint32_t frameIndex;

	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout computePipelineLayout;
	VkPipeline computePipeline;
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
	VkFormat format,
	uint32_t windowWidth,
	uint32_t windowHeight,
	uint32_t queueFamilyIndex,
	VkSwapchainKHR* pSwapchain)
{
	VkSurfaceCapabilitiesKHR surfaceCapabilities = {0};
	if(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
	if(surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount)
	{
		imageCount = surfaceCapabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR swapchainCreateInfo = {0};
	swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainCreateInfo.pNext = NULL;
	swapchainCreateInfo.flags = 0;
	swapchainCreateInfo.surface = surface;
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = format; // we should query this
	swapchainCreateInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; // we should query this
	swapchainCreateInfo.imageExtent.width = windowWidth;
	swapchainCreateInfo.imageExtent.height = windowHeight;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainCreateInfo.queueFamilyIndexCount = 1;
	swapchainCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
	swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainCreateInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

	if(vkCreateSwapchainKHR(device, &swapchainCreateInfo, pAllocator, pSwapchain) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}
}

static void TRM_Renderer_createImageView(
	const VkAllocationCallbacks* pAllocator,
	VkDevice device,
	VkImage image,
	VkFormat format,
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
	imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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

	VkComputePipelineCreateInfo computPipelineCreateInfo = {0};
	computPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computPipelineCreateInfo.pNext = VK_NULL_HANDLE;
	computPipelineCreateInfo.flags = 0;
	computPipelineCreateInfo.stage = shaderStageCreateInfo;
	computPipelineCreateInfo.layout = pipelineLayout;
	computPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	computPipelineCreateInfo.basePipelineIndex = 0;

	if(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computPipelineCreateInfo, pAllocator, pComputePipeline) != VK_SUCCESS)
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
	*pSize = ftell(pFile);
	rewind(pFile);

	TRM_Memory_allocate(sizeof(uint32_t) * (*pSize), ppCode);

	fread(*ppCode, 1, *pSize, pFile);

	fclose(pFile);
}

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

	VkFormat swapchainFormat = VK_FORMAT_R8G8B8A8_UNORM; // we should query this
	TRM_Renderer_createSwapchain(
		pState->pAllocator,
		pState->physicalDevice,
		pState->device,
		pState->surface,
		swapchainFormat,
		windowWidth,
		windowHeight,
		pState->queueFamilyIndex,
		&pState->swapchain);

	VkImage* pSwapchainImages = NULL;
	vkGetSwapchainImagesKHR(pState->device, pState->swapchain, &pState->swapchainImageCount, NULL);
	TRM_Memory_allocate(sizeof(VkImage) * pState->swapchainImageCount, (void**)&pSwapchainImages);
	vkGetSwapchainImagesKHR(pState->device, pState->swapchain, &pState->swapchainImageCount, pSwapchainImages);

	TRM_Memory_allocate(sizeof(struct TRM_Renderer_SwapchainImage) * pState->swapchainImageCount, (void**)&pState->pSwapchainImages);

	for(uint32_t swapchainImageIndex = 0; swapchainImageIndex < pState->swapchainImageCount; ++swapchainImageIndex)
	{
		pState->pSwapchainImages[swapchainImageIndex].image = pSwapchainImages[swapchainImageIndex];

		TRM_Renderer_createImageView(
			pState->pAllocator,
			pState->device,
			pState->pSwapchainImages[swapchainImageIndex].image,
			swapchainFormat,
			&pState->pSwapchainImages[swapchainImageIndex].imageView);

		TRM_Renderer_createSemaphore(pState->pAllocator, pState->device, &pState->pSwapchainImages[swapchainImageIndex].imageRenderedSemaphore);
		pState->pSwapchainImages[swapchainImageIndex].transitionned = VK_FALSE;
	}

	TRM_Memory_deallocate(pSwapchainImages);

	TRM_Memory_allocate(sizeof(struct TRM_Renderer_FrameInFlight) * pState->swapchainImageCount, (void**)&pState->pFramesInFlight);

	for(uint32_t frameInFlightIndex = 0; frameInFlightIndex < pState->swapchainImageCount; ++frameInFlightIndex)
	{
		TRM_Renderer_allocateCommandBuffer(pState->commandPool, pState->device, &pState->pFramesInFlight[frameInFlightIndex].commandBuffer);
		TRM_Renderer_createFence(pState->pAllocator, pState->device, &pState->pFramesInFlight[frameInFlightIndex].commandBufferExecutedFence);
		TRM_Renderer_createSemaphore(pState->pAllocator, pState->device, &pState->pFramesInFlight[frameInFlightIndex].imageAvailableSemaphore);
		pState->pFramesInFlight[frameInFlightIndex].imageIndex = UINT32_MAX;
	}

	pState->frameIndex = 0;

	// === temporary ===

	VkDescriptorSetLayoutBinding binding = {0};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	binding.pImmutableSamplers = NULL;
	
	TRM_Renderer_createDescriptorSetLayout(pState->pAllocator, pState->device, 1, &binding, &pState->descriptorSetLayout);

	for(uint32_t frameInFlightIndex = 0; frameInFlightIndex < pState->swapchainImageCount; ++frameInFlightIndex)
	{
		TRM_Renderer_allocateDescriptorSet(pState->device, pState->descriptorPool, pState->descriptorSetLayout, &pState->pFramesInFlight[frameInFlightIndex].descriptorSet);
	}
	
	VkShaderModule computeShaderModule;
	uint32_t codeSize = 0;
	uint32_t* pCode = NULL;
	TRM_Renderer_readShader("../../../assets/shaders/compute.spv", &codeSize, &pCode); // very bad
	TRM_Renderer_createShaderModule(pState->pAllocator, pState->device, codeSize, pCode, &computeShaderModule);
	
	TRM_Renderer_createPipelineLayout(pState->pAllocator, pState->device, 1, &pState->descriptorSetLayout, &pState->computePipelineLayout);
	TRM_Renderer_createComputePipeline(pState->pAllocator, pState->device, computeShaderModule, pState->computePipelineLayout, &pState->computePipeline);
	
	vkDestroyShaderModule(pState->device, computeShaderModule, pState->pAllocator);
	TRM_Memory_deallocate(pCode);
}

void TRM_Renderer_terminate(void)
{
	if(pState != NULL)
	{
		if(vkDeviceWaitIdle(pState->device) != VK_SUCCESS)
		{
			exit(EXIT_FAILURE);
		}

		vkDestroyPipeline(pState->device, pState->computePipeline, pState->pAllocator);
		vkDestroyPipelineLayout(pState->device, pState->computePipelineLayout, pState->pAllocator);

		for(uint32_t frameInFlightIndex = 0; frameInFlightIndex < pState->swapchainImageCount; ++frameInFlightIndex)
		{
			vkDestroySemaphore(pState->device, pState->pFramesInFlight[frameInFlightIndex].imageAvailableSemaphore, pState->pAllocator);
			vkDestroyFence(pState->device, pState->pFramesInFlight[frameInFlightIndex].commandBufferExecutedFence, pState->pAllocator);
		}

		vkDestroyDescriptorSetLayout(pState->device, pState->descriptorSetLayout, pState->pAllocator);
		
		TRM_Memory_deallocate(pState->pFramesInFlight);

		for(uint32_t swapchainImageIndex = 0; swapchainImageIndex < pState->swapchainImageCount; ++swapchainImageIndex)
		{
			vkDestroySemaphore(pState->device, pState->pSwapchainImages[swapchainImageIndex].imageRenderedSemaphore, pState->pAllocator);
			vkDestroyImageView(pState->device, pState->pSwapchainImages[swapchainImageIndex].imageView, pState->pAllocator);
		}

		TRM_Memory_deallocate(pState->pSwapchainImages);
		
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
	if(vkWaitForFences(pState->device, 1, &pState->pFramesInFlight[pState->frameIndex].commandBufferExecutedFence, VK_FALSE, UINT64_MAX) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	if(vkResetFences(pState->device, 1, &pState->pFramesInFlight[pState->frameIndex].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	if(pState->pFramesInFlight[pState->frameIndex].imageIndex != UINT32_MAX)
	{
		pState->pSwapchainImages[pState->pFramesInFlight[pState->frameIndex].imageIndex].transitionned = VK_TRUE;
	}

	uint32_t imageIndex = 0;
	if(vkAcquireNextImageKHR(
		pState->device, 
		pState->swapchain, 
		UINT64_MAX, 
		pState->pFramesInFlight[pState->frameIndex].imageAvailableSemaphore, 
		VK_NULL_HANDLE, 
		&imageIndex) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	pState->pFramesInFlight[pState->frameIndex].imageIndex = imageIndex;

	VkDescriptorImageInfo imageInfo = {0};
	imageInfo.sampler = VK_NULL_HANDLE;
	imageInfo.imageView = pState->pSwapchainImages[imageIndex].imageView;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet descriptorWrite = {0};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.pNext = NULL;
	descriptorWrite.dstSet = pState->pFramesInFlight[pState->frameIndex].descriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	descriptorWrite.pImageInfo = &imageInfo;
	descriptorWrite.pBufferInfo = NULL;
	descriptorWrite.pTexelBufferView = NULL;

	vkUpdateDescriptorSets(pState->device, 1, &descriptorWrite, 0, NULL);

	vkResetCommandBuffer(pState->pFramesInFlight[pState->frameIndex].commandBuffer, 0);

	VkCommandBufferBeginInfo commandBufferBeginInfo = {0};
	commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	commandBufferBeginInfo.pNext = NULL;
	commandBufferBeginInfo.flags = 0;
	commandBufferBeginInfo.pInheritanceInfo = NULL;

	if(vkBeginCommandBuffer(pState->pFramesInFlight[pState->frameIndex].commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	// src present -> transfer opt
	{
		VkImageMemoryBarrier imageMemoryBarrier = {0};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.pNext = NULL;
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		imageMemoryBarrier.oldLayout = pState->pSwapchainImages[imageIndex].transitionned ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.image = pState->pSwapchainImages[imageIndex].image;
		imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
		imageMemoryBarrier.subresourceRange.levelCount = 1;
		imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
		imageMemoryBarrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(
			pState->pFramesInFlight[pState->frameIndex].commandBuffer, 
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
			0, 
			0, 
			NULL, 
			0, 
			NULL, 
			1, 
			&imageMemoryBarrier);
	}

	vkCmdBindDescriptorSets(pState->pFramesInFlight[pState->frameIndex].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pState->computePipelineLayout, 0, 1, &pState->pFramesInFlight[pState->frameIndex].descriptorSet, 0, NULL);
	vkCmdBindPipeline(pState->pFramesInFlight[pState->frameIndex].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pState->computePipeline);
	vkCmdDispatch(pState->pFramesInFlight[pState->frameIndex].commandBuffer, 500 / 8, 500 / 8, 1);

	// transfer dst -> present src
	{
		VkImageMemoryBarrier imageMemoryBarrier = {0};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.pNext = NULL;
		imageMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = 0;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		imageMemoryBarrier.image = pState->pSwapchainImages[imageIndex].image;
		imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
		imageMemoryBarrier.subresourceRange.levelCount = 1;
		imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
		imageMemoryBarrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(
			pState->pFramesInFlight[pState->frameIndex].commandBuffer, 
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 
			0, 
			0, 
			NULL, 
			0, 
			NULL, 
			1, 
			&imageMemoryBarrier);
	}

	if(vkEndCommandBuffer(pState->pFramesInFlight[pState->frameIndex].commandBuffer) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

	VkSubmitInfo submitInfo = {0};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pNext = NULL;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &pState->pFramesInFlight[pState->frameIndex].imageAvailableSemaphore;
	submitInfo.pWaitDstStageMask = &waitDstStageMask;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &pState->pFramesInFlight[pState->frameIndex].commandBuffer;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &pState->pSwapchainImages[imageIndex].imageRenderedSemaphore;

	if(vkQueueSubmit(pState->queue, 1, &submitInfo, pState->pFramesInFlight[pState->frameIndex].commandBufferExecutedFence) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	VkPresentInfoKHR presentInfo = {0};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = NULL;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &pState->pSwapchainImages[imageIndex].imageRenderedSemaphore;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &pState->swapchain;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = NULL;

	if(vkQueuePresentKHR(pState->queue, &presentInfo) != VK_SUCCESS)
	{
		exit(EXIT_FAILURE);
	}

	pState->frameIndex = (pState->frameIndex + 1) % pState->swapchainImageCount;
}
