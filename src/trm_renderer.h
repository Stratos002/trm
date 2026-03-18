#ifndef TRM_RENDERER
#define TRM_RENDERER

#include "vulkan/vulkan.h"
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#include <stdint.h>

void TRM_Renderer_start(GLFWwindow* pWindow, uint32_t windowWidth, uint32_t windowHeight);

void TRM_Renderer_terminate();

void TRM_Renderer_render();

#endif