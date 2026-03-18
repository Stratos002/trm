#include "GLFW/glfw3.h"

#include <stdio.h>
#include <stdint.h>

#include "trm_memory.h"
#include "trm_renderer.h"

#define TRM_WINDOW_WIDTH 1000
#define TRM_WINDOW_HEIGHT 1000

int main()
{
	if (!glfwInit())
	{
		return 1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* pWindow = glfwCreateWindow(TRM_WINDOW_WIDTH, TRM_WINDOW_HEIGHT, "torment", NULL, NULL);
	if (pWindow == NULL)
	{
		return 1;
	}

	TRM_Memory_start();
	TRM_Renderer_start(pWindow, TRM_WINDOW_WIDTH, TRM_WINDOW_HEIGHT);

	while (!glfwWindowShouldClose(pWindow))
	{
		glfwPollEvents();
		TRM_Renderer_render();
	}

	TRM_Renderer_terminate();
	TRM_Memory_terminate();

	glfwDestroyWindow(pWindow);
	glfwTerminate();

	return 0;
}