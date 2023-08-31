#pragma once

#ifndef DISPLAYER_H
#define DISPLAYER_H

#define GLFW_INCLUDE_VULKAN
#include <functional>

#include "GLFW/glfw3.h"

#include<Windows.h>
#include<vulkan/vulkan_win32.h>
#include<vector>

#include "intellisense_cuda.h"

class VkEngine
{
public:
	bool m_isInitialized{ false };
	bool m_framebufferResized{ false };
	int m_numFrames{ 0 };

	VkExtent2D							m_windowExtent{ 1920, 1080 };
	GLFWwindow*							m_window;

	VkInstance							m_instance;
	VkDebugUtilsMessengerEXT			m_debugMessenger;
	VkPhysicalDevice					m_physicalDevice;
	VkDevice							m_logicalDevice;
	VkSurfaceKHR						m_surface;

	VkSwapchainKHR						m_swapchain;
	VkFormat							m_swapchainImageFormat;
	std::vector<VkImage>				m_swapchainImages;
	std::vector<VkImageView>			m_swapchainImageViews;

	VkImage								m_externalImage;
	VkDeviceMemory						m_externalImageMem;
	size_t								m_imageSize;

	VkQueue								m_graphicsQueue;
	uint32_t							m_graphicsQueueFamily;

	VkCommandPool						m_commandPool;
	VkCommandBuffer						m_mainCommandBuffer;

	VkRenderPass						m_renderPass;
	std::vector<VkFramebuffer>			m_framebuffers;

	VkSemaphore							m_presentSemaphore;
	VkSemaphore							m_renderSemaphore;
	VkFence								m_renderFence;

	//cudaStream_t						  m_stream;
	//cudaExternalSemaphore_t			  m_cudaWaitSemaphore;
	//cudaExternalSemaphore_t			  m_cudaSignalSemaphore;
	//cudaExternalSemaphore_t			  m_cudaTimelineSemaphore;
	cudaExternalMemory_t				m_cudaExternalMem;
	unsigned char*						m_cudaFramebufferMem;

	// initializes everything in the engine
	void init();

	// shuts down the engine
	void cleanup();

	// draw loop
	void draw(std::function<void(unsigned char* framebuffer)> renderJob);

	// run main loop
	void run(std::function<void(unsigned char* framebuffer)> renderJob);

private:
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

	void initVulkan();
	void initSwapchain();
	void initCommands();
	void initDefaultRenderPass();
	void initFramebuffers();
	void initSyncStructs();
	void createExternalImage();
	void initCuda();

	VkCommandBuffer beginCmd();
	void endCmd(VkCommandBuffer cmd, VkFence fence);
	void recordImageLayoutTransition(VkImage image, VkCommandBuffer cmd, VkImageLayout oldLayout, VkImageLayout newLayout);
	void importCudaExternalMemory(void** cudaPtr, cudaExternalMemory_t& cudaMem, VkDeviceMemory &vkMem, 
		VkDeviceSize memSize, VkExternalMemoryHandleTypeFlagBits handleType);
	HANDLE getMemHandle(VkDeviceMemory vkMem);
	virtual void render();
};

#endif