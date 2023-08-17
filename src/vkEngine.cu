#include "vkEngine.h"
#include "vkbootstrap/VkBootstrap.h"
#include "vkInitializers.h"
#include <cmath>
#include <iostream>

#include "intellisense_cuda.h"

__global__ void setWhite(int width, int height, unsigned char* data, float flash)
{
	int tidX = threadIdx.x + blockIdx.x * blockDim.x;
	int tidY = threadIdx.y + blockIdx.y * blockDim.y;

	if (tidX >= width || tidY >= height) return;

	int index = 4 * (tidY * width + tidX);

	auto pixel = (unsigned char)(flash * 255);

	data[index + 0] = pixel;
	data[index + 1] = pixel;
	data[index + 2] = pixel;
	data[index + 3] = 0;
}

#ifdef _WIN64
#include <VersionHelpers.h>
#include <dxgi1_2.h>
#include <aclapi.h>

class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES* operator&();
	~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
	m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
		1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
	if (!m_winPSecurityDescriptor) {
		throw std::runtime_error(
			"Failed to allocate memory for security descriptor");
	}

	PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor +
		SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor,
		SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
		SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
		0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions =
		STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&() {
	return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
	PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor +
		SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	if (*ppSID) {
		FreeSid(*ppSID);
	}
	if (*ppACL) {
		LocalFree(*ppACL);
	}
	free(m_winPSecurityDescriptor);
}
#endif /* _WIN64 */

void VkEngine::init()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_window = glfwCreateWindow(
		m_windowExtent.width,
		m_windowExtent.height,
		"VKEngine",
		nullptr,
		nullptr
	);

	glfwSetWindowUserPointer(m_window, this);
	glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);

	initVulkan();
	initSwapchain();
	initCommands();
	initDefaultRenderPass();
	initFramebuffers();
	initSyncStructs();
	createExternalImage();
	initCuda();

	m_isInitialized = true;
}

void VkEngine::initCuda()
{
	//m_stream = 0;
	m_cudaFramebufferMem = nullptr;

	importCudaExternalMemory((void**)&m_cudaFramebufferMem, m_cudaExternalMem, m_externalImageMem,
		m_imageSize, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
}


void VkEngine::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	auto app = reinterpret_cast<VkEngine*>(glfwGetWindowUserPointer(window));
	app->m_framebufferResized = true;
}

void VkEngine::cleanup()
{
	if ( m_isInitialized )
	{
		vkDeviceWaitIdle(m_logicalDevice);

		vkDestroyCommandPool(m_logicalDevice, m_commandPool, nullptr);

		// destroy sync objs
		vkDestroyFence(m_logicalDevice, m_renderFence, nullptr);
		vkDestroySemaphore(m_logicalDevice, m_presentSemaphore, nullptr);
		vkDestroySemaphore(m_logicalDevice, m_renderSemaphore, nullptr);

		vkDestroySwapchainKHR(m_logicalDevice, m_swapchain, nullptr);

		vkDestroyRenderPass(m_logicalDevice, m_renderPass, nullptr);

		for ( int i = 0; i < m_swapchainImageViews.size(); ++i )
		{
			vkDestroyFramebuffer(m_logicalDevice, m_framebuffers[i], nullptr);

			vkDestroyImageView(m_logicalDevice, m_swapchainImageViews[i], nullptr);
		}

		vkFreeMemory(m_logicalDevice, m_externalImageMem, nullptr);
		vkDestroyImage(m_logicalDevice, m_externalImage, nullptr);

		vkDestroyDevice(m_logicalDevice, nullptr);
		vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
		vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
		vkDestroyInstance(m_instance, nullptr);

		glfwDestroyWindow(m_window);
		glfwTerminate();
	}
}

void VkEngine::draw(std::function<void(unsigned char* framebuffer)> renderJob)
{

	if (vkWaitForFences(m_logicalDevice, 1, &m_renderFence, true, 1000000000) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to fence draw loop!");
	}
	if (vkResetFences(m_logicalDevice, 1, &m_renderFence) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to reset fence!");
	}

	uint32_t swapchainImageIndex;
	vkAcquireNextImageKHR(m_logicalDevice, m_swapchain, 1000000000, m_presentSemaphore, nullptr, &swapchainImageIndex);

	vkResetCommandBuffer(m_mainCommandBuffer, 0);

	auto cmd = m_mainCommandBuffer;

	VkCommandBufferBeginInfo cmdBeginInfo{};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;
	cmdBeginInfo.pInheritanceInfo = nullptr;
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	if ( vkBeginCommandBuffer(cmd, &cmdBeginInfo) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to begin cmd buffer!");
	}

	VkClearValue clearValue;
	clearValue.color = { {0.0f, 0.0f, 0.0f, 1.0f} };

	VkRenderPassBeginInfo rpInfo{};
	rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpInfo.pNext = nullptr;

	rpInfo.renderPass = m_renderPass;
	rpInfo.renderArea.offset.x = 0.0f;
	rpInfo.renderArea.offset.y = 0.0f;
	rpInfo.renderArea.extent = m_windowExtent;
	rpInfo.framebuffer = m_framebuffers[swapchainImageIndex];

	rpInfo.clearValueCount = 1;
	rpInfo.pClearValues = &clearValue;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdEndRenderPass(cmd);

	VkImageCopy copyRegion{};
	copyRegion.extent = { m_windowExtent.width, m_windowExtent.height, 1 };
	copyRegion.dstOffset = { 0, 0, 0 };
	copyRegion.srcOffset = { 0, 0, 0 };
	copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.srcSubresource.baseArrayLayer = 0;
	copyRegion.srcSubresource.layerCount = 1;
	copyRegion.srcSubresource.mipLevel = 0;
	copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.dstSubresource.baseArrayLayer = 0;
	copyRegion.dstSubresource.layerCount = 1;
	copyRegion.dstSubresource.mipLevel = 0;

	// actually do the render job by cuda
	renderJob(m_cudaFramebufferMem);

	recordImageLayoutTransition(m_swapchainImages[swapchainImageIndex], m_mainCommandBuffer, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	vkCmdCopyImage(cmd, m_externalImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_swapchainImages[swapchainImageIndex],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
	recordImageLayoutTransition(m_swapchainImages[swapchainImageIndex], m_mainCommandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStage;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &m_presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &m_renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;


	if ( vkQueueSubmit(m_graphicsQueue, 1, &submit, m_renderFence) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to submit cmd!");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;

	presentInfo.pSwapchains = &m_swapchain;
	presentInfo.swapchainCount = 1;

	presentInfo.pWaitSemaphores = &m_renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;

	presentInfo.pImageIndices = &swapchainImageIndex;

	if ( vkQueuePresentKHR(m_graphicsQueue, &presentInfo) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to present queue!");
	}

	++m_numFrames;
}

void VkEngine::run(std::function<void(unsigned char* framebuffer)> renderJob)
{
	while ( !glfwWindowShouldClose(m_window) )
	{
		glfwPollEvents();
		draw(renderJob);
		std::cout << "1 frame" << std::endl;
	}
}

void VkEngine::initVulkan()
{
	vkb::InstanceBuilder builder;
	auto instRet = builder.set_app_name("TinyPathTracer")
		.request_validation_layers(true)
		.require_api_version(1, 3, 0)
		.use_default_debug_messenger()
		.build();

	vkb::Instance vkbInst = instRet.value();

	m_instance = vkbInst.instance;
	m_debugMessenger = vkbInst.debug_messenger;

	glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);

	std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
	};

	vkb::PhysicalDeviceSelector selector{ vkbInst };
	vkb::PhysicalDevice physicalDevice = selector.set_minimum_version(1, 1)
		.add_desired_extensions(deviceExtensions)
		.set_surface(m_surface)
		.select()
		.value();

	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	vkb::Device vkbDevice = deviceBuilder.build().value();
	m_logicalDevice = vkbDevice.device;
	m_physicalDevice = physicalDevice.physical_device;

	m_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	m_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
}

void VkEngine::initSwapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{ m_physicalDevice, m_logicalDevice, m_surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		.set_desired_extent(m_windowExtent.width, m_windowExtent.height)
		.build()
		.value();

	m_swapchain = vkbSwapchain.swapchain;
	m_swapchainImages = vkbSwapchain.get_images().value();
	m_swapchainImageViews = vkbSwapchain.get_image_views().value();
	m_swapchainImageFormat = vkbSwapchain.image_format;

}

void VkEngine::initCommands()
{
	VkCommandPoolCreateInfo commandPoolCreateInfo = vkinit::commandPoolCreateInfo(m_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	if ( vkCreateCommandPool(m_logicalDevice, &commandPoolCreateInfo, nullptr, &m_commandPool) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to create command pool!");
	}

	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::commandBufferAllocateInfo(m_commandPool, 1);

	if ( vkAllocateCommandBuffers(m_logicalDevice, &cmdAllocInfo, &m_mainCommandBuffer) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to create command buffer!");
	}
}

void VkEngine::initDefaultRenderPass()
{
	VkAttachmentDescription colorAttach{};
	colorAttach.format = m_swapchainImageFormat;
	colorAttach.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachRef{};
	colorAttachRef.attachment = 0;
	colorAttachRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttach;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	if ( vkCreateRenderPass(m_logicalDevice, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to create render pass!");
	}
}

void VkEngine::initFramebuffers()
{
	VkFramebufferCreateInfo fbInfo{};
	fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fbInfo.pNext = nullptr;

	fbInfo.renderPass = m_renderPass;
	fbInfo.attachmentCount = 1;
	fbInfo.width = m_windowExtent.width;
	fbInfo.height = m_windowExtent.height;
	fbInfo.layers = 1;

	const uint32_t swapchainImageCount = m_swapchainImages.size();
	m_framebuffers.resize(swapchainImageCount);

	for ( uint32_t i = 0; i < swapchainImageCount; ++i )
	{
		fbInfo.pAttachments = &m_swapchainImageViews[i];
		if ( vkCreateFramebuffer(m_logicalDevice, &fbInfo, nullptr, &m_framebuffers[i]) != VK_SUCCESS )
		{
			throw std::runtime_error("Failed to create framebuffers!");
		}
	}
}

void VkEngine::initSyncStructs()
{
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.pNext = nullptr;

	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if ( vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &m_renderFence) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to create fence!");
	}

	VkSemaphoreCreateInfo semaphoreCreateInfo{};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = nullptr;
	semaphoreCreateInfo.flags = 0;

	if ( vkCreateSemaphore(m_logicalDevice, &semaphoreCreateInfo, nullptr, &m_presentSemaphore) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to create semaphore!");
	}
	if (vkCreateSemaphore(m_logicalDevice, &semaphoreCreateInfo, nullptr, &m_renderSemaphore) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create semaphore!");
	}
}

VkCommandBuffer VkEngine::beginCmd()
{
	VkCommandBufferAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = 1;
	allocateInfo.commandPool = m_commandPool;

	VkCommandBuffer cmd{};
	if ( vkAllocateCommandBuffers(m_logicalDevice, &allocateInfo, &cmd) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to create cmd");
	}

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	if ( vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to begin cmd");
	}

	return cmd;
}

void VkEngine::endCmd(VkCommandBuffer cmd, VkFence fence)
{
	if ( vkEndCommandBuffer(cmd) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to end cmd");
	}

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;

	if ( vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to submit cmd");
	}
}

void VkEngine::recordImageLayoutTransition(VkImage image, VkCommandBuffer cmd, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}

	vkCmdPipelineBarrier(
		cmd,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);
}

void VkEngine::importCudaExternalMemory(void** cudaPtr, cudaExternalMemory_t& cudaMem, VkDeviceMemory& vkMem, VkDeviceSize memSize, VkExternalMemoryHandleTypeFlagBits handleType)
{
	cudaExternalMemoryHandleDesc externalMemoryHandleDesc{};
	memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));

	externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	externalMemoryHandleDesc.size = memSize;
	externalMemoryHandleDesc.handle.win32.handle = getMemHandle(m_externalImageMem);

	cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc);

	cudaExternalMemoryBufferDesc externalMemoryBufferDesc{};
	externalMemoryBufferDesc.offset = 0;
	externalMemoryBufferDesc.flags = 0;
	externalMemoryBufferDesc.size = memSize;

	cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemoryBufferDesc);
}
 
HANDLE VkEngine::getMemHandle(VkDeviceMemory vkMem)
{
	HANDLE fd = 0;

	VkMemoryGetWin32HandleInfoKHR vkMemoryGetFdInfo{};
	vkMemoryGetFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetFdInfo.pNext = nullptr;
	vkMemoryGetFdInfo.memory = vkMem;
	vkMemoryGetFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	auto fpGetMemoryFdKhr = (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(m_instance, "vkGetMemoryWin32HandleKHR");
	//PFN_vkGetMemoryFdKHR fpGetMemoryFdKhr = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(m_instance, "vkGetMemoryFdKHR");

	if ( fpGetMemoryFdKhr(m_logicalDevice, &vkMemoryGetFdInfo, &fd) != VK_SUCCESS )
	{
		throw std::runtime_error("Failed to get ex mem handle!");
	}

	return fd;
}

void VkEngine::createExternalImage()
{

	VkExternalMemoryImageCreateInfo exImageMemInfo{};
	exImageMemInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	exImageMemInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	exImageMemInfo.pNext = nullptr;

	VkImageCreateInfo exImageInfo{};
	exImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	exImageInfo.pNext = &exImageMemInfo;
	exImageInfo.extent.width = m_windowExtent.width;
	exImageInfo.extent.height = m_windowExtent.height;
	exImageInfo.extent.depth = 1;
	exImageInfo.flags = static_cast<VkSwapchainCreateFlagBitsKHR>(0);
	exImageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	exImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	exImageInfo.mipLevels = 1;
	exImageInfo.arrayLayers = 1;
	exImageInfo.imageType = VK_IMAGE_TYPE_2D;
	exImageInfo.tiling = VK_IMAGE_TILING_LINEAR;
	exImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	exImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	exImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (vkCreateImage(m_logicalDevice, &exImageInfo, nullptr, &m_externalImage) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ex image!");
	}

	VkMemoryRequirements memReq{};
	vkGetImageMemoryRequirements(m_logicalDevice, m_externalImage, &memReq);

	auto findMemoryType = [this](uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type!");
	};

	WindowsSecurityAttributes windowsSecurity;

	VkExportMemoryWin32HandleInfoKHR exportMemoryWin32HandleInfoKhr{};
	exportMemoryWin32HandleInfoKhr.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
	exportMemoryWin32HandleInfoKhr.pNext = nullptr;
	exportMemoryWin32HandleInfoKhr.pAttributes = &windowsSecurity;
	exportMemoryWin32HandleInfoKhr.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	exportMemoryWin32HandleInfoKhr.name = nullptr;

	VkExportMemoryAllocateInfoKHR exportInfo{};
	exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
	exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
	exportInfo.pNext = &exportMemoryWin32HandleInfoKhr;

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	allocInfo.pNext = &exportInfo;

	VkMemoryRequirements vkMemoryRequirements = {};
	vkGetImageMemoryRequirements(m_logicalDevice, m_externalImage, &vkMemoryRequirements);
	m_imageSize = vkMemoryRequirements.size;

	if (vkAllocateMemory(m_logicalDevice, &allocInfo, nullptr, &m_externalImageMem) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate device mem for ex image!");
	}

	if (vkBindImageMemory(m_logicalDevice, m_externalImage, m_externalImageMem, 0) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind device mem & ex image!");
	}

	auto cmd = beginCmd();
	recordImageLayoutTransition(m_externalImage, cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
	endCmd(cmd, VK_NULL_HANDLE);
}

void VkEngine::render()
{
	int width = m_windowExtent.width;
	int height = m_windowExtent.height;
	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15) / 16, (height + 15) / 16);
	float flash = std::abs(std::sin(m_numFrames / 120.f));
	setWhite KERNEL_DIM(gridSize, blockSize) (width, height, m_cudaFramebufferMem, flash);
	cudaDeviceSynchronize();
}
