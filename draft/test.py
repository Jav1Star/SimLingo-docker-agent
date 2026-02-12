
def get_nvidia_discrete_gpu_indices():
    """
    获取所有 NVIDIA 独立显卡的原始 Vulkan 编号
    """
    import vulkan as vk
    # 1. 创建最小化的 Vulkan Instance
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="DeviceQuery",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0,
    )

    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info,
    )

    # 创建实例
    instance = vk.vkCreateInstance(create_info, None)
    
    nvidia_indices = []
    NVIDIA_VENDOR_ID = 0x10DE

    try:
        # 2. 获取物理设备列表
        physical_devices = vk.vkEnumeratePhysicalDevices(instance)
        print(physical_devices)

        for index, device in enumerate(physical_devices):
            
            # 3. 获取并判定设备属性
            props = vk.vkGetPhysicalDeviceProperties(device)
            
            is_nvidia = (props.vendorID == NVIDIA_VENDOR_ID)
            is_discrete = (props.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)

            if is_nvidia and is_discrete:
                nvidia_indices.append(index)
    finally:
        # 4. 清理资源：必须销毁 instance
        vk.vkDestroyInstance(instance, None)

    return nvidia_indices
print(get_nvidia_discrete_gpu_indices())