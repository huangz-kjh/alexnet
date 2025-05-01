import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 打印所有可用的CUDA设备数量
    device_count = torch.cuda.device_count()
    print(f"有 {device_count} 个可用的CUDA设备")

    # 打印每个CUDA设备的名称和性能信息
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA 设备 {i}: {device_name}")
        device_properties = torch.cuda.get_device_properties(i)
        print(f"设备属性: {device_properties}")

    # 将张量移到CUDA设备上进行计算
    a = torch.tensor([1.0, 2.0, 3.0])
    a = a.cuda()  # 将张量移动到CUDA设备上
    b = torch.tensor([4.0, 5.0, 6.0])
    b = b.cuda()  # 将张量移动到CUDA设备上
    c = a + b  # 在CUDA设备上进行计算
    print(f"计算结果: {c}")

else:
    print("CUDA 不可用")