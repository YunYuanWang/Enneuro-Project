#!/usr/bin/env python3
"""
测试基于ResNet18架构的自动驾驶模型
"""

import sys
from pathlib import Path

# 添加eneuro包的路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'code'))

import numpy as np
from model import ResNet18AutoDrive


def test_resnet_autodrive():
    """测试ResNet18AutoDrive模型"""
    print("开始测试ResNet18AutoDrive模型...")
    
    # 创建模型实例
    model = ResNet18AutoDrive()
    print("模型创建成功")
    
    # 打印模型信息
    print("\n模型结构:")
    print(f"输入通道: {model.in_channels}")
    print(f"输出类别: {model.num_classes}")
    
    # 统计参数数量
    params = list(model.params())
    total_params = 0
    for param in params:
        if hasattr(param, 'data') and param.data is not None:
            total_params += np.prod(param.data.shape)
    print(f"\n模型总参数量: {total_params:,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    
    # 创建随机输入 (batch_size=1, channels=3, height=120, width=160)
    batch_size = 1
    input_data = np.random.randn(batch_size, 3, 120, 160).astype(np.float32)
    print(f"输入形状: {input_data.shape}")
    
    # 执行前向传播
    output = model(input_data)
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output}")
    
    # 测试不同形状的输入
    print("\n测试不同形状的输入...")
    input_data_flat = np.random.randn(batch_size, 3 * 120 * 160).astype(np.float32)
    print(f"扁平输入形状: {input_data_flat.shape}")
    output_flat = model(input_data_flat)
    print(f"输出形状: {output_flat.shape}")
    print(f"输出值: {output_flat}")
    
    # 测试批量输入
    print("\n测试批量输入...")
    batch_size = 4
    input_batch = np.random.randn(batch_size, 3, 120, 160).astype(np.float32)
    print(f"批量输入形状: {input_batch.shape}")
    output_batch = model(input_batch)
    print(f"批量输出形状: {output_batch.shape}")
    
    print("\n测试完成! 模型工作正常。")


if __name__ == "__main__":
    test_resnet_autodrive()
