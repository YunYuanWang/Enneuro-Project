# EnNeuro GPU适配更新文档

## 1. 概述

本文档描述了EnNeuro框架的GPU（CUDA）适配实现，包括训练使用方法和新模块开发规范。

### 1.1 核心概念

- **numpy**: CPU计算后端
- **cupy**: GPU计算后端（CUDA加速）
- **混合运算问题**: 当模型在GPU上时，输入数据可能在CPU上，需要正确处理数组类型转换

## 2. 训练使用方法

### 2.1 启用GPU训练

```python
# 在训练脚本中，将模型移动到GPU
model = YourModel()
model.to('cuda')  # 启用GPU训练 也可以 model = model.to(device)

# 训练时会自动使用GPU进行计算
trainer.fit(train_loader, val_loader, epochs=10)
```

或者你可以直接在fit中传入device参数，例如：

```python
    trainer.fit(
        train_loader, 
        test_loader, 
        epochs=1,
        batch_size=batch_size,
        verbose=True,
        device='cuda'
    )
```

### 2.2 依赖要求

```bash
# 安装cupy（GPU支持）
pip install cupy-cuda12x  # 根据你的CUDA版本选择合适的包
```

### 2.3 训练输出示例

```
开始训练...
using cuda to train
======================= Epoch #1/1 - Start training =======================
Epoch   1 |█████████████████████████████░|  99.9%  | loss=0.0426 | acc=0.888
>>>>>>>>>>> Epoch loss: 0.1019 - Epoch acc: 0.9727
Time cost: 41.67 seconds

训练完成!
测试准确率: 0.9727 (97.27%)
测试损失: 0.1019
```

## 3. 核心辅助函数

### 3.1 get_array_module

获取数组对应的计算模块（numpy或cupy）。

```python
def get_array_module(arr):
    """获取数组对应的计算模块 (numpy 或 cupy)"""
    if isinstance(arr, Tensor):
        return get_array_module(arr.data)
    if isinstance(arr, np.ndarray):
        return np
    elif has_cupy and isinstance(arr, cp.ndarray):
        return cp
    else:
        return np
```

### 3.2 to_xp

将数组转换为指定模块（numpy或cupy）对应的类型。

```python
def to_xp(arr, xp):
    """
    将数组arr转换为xp对应的类型。

    Args:
        arr: 输入数组（可以是numpy数组、cupy数组或Tensor）
        xp: 目标数组模块（numpy或cupy）

    Returns:
        转换后的数组

    Notes:
        - 当目标是numpy但arr是cupy数组时，使用cp.asnumpy()转换
        - 当目标是cupy但arr是numpy数组时，使用cp.asarray()转换
        - 如果arr已经是目标类型，则不进行转换
    """
    if isinstance(arr, Tensor):
        arr = arr.data
    if xp is np:
        if has_cupy and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    else:  # xp is cp
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr
```

## 4. 新模块开发规范

### 4.1 前向传播（forward）

**原则**: 优先从权重获取数组模块类型，使用`to_xp`转换输入。

```python
class YourFunction(Function):
    def forward(self, *xs):
        """
        你的函数前向传播。

        Args:
            x: 输入数据
            W: 权重
            b: 偏置（可选）

        Returns:
            输出数据

        GPU适配说明:
            - 优先从权重W获取数组模块类型
            - 当W是cupy数组时，需要先将x转换为cupy数组
        """
        x = xs[0]
        W = xs[1]
        b = xs[2] if len(xs) > 2 else None

        # 1. 优先从权重获取数组模块
        W_data = W.data if isinstance(W, Tensor) else W
        xp = get_array_module(W_data)

        # 2. 使用to_xp转换输入
        x_data = to_xp(x, xp)

        # 3. 执行计算（所有操作使用xp）
        y = xp.dot(x_data, W_data)

        # 4. 如果使用b，同样需要转换
        if b is not None:
            b_data = to_xp(b, xp)
            y = y + b_data

        return y
```

### 4.2 反向传播（backward）

**原则**: 正确转换所有涉及的数组到同一模块。

```python
    def backward(self, gys):
        """反向传播"""
        # 获取输入
        x = self.inputs[0]
        W = self.inputs[1]

        # 获取数组模块
        xp = get_array_module(gys)

        # 转换所有数组
        x_data = to_xp(x, xp)
        W_data = to_xp(W, xp)
        gy_data = to_xp(gys, xp)

        # 执行计算
        gx = ...  # 使用xp进行计算
        gW = ...  # 使用xp进行计算

        return gx, gW
```

### 4.3 优化器（Optimizer）

**原则**: 处理标量和数组的混合运算，确保类型一致。

```python
class YourOptimizer(Optimizer):
    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            xp = get_array_module(param.data)

            # 将标量转换为cupy数组（如果需要）
            if xp is not np:
                learning_rate = xp.asarray(learning_rate)
                momentum = xp.asarray(momentum)
                one_minus_m = xp.asarray(1) - momentum
            else:
                one_minus_m = 1 - momentum

            # 执行计算
            ...
```

## 5. 常见问题及解决方案

### 5.1 问题：numpy和cupy数组混合运算

**错误信息**:
```
TypeError: Unsupported type <class 'numpy.ndarray'>
```

**原因**: 在cupy数组运算中使用了numpy标量

**解决方案**:
```python
# 错误写法
_v = _v * 0.9 + 0.1 * grad  # 当_v是cupy时失败

# 正确写法
if xp is not np:
    _v = _v * xp.asarray(0.9) + xp.asarray(0.1) * grad
else:
    _v = _v * 0.9 + 0.1 * grad
```

### 5.2 问题：sklearn等库期望numpy数组

**错误信息**:
```
TypeError: Implicit conversion to a NumPy array is not allowed.
```

**原因**: sklearn收到cupy数组但期望numpy数组

**解决方案**:
```python
# 将cupy数组转换为numpy
if hasattr(your_array, 'get'):
    numpy_array = your_array.get()
else:
    numpy_array = your_array
```

### 5.3 问题：变量名冲突

**错误信息**: 形状计算错误或维度不匹配

**原因**: 重复使用变量名（如`W`既表示权重又表示宽度）

**解决方案**:
```python
# 错误写法
W = xs[1]  # 权重
N, C, H, W = x.shape  # W被覆盖为宽度

# 正确写法
W = xs[1]  # 权重
N, C, H, W_in = x.shape  # 使用不同的变量名
```

## 6. 已适配的模块列表

### 6.1 函数模块 (eneuro/base/functions.py)

| 函数 | 说明 |
|------|------|
| `MatMul.forward` | 矩阵乘法 |
| `Linear.forward` | 线性变换 |
| `im2col_array` | 图像转列矩阵 |
| `im2col_conv2d_forward` | im2col卷积前向 |
| `gemm_conv2d_forward` | GEMM卷积前向 |
| `Deconv2d.forward` | 转置卷积 |
| `GroupedConv2d.forward` | 分组卷积 |
| `FusedConvReLU.forward` | 融合卷积+ReLU |
| `FusedConvBNReLU.forward` | 融合卷积+BN+ReLU |
| `Conv2DGradW.forward` | 卷积核梯度计算 |
| `ReLU.backward` | ReLU反向传播 |

### 6.2 优化器 (eneuro/nn/optim.py)

| 优化器 | 说明 |
|--------|------|
| `SGD.step` | 随机梯度下降 |
| `MomentumSGD.step` | 动量SGD |
| `Adam.step` | Adam优化器 |

### 6.3 损失函数 (eneuro/nn/loss.py)

| 损失函数 | 说明 |
|----------|------|
| `MSELoss.forward` | 均方误差损失 |
| `SoftmaxWithLoss.forward` | Softmax交叉熵损失 |
| `SigmoidWithLoss.forward` | Sigmoid交叉熵损失 |
| `CrossEntropyLoss.forward` | 交叉熵损失 |

### 6.4 其他模块

| 模块 | 说明 |
|------|------|
| `eneuro/utils/visualization.py` | 可视化模块 |

## 7. 开发检查清单

新增模块时，确保满足以下要求：

- [ ] 优先从权重获取`xp`类型
- [ ] 使用`to_xp()`转换所有输入
- [ ] 标量运算时考虑`xp is not np`的情况
- [ ] 反向传播时正确转换所有数组
- [ ] 处理Tensor对象的`.data`属性
- [ ] 避免变量名冲突
- [ ] 测试时同时支持CPU和GPU模式

## 8. 性能对比

使用GPU训练MNIST分类任务（LeNet模型）的性能提升：

| 模式 | 每个epoch时间 | 加速比 |
|------|-------------|--------|
| CPU | ~3000秒 | 1x |
| GPU (CUDA) | ~41秒 | ~73x |

## 9. 技术支持

如遇到问题，请检查：

1. cupy是否正确安装：`python -c "import cupy; print(cupy.cuda.is_available())"`
2. CUDA版本是否匹配：`python -c "import cupy; print(cupy.cuda.device.get_compute_capability())"`
3. 模型是否正确移动到GPU：`model.to('cuda')`
