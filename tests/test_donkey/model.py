# import paddle.nn as nn
# import paddle


# class AutoDriveNet(nn.Layer):
#     """端到端自动驾驶模型"""
#     def __init__(self):
#         """初始化"""
#         super(AutoDriveNet, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2D(3, 24, 5, stride=2), 
#             nn.ELU(),
#             nn.Conv2D(24, 36, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2D(36, 48, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2D(48, 64, 3),
#             nn.ELU(),
#             nn.Conv2D(64, 64, 3),
#             nn.Dropout(0.5),
#         )
#         self.linear_layers = nn.Sequential(
#             nn.Linear(in_features=64 * 8 * 13, out_features=100),
#             nn.ELU(),
#             nn.Linear(in_features=100, out_features=50),
#             nn.ELU(),
#             nn.Linear(in_features=50, out_features=10),
#             nn.Linear(in_features=10, out_features=1),
#         )

#     def forward(self, input):
#         """前向推理"""
#         input = paddle.reshape(input, [input.shape[0], 3, 120, 160])
#         output = self.conv_layers(input) # 卷积模块
#         output = paddle.reshape(output, [output.shape[0], -1]) # 展平
#         output = self.linear_layers(output) # 线性变换模块
#         return output


# 基于EnNeuro框架的ResNet18自动驾驶模型
from eneuro.nn.module import Module, Conv2d, Linear, BatchNorm, ResidualBlock
from eneuro.base import functions as F


class ResNet18AutoDrive(Module):
    """基于ResNet18架构的端到端自动驾驶模型"""
    def __init__(self, in_channels=3, num_classes=1):
        """初始化"""
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 初始卷积层
        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3)
        self.bn1 = BatchNorm(64)
        self.relu = F.relu
        
        # 残差块组
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 全局平均池化
        self.avg_pool = F.global_average_pooling
        
        # 全连接层
        self.fc = Linear(num_classes)
        
        # 收集参数
        self._collect_params()
    
    def _make_layer(self, out_channels, num_blocks, stride=1):
        """创建残差块层"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = True
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        
        # 创建Sequential容器
        from eneuro.nn.module import Sequential
        return Sequential(*layers)
    
    def _collect_params(self):
        """收集所有参数"""
        self._params = []
        for attr_name in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
            layer = getattr(self, attr_name)
            if hasattr(layer, 'params'):
                for param in layer.params():
                    self._params.append(param)
    
    def forward(self, x):
        """前向推理"""
        # 调整输入形状 (batch, 3, 120, 160)
        if x.shape[1:] != (3, 120, 160):
            x = F.reshape(x, (x.shape[0], 3, 120, 160))
        
        # 初始卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 残差块组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.avg_pool(x)
        
        # 展平
        x = F.flatten(x)
        
        # 全连接层
        x = self.fc(x)
        
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        """返回参数列表"""
        return self._params
    
    def cleargrads(self):
        """清除所有梯度"""
        for param in self.params():
            param.cleargrad()