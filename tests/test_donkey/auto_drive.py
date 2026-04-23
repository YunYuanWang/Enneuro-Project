# 导入系统库
import cv2
import numpy as np
import gymnasium as gym
import gym_donkeycar
import paddle
import os

# 导入自定义库
from model import AutoDriveNet


# 设置模拟器环境
env = gym.make("donkey-generated-roads-v0")

# 重置当前场景
obv = env.reset()

# 设置GPU环境
# paddle.set_device("gpu")
paddle.set_device("cpu")

# 加载训练好的模型
model = AutoDriveNet()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(script_dir, "results", "model.pdparams")  # 数据集根目录
checkpoint = paddle.load(model_folder)
model.set_state_dict(checkpoint)
model.eval()

# 开始启动
action = np.array([0, 0.2])  # 动作控制，第1个转向值，第2个油门值

# 执行动作并获取图像
frame, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated

# 运行2500次动作
for t in range(2500):
    # 图像转Tensor
    img = paddle.to_tensor(frame.copy(), stop_gradient=True)
    # 归一化到0~1
    img /= 255.0
    # 调整通道，从HWC调整为CHW
    img = img.transpose([2, 0, 1])
    # 扩充维度，从CHW扩充为NCHW
    img.unsqueeze_(0)
    # 模型推理
    with paddle.no_grad():
        # 前向推理获得预测的转向角度
        prelabel = model(img).squeeze(0).cpu().detach().numpy()
        steering_angle = prelabel[0]
        # 执行动作并重新获取图像
        factor = 1.5  # 动作增强因子
        action = np.array([steering_angle * factor, 0.2])
        frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

# 运行完以后重置当前场景
obv = env.reset()
