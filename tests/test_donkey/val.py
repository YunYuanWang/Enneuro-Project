# 导入系统库
import time
import os

# 导入Paddle库
import paddle
import paddle.vision.transforms as transforms

# 导入自定义库
from dataset import AutoDriveDataset
from model import AutoDriveNet


# 定义设备运行环境
paddle.set_device("gpu")
# 加载训练好的模型文件
model = AutoDriveNet()
script_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(script_dir, "results", "model.pdparams")  # 数据集根目录
checkpoint = paddle.load(model_folder)
model.set_state_dict(checkpoint)
# 定义预处理器
transformations = transforms.Compose(
    [
        transforms.ToTensor(),  # 通道置前并且将0-255RGB值映射至0-1
    ]
)
# 创建验证数据集类实例
val_dataset = AutoDriveDataset(mode="val", transform=transformations)
# 创建数据集加载器
val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_size=400,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    return_list=True,
)
# 定义评估指标
criterion = paddle.nn.MSELoss()
# 记录均方误差值
MSEs = 0
nbatch = 0
# 记录测试时间
model.eval()
start = time.time()
with paddle.no_grad():
    # 逐批样本进行推理计算
    for i, (imgs, labels) in enumerate(val_loader):
        # 前向传播
        pre_labels = model(imgs)
        # 计算误差
        loss = criterion(pre_labels, labels)
        MSEs += loss.numpy()
        nbatch += 1
# 输出平均均方误差
print("MSE: " + ("%f" % (MSEs / nbatch)))
print("平均单张样本用时  {:.3f} 秒".format((time.time() - start) / len(val_dataset)))