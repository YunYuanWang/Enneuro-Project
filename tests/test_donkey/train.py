# 导入系统库
import os

# 导入paddle库
import paddle
from visualdl import LogWriter
import paddle.vision.transforms as transforms

# 导入自定义库
from model import AutoDriveNet
from dataset import AutoDriveDataset


# 参数定义
batch_size = 400  # 批大小
start_epoch = 1  # 轮数起始位置
epochs = 100  # 遍历次数
lr = 1e-4  # 学习率
# 设备参数
paddle.set_device("gpu")
# paddle.set_device("cpu")
# 全局记录器
writer = LogWriter()
# 初始化模型
model = AutoDriveNet()
# 初始化优化器
optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
# 定义损失函数
criterion = paddle.nn.MSELoss()
# 定义预处理器
transformations = transforms.Compose(
    [
        transforms.ToTensor(),  # 通道置前并且将0-255值映射至0-1
    ]
)
# 定义数据集类变量
train_dataset = AutoDriveDataset(mode="train", transform=transformations)
# 定义数据集加载器
train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    return_list=True,
)

# 开始逐轮迭代训练
for epoch in range(start_epoch, epochs + 1):
    # 开启训练模式
    model.train()
    # 统计单个epoch的损失函数
    loss_epoch = 0
    # 按批处理
    for i, (imgs, labels) in enumerate(train_loader):
        # 前向传播
        pre_labels = model(imgs)
        # 计算损失
        loss = criterion(pre_labels, labels)
        # 后向传播
        optimizer.clear_grad()
        loss.backward()
        # 更新模型
        optimizer.step()
        # 记录损失值
        loss_epoch += float(loss)
        # # 打印结果
        # print("第 " + str(i) + " 个batch训练结束")

    # 手动释放内存
    del imgs, labels, pre_labels
    # 监控损失值变化
    loss_epoch_avg = loss_epoch / train_dataset.__len__()
    writer.add_scalar("MSE_Loss", loss_epoch_avg, epoch)
    print("epoch:" + str(epoch) +"  MSE_Loss:" + str(loss_epoch_avg))

# 保存模型
script_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(script_dir, "results")  # 数据集根目录
paddle.save(model.state_dict(), save_folder + "/model.pdparams")
# 训练结束关闭监控
writer.close()