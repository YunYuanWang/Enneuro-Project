import paddle
import paddle.vision.transforms as transforms
from dataset import AutoDriveDataset

# 定义预处理操作
transformations = transforms.Compose([transforms.ToTensor()])
# 创建数据集实例
dataset = AutoDriveDataset(mode="train", transform=transformations)
total_num = dataset.__len__()
print("数据集图片总数量:", total_num)
# 创建数据集加载器
batch_size = 16
loader = paddle.io.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    return_list=True,
)
# 计算遍历1次完整数据集需要加载的次数(total_num/batch_size)
print("加载次数:", len(loader))