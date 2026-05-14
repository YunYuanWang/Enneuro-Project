import os
import random
from tools import creat_data_list, getFileList


# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
org_img_folder = os.path.join(script_dir, "data")  # 数据集根目录
train_ratio = 0.8  # 训练集占比

# 检索jpg文件
jpglist = getFileList(org_img_folder, [], "jpg")
print("检索到 " + str(len(jpglist)) + " 个jpg文件\n")

# 解析转向值
file_list = list()
for jpgpath in jpglist:
    print(jpgpath)
    curDataDir = os.path.dirname(jpgpath)
    basename = os.path.basename(jpgpath)
    angle = (basename[:-4]).split("_")[-1]
    imgPath = os.path.join(curDataDir, basename).replace("\\", "/")
    file_list.append((imgPath, angle))

# 切分数据
random.seed(256)
random.shuffle(file_list)
train_num = int(len(file_list) * train_ratio)
train_list = file_list[0:train_num]
val_list = file_list[train_num:]

# 创建列表文件
creat_data_list(train_list, mode="train")
creat_data_list(val_list, mode="val")