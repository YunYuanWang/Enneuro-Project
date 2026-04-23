import numpy as np
import cv2
import paddle


class AutoDriveDataset(paddle.io.Dataset):
    """数据集加载器"""

    def __init__(self, mode, transform=None):
        """
        :参数 mode: 'train' 或者 'val'
        :参数 transform: 图像预处理方式
        """
        self.mode = mode.lower()
        self.transform = transform
        assert self.mode in {"train", "val"}
        # 读取数据集列表文件信息
        if self.mode == "train":
            file_path = "./train.txt"
        else:
            file_path = "./val.txt"
        self.file_list = list()
        with open(file_path, "r") as f:
            files = f.readlines()
            for file in files:
                if file.strip() is None:
                    continue
                self.file_list.append([file.split(" ")[0], float(file.split(" ")[1])])

    def __getitem__(self, i):
        """
        :参数 i: 图像检索号
        :返回: 返回第i个图像和转向值
        """
        # 读取图像
        img = cv2.imread(self.file_list[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 预处理
        if self.transform:
            img = self.transform(img)
        # 读取转向值
        label = self.file_list[i][1]
        label = paddle.to_tensor(np.array([label]), dtype="float32")
        return img, label

    def __len__(self):
        """返回: 图像总数"""
        return len(self.file_list)