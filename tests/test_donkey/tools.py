import os


def creat_data_list(file_list, mode="train"):
    """创建txt文件列表"""
    with open((mode + ".txt"), "w") as f:
        for imgpath, angle in file_list:
            f.write(imgpath + " " + str(angle) + "\n")
    print(mode + ".txt 已生成")


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir: 文件夹根目录
    输入 ext: 扩展名
    返回: 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist
