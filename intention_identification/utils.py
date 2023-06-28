import os

def create_file_dir(file_path):
    create_dir(os.path.dirname(file_path))

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def read_file(files):
    """
    遍历读取给定文件的数据，按行返回数据
    :param files: 路径列表 或者路径字符串
    :return:
    """
    if isinstance(files, str):  # 一个文件
        files = [files]
    for file in files:
        with open(file, 'r', encoding='utf-8') as reader:
            for line in reader:
                line = line.strip()
                yield line
