import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def get_project_absolute_path(path):
    """返回项目的绝对路径"""
    return os.path.abspath(os.path.join(ROOT_DIR, path))
