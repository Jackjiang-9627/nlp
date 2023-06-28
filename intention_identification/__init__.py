import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def get_project_absolute_path(path):
    return os.path.abspath(os.path.join(ROOT_DIR, path))