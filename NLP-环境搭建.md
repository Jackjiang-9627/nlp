# NLP编程环境搭建

## 一、搭建Python开发环境

### 1、下载Anaconda开发工具或者PyCharm+miniconda

### 2、conda创建虚拟环境


| 操作             | 命令                                                                           | 注释                                              |
| ------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------- |
| 创建虚拟环境     | conda create --name d2l python=3.9 -y                                          | 安装在user目录                                    |
| 创建虚拟环境     | conda create --prefix=D:/miniconda/envs/d2l python=3.9                         | 安装在指定路径(推荐)                              |
| 激活环境         | conda activate d2l                                                             | d2l为环境名称                                     |
| 退出当前环境     | conda deactivate                                                               |                                                   |
| 查看当前环境信息 | conda info                                                                     | 当前环境的详细信息                                |
| 显示所有环境     | conda info --envs                                                              | 环境名称+位置                                     |
| 删除环境         | conda remove -n env_name --all                                                 | 删除环境                                          |
| 环境重命名       | conda create -n d2l2 --clone d2l(或者环境的路径)<br/>conda remove -n d2l --all | 把环境 d2l 重命名成 d2l2<br/>其实就是先复制再删除 |

### 3、更改镜像源


| 操作                   | 命令                                                     | 注释        |
| ------------------------ | ---------------------------------------------------------- | ------------- |
| 删除并恢复默认源       | conda config --remove-key channels                       |             |
| 添加指定源             | conda config --add channels *                            | *指代镜像源 |
| 删除指定源             | conda config --remove channels *                         | *指代镜像源 |
| 下载时显示使用的镜像源 | conda config --set show_channel_urls yes                 |             |
| 中科大镜像             | https://pypi.mirrors.ustc.edu.cn/simple                  |             |
| 清华镜像               | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ |             |
| 豆瓣镜像               | http://pypi.douban.com/simple/                           |             |
| 阿里镜像               | https://mirrors.aliyun.com/pypi/simple/                  |             |
| 百度镜像               | https://mirror.baidu.com/pypi/simple                     |             |

## 二、搭建PyTorch（GPU）开发环境

1. 打开conda prompt（管理员）进入conda需要安装的环境 --> conda activate env_name；
2. 打开NVIDIA控制面板，点击 帮助——系统信息——组件——3D设置--查看NVCUDA64.DLL的产品名称里的cuda版本号（命令行输入nvidia-smi命令进行查看）；
3. 登录[PyTorch官网](https://pytorch.org/)查询自己能够下载的PyTorch版本，高版本的cuda是可以兼容低版本的cuda的，复制下载指令到步骤1；
4. [英伟达官网](https://developer.nvidia.com/cuda-toolkit-archive)下载对应的CUDA版本。我下载的版本是CUDA11.6（PyTorch非必要，TensorFlow需要）；
5. 安装([torchtext · PyPI](https://pypi.org/project/torchtext/0.14.0/))，新建一个虚拟环境来搭建需要torchtext的低版本,如果pytorch版本高于torchtext对应的pytorch版本，直接用pip install torchtext，会将pytorch更改为cpu版本。比如：
   * pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
   * pip install torchtext==0.13.0
6. 验证：
   ```python
   import torch
   torch.__version__
   torch.version.cuda
   torch.cuda.is_availabel()
   ```

## 三、搭建DGL环境

1. 与本地PyTorch的CUDA版本一致，我这里均为cuda11.6。
2. 如何查看本地Pytorch的cuda版本？pip list 查找 或者`torch.__version__`
3. pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html
4. pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

## 四、安装Transformers库

1. `pip install transformers`
2. 检查`transformers.__version__`

## 五、NLP常用工具

1. 自然语言处理工具包——SpaCy
   `pip install spacy`
2. 中文分词工具——Jieba
   `pip install jieba`
3. 中文转拼音工具——Pypinyin
   `pip install pypinyin`
4. 评估翻译质量的算法库——SacreBLEU
   `pip install sacrebleu`
