# Pytorch envs build

## cuda11.3
### 官方镜像
1. pytorch官方镜像
```bash
docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
```

2. NVIDIA 官方镜像
```bash
docker pull nvcr.io/nvidia/pytorch:21.10-py3 
```

因以下命令将映射当前文件夹到容器内为`workspace`，需切换到项目文件夹后运行命令。
```bash
docker run  --gpus all -p 8889:8889 --shm-size=8gb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ --rm -it nvcr.io/nvidia/pytorch:21.10-py3
```
> `--rm`参数会让容器在停止后自动删除


同时运行jupyter-lab
```bash
docker run  --gpus all -p 8889:8889 --shm-size=8gb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ --rm -it pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel jupyter-lab --port=8889 --no-browser --ip 0.0.0.0 --allow-root
```


## cuda10.2

```bash
docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel          
```
