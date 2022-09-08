# Setup Docker

## Install
[https://www.docker.com/](https://www.docker.com/)

## Docker hub
[https://hub.docker.com/](https://hub.docker.com/)
## Comman Usage

```bash
# 查看、删除容器
docker ps [-a] [| grep keyword_to_search]
docker rm container_name/id
# 查看、删除镜像
docker images [| grep keyword_to_search]
docker rmi container_name/id
# 数据复制 
docker cp /path/of/data  container_name/id:/new/path/of/data
```