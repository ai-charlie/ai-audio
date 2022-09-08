# 每一个git项目的准备工作
## 0. Previous
```bash
apt-get install git
ssh-keygen -t rsa #三次回车

```
### 复制`id_rsa.pub`文件内容到github
登录github -> Accounting settings图标 -> SSH key -> Add SSH key -> 填写SSH key的名称（可以起一个自己容易区分的），然后将拷贝的 .ssh/id\_rsa.pub文件内容粘帖 -> add key 按钮添加。

## 1. 在github创建一个仓库 project_name

## 2. 本地创建相同名字的仓库 project_name
撰写代码
```bash
git add . # 提交所有变动文件
git commit "first commit"
git config --local user.name "ai-cahrlie"
git config --local user.email "ai-charlie@outlook.com"
git remote add origin git@github.com:ai-charlie/ai-audio.git
git branch -M main
git push -u origin main
```

## 3. 每一次提交
```bash
git pull # 拉取
git add . # 提交所有变动文件
git commit "XXX commit"
git push -u origin main
```
