# 基于波动方程约束的地震正反演算法
---

## 概述
我们使用在PINNs的基础上进行修改使其能运用到波动方程的地震正反演上。

## 安装
本仓库需要安装tensorflow用于实现PINNs

我们推荐一个新的工作环境，如下所式：
```bash
conda create -n Wavepinns python=3.6 # Use Anaconda package manager
conda activate Wavepinns
```
然后安装以下python依赖包：
```bash
pip install --ignore-installed --upgrade [packageURL]# install tensorflow (get packageURL from https://www.tensorflow.org/install/pip, see tensorflow website for details)
pip install tensorflow==1.15
pip install matplotlib
pip install pyDOE
```
以及画图需要的texlive环境

## WavePINNs工作流程
首先介绍各个文件夹的用途：
*  ``
