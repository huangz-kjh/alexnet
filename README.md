# 🧠 AlexNet Implementation

> **Reproduction and Analysis of AlexNet for Image Classification**

---

## 🚀 项目简介

本项目基于经典深度学习模型 AlexNet 实现图像分类任务，并对其结构、训练策略及性能进行复现与分析。

AlexNet 是深度学习发展史上的里程碑模型，在 ImageNet Large Scale Visual Recognition Challenge 2012 中取得突破性成绩，推动了 CNN 在计算机视觉领域的广泛应用。

---

## 🎯 项目目标

* ✅ 从零实现 AlexNet 网络结构
* ✅ 理解 CNN 在图像分类中的工作机制
* ✅ 复现经典模型训练流程
* ✅ 分析模型性能与改进空间

---

## 💡 为什么选择 AlexNet？

* 📌 第一个在大规模数据上成功的深度CNN
* 📌 引入 ReLU，解决梯度消失问题
* 📌 使用 Dropout，缓解过拟合
* 📌 使用 GPU 训练（历史性突破）

> **AlexNet = 现代深度学习视觉模型的起点**

---

## 🏗️ 模型结构（Architecture）

AlexNet 由 5 层卷积层 + 3 层全连接层组成：

```text
Input (224x224x3)
   ↓
Conv1 + ReLU + MaxPool
   ↓
Conv2 + ReLU + MaxPool
   ↓
Conv3 + ReLU
   ↓
Conv4 + ReLU
   ↓
Conv5 + ReLU + MaxPool
   ↓
FC6 + Dropout
   ↓
FC7 + Dropout
   ↓
FC8 (Softmax)
```

<img width="1770" height="721" alt="image" src="https://github.com/user-attachments/assets/9b84463e-051a-4735-9b26-4ce2d226225b" />

---

## 🔍 核心技术点（Key Insights）

### 🔹 1. ReLU 激活函数

相比 Sigmoid / Tanh：

* 收敛更快
* 缓解梯度消失

---

### 🔹 2. Dropout

* 随机丢弃神经元
* 防止过拟合

---

### 🔹 3. 数据增强（Data Augmentation）

* 随机裁剪
* 翻转
* RGB扰动

👉 提升泛化能力

---

### 🔹 4. 局部响应归一化（LRN）

* 模拟生物神经抑制机制
* 增强特征表达（现在较少使用）

---

## 📊 实验结果（Results）

| 模型      | Accuracy |
| ------- | -------- |
| AlexNet | 78.5%    |

---

## 🖼️ 可视化（Visualization）

```md
## 🔍 Prediction Examples

<img width="865" height="435" alt="image" src="https://github.com/user-attachments/assets/c46ceeac-1770-44b7-b298-b3b19aa3bd59" />

---

## 📦 项目结构（Project Structure）

```bash
AlexNet/
├── models/
│   └── alexnet.py
├── data/
├── utils/
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

---

## ⚡ 快速开始（Quick Start）

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 训练模型

```bash
python train.py
```

### 3️⃣ 测试模型

```bash
python test.py
```
