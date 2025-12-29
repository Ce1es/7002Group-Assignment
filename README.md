# 🌍 WQF7002 Project: AI-Powered Satellite Wildfire Detection

> **Group Assignment** | **Course:** WQF7002 AI Techniques | **Theme:** SDG 13 - Climate Action

## 📖 项目简介 (Project Overview)
本项目旨在构建一个端到端的 AI 应用程序，利用深度学习技术分析卫星图像，自动检测森林区域是否存在野火 (Wildfire)。

该项目直接响应联合国可持续发展目标 **SDG 13: Climate Action (气候行动)**。通过早期检测野火，我们可以帮助减少因森林燃烧导致的巨大碳排放，保护作为关键“碳汇”的森林生态系统。

## 🎯 核心功能 (Key Features)
* **自动化检测**：输入一张卫星图片，模型能自动判断是 "Wildfire" (野火) 还是 "No Wildfire" (正常森林)。
* **高精度模型**：基于 **ResNet-18** 架构进行迁移学习，测试集准确率达到 **99.35%**。
* [cite_start]**交互式界面**：集成了 **Gradio** Web 界面，支持用户上传图片并实时查看预测结果 [cite: 58]。
* [cite_start]**数据可视化**：包含完整的数据探索性分析 (EDA)，展示了类别分布和样本图像 [cite: 47]。

---

## 📊 数据集 (Dataset)
[cite_start]我们使用了 Kaggle 上的公开数据集 **"Wildfire Prediction Dataset"** [cite: 46]。

* **数据来源**: [Kaggle Link](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
* **数据总量**: 约 42,000 张卫星图像 (350x350px)。
* **类别分布**:
    * 🔥 `wildfire`: 22,710 张 (有火/烟雾)
    * 🌲 `nowildfire`: 20,140 张 (正常森林)
    * *数据极其平衡，无需进行过采样处理。*
* [cite_start]**数据划分**: 遵循 70% 训练 / 15% 测试 / 15% 验证的标准划分 [cite: 49]。

## 🛠️ 技术栈 (Tech Stack)
* **开发环境**: Google Colab (T4 GPU)
* **核心框架**: PyTorch, Torchvision
* [cite_start]**预训练模型**: ResNet-18 (Pre-trained on ImageNet) [cite: 51]
* **前端展示**: Gradio
* **数据处理**: Pandas, Matplotlib, Seaborn, PIL

---

## 🚀 模型表现 (Model Performance)
[cite_start]经过 3 个 Epoch 的微调训练 (Fine-tuning)，模型在测试集 (Test Set, 6300张图片) 上表现优异 [cite: 55]：

| Metric            | Value      | 说明                 |
| :---------------- | :--------- | :------------------- |
| **Accuracy**      | **99.35%** | 极高的识别准确率     |
| **Training Loss** | 0.0416     | 收敛良好             |
| **Test Loss**     | 0.0232     | 泛化能力强，无过拟合 |

---

## 📂 项目结构 (Project Structure)
```text
├── data/                      # 数据集文件夹 (运行时自动下载)
│   ├── train/                 # 训练集
│   ├── test/                  # 测试集
│   └── valid/                 # 验证集
├── wildfire_model.pth         # 训练好的模型权重文件 (可直接加载)
├── app.py                     # Gradio 应用程序代码
├── Group2_Assignment.ipynb    # 完整项目代码 (Jupyter Notebook)
└── README.md                  # 项目说明文档
```

------

## ⚠️ 局限性与讨论 (Limitations & Discussion)

在测试过程中，发现了以下值得注意的现象（可用于作业的 Discussion 部分）1：

1. **域偏移 (Domain Shift)**：
   - 模型在测试集（同源卫星图）上表现完美。
   - 但在识别从 Google 图片搜索下载的“艺术化/高饱和度”航拍图时，可能会出现误报。
   - **原因**：训练数据主要为色彩平淡的科研卫星图，模型可能对高饱和度的颜色或极高清晰度的纹理不敏感。
2. **未来改进**：
   - 引入更多样化的数据源（如 Google Earth 截图）。
   - 在训练中增加色彩抖动 (Color Jittering) 等数据增强手段。

------

## 👨‍💻 如何运行 (How to Run)

**对于组员/Reviewer：**

1. 打开我们的 **Google Colab Notebook** 链接。
2. 点击顶部菜单的 **"Runtime" -> "Run all"**。
   - *步骤 1 会自动下载数据（需 Kaggle API Key）。*
   - *步骤 2 会加载模型。*
   - *最后一步会生成 Gradio 链接。*
3. 点击生成的 `https://xxxx.gradio.live` 链接即可进行 Demo 演示。



## Q&A

### Q: 为什么选择这个主题？和我们的SDG 13 (Climate Action)有关联吗？

### A：

1. 为什么它符合 SDG 13 (Climate Action)？

SDG 13 的核心不仅仅是监测“天气 (Weather)”，而是应对“气候变化 (Climate Change)”。

- **森林是“碳汇” (Carbon Sink)**：树木吸收二氧化碳。如果你砍伐森林 (Deforestation)，地球吸收二氧化碳的能力就变弱了，温室效应加剧。
- **野火是“碳源” (Carbon Source)**：森林燃烧会瞬间向大气释放巨量的二氧化碳，直接导致全球变暖加速。
- **SDG 13 的具体目标**：联合国 SDG 13 的子目标中明确包括“加强对气候相关灾害（如野火）的抵御能力”和“将气候变化措施纳入政策”。

2. 作业文档有提到

在 **Project instructions** 的第 1 步中，作业明确列出了这样一个例子：

> "Clearly define the issue (e.g., predicting disease outbreaks for better healthcare **or analysing satellite images for climate monitoring**) and explain how your solution advances the chosen goal(s)." 

这里提到的 **"analysing satellite images for climate monitoring" (分析卫星图像进行气候监测)** 正是我们选择的方向。且这样的数据集有很多，找到了现成的优质的数据集。



------

### 📅 WQF7002 Group Project - 任务执行计划表 (SDG 13: Wildfire Detection)

#### 第一阶段：数据准备与探索

**目标**：完成数据获取、清洗与 EDA 分析 (占总分 12%：数据6% + 问题6%)。

| **步骤**         | **任务详情 (Action Items)**                                  | **对应作业要求** | **负责人** | **截止时间** |
| ---------------- | ------------------------------------------------------------ | ---------------- | ---------- | ------------ |
| **1.1 数据获取** | 访问 Kaggle/Hugging Face，下载带有标签的卫星图像数据集 (如 Wildfire/No_Fire) 1。 | Gather Data      | 蒋伟       | 1.2          |
| **1.2 数据清洗** | 检查图片完整性，处理坏图；若数据不平衡(如火灾图太少)，需做增强处理 2。 | Clean Dataset    | 蒋伟       | 1.2          |
| **1.3 EDA 分析** | 编写代码生成统计图表（如类别分布柱状图），展示样本图片，撰写分析报告 3。 | EDA              | 蒋伟       | 1.2          |
| **1.4 数据划分** | 将数据按 **70% 训练集 / 30% 测试集** 切分，并记录划分理由 4。 | Split Data       | 蒋伟       | 1.2          |

#### 第二阶段：模型开发与微调 

**目标**：完成核心模型训练与评估 (占总分 14%：模型8% + 评估6%)。

| **步骤**         | **任务详情 (Action Items)**                                  | **对应作业要求** | **负责人** | **截止时间** |
| ---------------- | ------------------------------------------------------------ | ---------------- | ---------- | ------------ |
| **2.1 模型选择** | 选择预训练模型 (如 ResNet-50 或 ViT)，根据任务需求论证选择理由 5。 | Select Model     | 蒋伟       | 1.2          |
| **2.2 模型微调** | 使用 PyTorch/Hugging Face `Trainer` 进行 Fine-tuning，冻结预训练层，训练分类层 6。 | Fine-tune        | 蒋伟       | 1.2          |
| **2.3 评估优化** | 在测试集上计算 Accuracy, F1-Score，生成混淆矩阵 (Confusion Matrix) 7。 | Evaluate         | 蒋伟       | 1.2          |

#### 第三阶段：应用开发与部署

**目标**：构建交互式前端与完成伦理讨论 (占总分 10%：UI 6% + 伦理 4%)。

| **步骤**           | **任务详情 (Action Items)**                                  | **对应作业要求**    | **负责人** | **截止时间** |
| ------------------ | ------------------------------------------------------------ | ------------------- | ---------- | ------------ |
| **3.1 界面开发**   | 使用 **Gradio** 或 Streamlit 开发网页，支持上传图片并显示预测结果 8。 | User Interface      | 蒋伟       | 1.2          |
| **3.2 部署(加分)** | (可选) 将应用部署到 Hugging Face Spaces，生成可访问的在线链接 9。 | Deployment          | 蒋伟       |              |
| **3.3 伦理讨论**   | 撰写关于模型偏差、误报风险及对 SDG 13 实际贡献的讨论文案 10。 | Ethics & Discussion |            |              |

#### 第四阶段：PPT 制作与演练

**目标**：整合所有内容，准备 15 分钟演讲 (占总分 4% + 整体表现)。

| **步骤**         | **任务详情 (Action Items)**                                  | **对应作业要求**    | **负责人** | **截止时间** |
| ---------------- | ------------------------------------------------------------ | ------------------- | ---------- | ------------ |
| **4.1 PPT 制作** | 制作幻灯片，必须包含：问题、数据、模型、结果、Demo、伦理、未来工作。 | Slides Requirements |            |              |
| **4.2 演讲排练** | 全员排练。控制时长：10分钟展示 + 5分钟 Q&A，确保主讲人英语流利。 | Presentation        |            |              |

------

