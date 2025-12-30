# 🌍 WQF7002 Group Project: AI-Powered Satellite Wildfire Detection

> **Course:** WQF7002 AI Techniques | **Theme:** SDG 13 - Climate Action
> **Group:** Group 2

## 🔗 Live Demo (Deployment)
We have successfully deployed our application on **Hugging Face Spaces**. You can test the model interactively via the link below:
👉 **[Click Here to Try the App](https://huggingface.co/spaces/CeIeste/wildfire-detection-sdg13)**

[cite_start]*(This feature addresses the Bonus Deployment requirement )*

---

## 📖 1. Problem Statement & SDG Alignment
**Target SDG:** **Goal 13: Climate Action**

Wildfires are not just local disasters; they are massive carbon sources that accelerate global warming. [cite_start]This project aims to build an end-to-end AI application that contributes to SDG 13 by enabling **early detection of wildfires** through satellite imagery[cite: 3, 41].

* **Objective:** To automate the classification of satellite images into "Wildfire" or "No Wildfire" zones.
* [cite_start]**Impact:** Early identification helps preserve forests (carbon sinks) and reduces greenhouse gas emissions caused by biomass burning[cite: 44].

## 📊 2. Data Preparation & EDA
[cite_start]We selected a public dataset from **Kaggle** suitable for binary classification tasks[cite: 46].

* **Dataset Source:** [Wildfire Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
* **Data Volume:** ~42,000 satellite images (350x350px).
* **Class Balance:**
    * 🔥 `wildfire`: 22,710 images
    * 🌲 `nowildfire`: 20,140 images
    * *Note: The dataset is well-balanced, requiring no oversampling techniques.*
* [cite_start]**Data Splitting:** We followed a standard ratio of **70% Training, 15% Validation, and 15% Testing** to ensure robust evaluation[cite: 49].
* [cite_start]**EDA:** Exploratory Data Analysis was performed to verify class distribution and visual integrity[cite: 47].

## 🧠 3. Model Development
We utilized **Transfer Learning** to achieve high accuracy with computational efficiency.

* [cite_start]**Architecture:** **ResNet-18** (Pre-trained on ImageNet)[cite: 51].
* **Justification:** ResNet-18 captures complex visual features (like smoke textures) while remaining lightweight enough for rapid inference.
* [cite_start]**Fine-tuning Strategy:** We froze the initial feature extraction layers and fine-tuned the fully connected layers on our wildfire dataset[cite: 53].
* **Frameworks:** PyTorch, Torchvision.

## 📈 4. Evaluation Results
The model was evaluated on a held-out test set of 6,300 images. [cite_start]The performance metrics indicate excellent generalization[cite: 55].

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **99.35%** | Exceptionally high reliability on the test set. |
| **Training Loss** | 0.0416 | Indicates successful convergence. |
| **Test Loss** | 0.0232 | Lower than training loss, suggesting no overfitting. |

## 💻 5. User Interface (UI)
[cite_start]We built an interactive web interface using **Gradio**[cite: 58].
* **Features:** Allows users to upload satellite images and receive real-time predictions with confidence scores.
* **Accessibility:** The app is hosted permanently on Hugging Face Spaces for public access.

## ⚠️ 6. Discussion & Limitations
[cite_start]While the model performs remarkably well on the test dataset, our real-world testing revealed important insights regarding **Domain Shift**[cite: 61]:

1.  **Observation**: The model achieved ~99% accuracy on the Kaggle test set (scientific satellite imagery). However, when testing with "artistic" or high-saturation aerial photos found on Google Images, the model showed a higher False Positive rate.
2.  **Analysis**: This is a classic case of **Domain Shift**. The model was trained on specific sensor data (likely specific color calibration) and struggles to generalize to processed photography with different color distributions.
3.  [cite_start]**Future Work**: To address this, future iterations would include **Data Augmentation** (color jittering) and incorporating diverse data sources (e.g., Google Earth screenshots) to improve robustness[cite: 63].

---

## 📂 Project Structure
```text
├── data/                      # Dataset folder (downloaded via script)
│   ├── train/                 # Training set
│   ├── test/                  # Test set
│   └── valid/                 # Validation set
├── wildfire_model.pth         # Trained model weights
├── app.py                     # Gradio application code
├── requirements.txt           # Dependencies for deployment
├── Group2_Assignment.ipynb    # Full training notebook (EDA + Training)
└── README.md                  # Project documentation
