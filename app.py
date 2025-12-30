import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import torch.nn.functional as F

# 1. 配置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 重新加载模型架构
def load_model():
    # 加载 ResNet18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # 加载我们训练好的权重 (wildfire_model.pth)
    # map_location确保即使没有GPU也能在CPU上运行
    model.load_state_dict(torch.load("wildfire_model.pth", map_location=device))
    model = model.to(device)
    model.eval() # 设为评估模式
    return model

# 加载模型 (这一步需要 wildfire_model.pth 文件在同一个目录下)
try:
    model = load_model()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed. Please check if the wildfire_model.pth file has been uploaded.\nError message: {e}")
    model = None

# 4. 定义预测函数
class_names = ['nowildfire', 'wildfire']

def predict(image):
    if model is None:
        return {"Error": "Model not found"}
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
    
    return {class_names[i]: float(probs[0][i]) for i in range(2)}

# 5. 启动 Gradio 界面
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Satellite Image"),
    outputs=gr.Label(num_top_classes=2),
    title="🛰️ AI Wildfire Detection System",
    description="Upload a satellite image to detect wildfire risks. (SDG 13: Climate Action)",
    examples=None 
)

if __name__ == "__main__":
    iface.launch()