from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import urllib.request
from app.services.gradcam import GradCAM, overlay_heatmap_on_image
import numpy as np
import cv2
import base64

model = models.mobilenet_v2(pretrained=True)
model.eval()

print(model)
print("TARGET LAYER:", model.features[-1])

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

async def analyze_image(file):
    image = Image.open(file.file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probs, 1)

    confidence_score = float(confidence.item())
    confidence_percent = round(confidence_score * 100, 2)
    predicted_label = classes[predicted.item()]

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    cam, _, _ = gradcam.generate(input_tensor, class_idx=predicted.item())
    gradcam.remove_hooks()

    image_np = np.array(image.resize((224, 224)))
    overlay, heatmap = overlay_heatmap_on_image(image_np, cam)

    _, buffer = cv2.imencode(".jpg", overlay)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    if confidence_score >= 0.80:
        explanation = "Model bu tahminden oldukça emin görünüyor."
    elif confidence_score >= 0.50:
        explanation = "Model bu tahminde orta seviyede emin."
    else:
        explanation = "Model bu tahminden çok emin değil. Görsel net olmayabilir."

    return {
        "label": predicted_label,
        "confidence": confidence_percent,
        "explanation": explanation,
        "gradcam": heatmap_base64,
    }