import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np
import cv2
import io
import base64

# MODEL YÜKLEME
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()

# sınıf isimleri
class_names = weights.meta["categories"]

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# GRAD-CAM
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights_cam = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights_cam * acts).sum(dim=1, keepdim=True)

    cam = torch.relu(cam)
    cam = cam.squeeze().detach().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam

def analyze_uploaded_image(file_bytes: bytes):

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = round(confidence.item() * 100, 2)

    cam = generate_gradcam(model, input_tensor)

    img_np = np.array(image.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_np * 0.6

    _, buffer = cv2.imencode(".jpg", overlay.astype(np.uint8))
    gradcam_base64 = base64.b64encode(buffer).decode()

    explanation = f"Model bu görseli '{label}' olarak sınıflandırdı."

    return {
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "gradcam": gradcam_base64
    }