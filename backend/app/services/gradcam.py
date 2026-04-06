import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]          
        activations = self.activations[0]      

        weights = torch.mean(gradients, dim=(1, 2))  

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        cam = cam.cpu().numpy()
        return cam, class_idx, output

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap_on_image(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    
    heatmap = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap