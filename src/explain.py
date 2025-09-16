import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from captum.attr import LayerGradCam
import torch.nn.functional as F

from .model import ResNet50Classifier


def load_model(ckpt_path: str, num_classes: int, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = ResNet50Classifier(num_classes=num_classes, freeze_until_layer=0)
    model.load_state_dict(ckpt["model_state"])

    model.to(device)
    model.eval()
    return model, cfg


def _preprocess_rgb(rgb_img, device):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = tf(rgb_img).unsqueeze(0).to(device)
    return x


def grad_cam_on_image(model, img_bgr, device="cuda", target_layer_name="layer4.2"):
    # Prepare input
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb_img.astype(np.float32) / 255.0
    inp = _preprocess_rgb(rgb_img, device)

    # Select target conv layer
    target_layer = dict(model.backbone.named_modules())[target_layer_name]

    # Forward to get predicted class
    with torch.no_grad():
        logits = model(inp)
        pred_idx = int(logits.softmax(dim=1).argmax(dim=1).item())

    # Captum LayerGradCam
    lgc = LayerGradCam(model, target_layer)
    attributions = lgc.attribute(inp, target=pred_idx)  # shape: (1, C, H, W)
    # Average across channels if needed (some layers may output multiple channels)
    cam = attributions.mean(dim=1, keepdim=True)  # (1,1,h,w)
    cam = F.relu(cam)
    # Upsample to input size
    cam_up = F.interpolate(cam, size=(rgb_img.shape[0], rgb_img.shape[1]), mode="bilinear", align_corners=False)
    cam_up = cam_up.squeeze().detach().cpu().numpy()
    # Normalize to [0,1]
    cam_up -= cam_up.min()
    if cam_up.max() > 0:
        cam_up /= cam_up.max()

    # Colorize heatmap and overlay
    heatmap = (cam_up * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = (0.4 * heatmap_color[:, :, ::-1] + 0.6 * (rgb_float * 255)).astype(np.uint8)
    return overlay[:, :, ::-1]  # return BGR for cv2.imwrite


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="cam.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load checkpoint first to get num_classes
    ckpt_tmp = torch.load(args.ckpt, map_location=device)
    cfg_tmp = ckpt_tmp.get("cfg", {})
    num_classes = cfg_tmp.get("num_classes", 4)
    # Re-load properly through helper (ensures model is constructed consistently)
    model, _ = load_model(args.ckpt, num_classes=num_classes, device=device)
    img = cv2.imread(args.image)
    cam_vis = grad_cam_on_image(model, img, device=device)
    cv2.imwrite(args.out, cam_vis)
    print(f"Saved Grad-CAM to {args.out}")
