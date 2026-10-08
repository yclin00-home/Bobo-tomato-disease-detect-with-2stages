# =============================================================
# Step 9 (English version): Two-stage Gradio diagnosis app
#   Stage 1: Is it a tomato leaf?   (tomato_gate.pth)
#   Stage 2: Late blight / other disease / healthy (tomato_resnet18.pth)
# Required files: tomato_resnet18.pth, class_names.json,
#                 tomato_gate.pth, gate_classes.json
# =============================================================
import json
import os

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# ---------- Settings ----------
GATE_THRESHOLD = 0.90     # tomato-leaf probability below this -> reject
DISEASE_MIN_CONF = 0.60   # disease confidence below this -> "uncertain"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

for f in ["class_names.json", "tomato_resnet18.pth", "gate_classes.json", "tomato_gate.pth"]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"❌ File not found: {f}")

# ---------- Stage 2 model (disease classifier) ----------
with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)
num_classes = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("tomato_resnet18.pth", map_location=device))
model = model.to(device).eval()

# ---------- Stage 1 model (tomato leaf or not) ----------
with open("gate_classes.json", "r", encoding="utf-8") as f:
    gate_classes = json.load(f)
TOMATO_IDX = gate_classes.index("tomato_leaf")

gate_model = models.resnet18(weights=None)
gate_model.fc = nn.Linear(gate_model.fc.in_features, 2)
gate_model.load_state_dict(torch.load("tomato_gate.pth", map_location=device))
gate_model = gate_model.to(device).eval()

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------- OpenCV lesion-area ratio (same logic as before) ----------
def calculate_infection_ratio_from_cv(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(img_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_diseased = cv2.inRange(img_hsv, np.array([10, 30, 30]), np.array([35, 255, 255]))

    leaf_mask = cv2.bitwise_or(mask_green, mask_diseased)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    mask_dark = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    raw_lesion_mask = cv2.bitwise_or(mask_diseased, mask_dark)
    lesion_mask = cv2.bitwise_and(raw_lesion_mask, raw_lesion_mask, mask=leaf_mask)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    lesion_pixels = cv2.countNonZero(lesion_mask)
    if total_leaf_pixels == 0:
        return 0.0, None, None

    ratio = lesion_pixels / total_leaf_pixels * 100.0
    return (
        ratio,
        cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(lesion_mask, cv2.COLOR_GRAY2RGB),
    )


# ---------- Two-stage diagnosis ----------
def predict_tomato_disease(input_pil_img):
    if input_pil_img is None:
        return None, "No image", "Please upload or select an image.", None, None

    #input_pil_img = ImageOps.exif_transpose(input_pil_img).convert("RGB")
    input_pil_img = ImageOps.exif_transpose(input_pil_img)
    input_pil_img.thumbnail((1024, 1024))   # 最長邊超過 1024 就等比例縮小
    input_pil_img = input_pil_img.convert("RGB")
    
    x = val_transforms(input_pil_img).unsqueeze(0).to(device)

    # ===== Stage 1: Is it a tomato leaf? =====
    with torch.no_grad():
        p_tomato = torch.softmax(gate_model(x)[0], dim=0)[TOMATO_IDX].item()

    if p_tomato < GATE_THRESHOLD:
        advice = (f"🚫 **This is not a tomato leaf** (tomato-leaf probability only {p_tomato*100:.1f}%). "
                  "Please upload a clear photo of a single tomato leaf.")
        #return {"Not a tomato leaf": 1 - p_tomato, "Tomato leaf": p_tomato}, "N/A", advice, None, None
        return {"Rejected: not confident it is a tomato leaf": 1.0}, "N/A", advice, None, None
        
    # ===== Stage 2: Disease classification =====
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    confidences = {class_names[i]: float(probs[i]) for i in range(num_classes)}
    top_pred = max(confidences, key=confidences.get)
    top_conf = confidences[top_pred]
    name = top_pred.lower()

    if top_conf < DISEASE_MIN_CONF:
        advice = (f"❓ This is a tomato leaf, but the disease result is uncertain "
                  f"(top guess: {top_pred}, {top_conf*100:.1f}%). "
                  "Try another angle or lighting, or consult an expert.")
        return confidences, "N/A", advice, None, None

    if "healthy" in name:
        advice = f"✅ The leaf appears **healthy** (confidence {top_conf*100:.1f}%)."
        return confidences, "N/A", advice, None, None

    if "late" in name and "blight" in name:
        img_bgr = cv2.cvtColor(np.array(input_pil_img), cv2.COLOR_RGB2BGR)
        ratio, leaf_rgb, lesion_rgb = calculate_infection_ratio_from_cv(img_bgr)
        if ratio < 15.0:
            advice = (f"💡 **Recommendation**: Mild late blight infection (lesions cover {ratio:.1f}%). "
                      "Remove infected leaves promptly and apply an organic copper fungicide.")
        elif ratio < 45.0:
            advice = (f"⚠️ **Recommendation**: Moderate late blight infection (lesions cover {ratio:.1f}%). "
                      "Isolate the plant and use a systemic fungicide to control spread.")
        else:
            advice = (f"🚨 **Warning**: Severe late blight infection (lesions cover {ratio:.1f}%)! "
                      "Remove and destroy the plant to prevent spore spread.")
        return confidences, f"{ratio:.2f}%", advice, leaf_rgb, lesion_rgb

    # Other diseases
    advice = (f"⚠️ Disease detected: **{top_pred}** (confidence {top_conf*100:.1f}%). "
              "Please follow the recommended control measures for this disease.")
    return confidences, "N/A", advice, None, None


# ---------- Gradio UI ----------
example_images = [["./tomato-test.jpg"]] if os.path.exists("./tomato-test.jpg") else []

with gr.Blocks(title="Tomato Leaf Disease AI Diagnosis") as demo:
    gr.Markdown("# Tomato Leaf Disease AI Diagnosis & Lesion Area Analysis")
    gr.Markdown("Pipeline: **① Is it a tomato leaf? → ② Late blight / other disease / healthy**")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Select or upload an image",
                                   sources=["upload", "clipboard"])
            if example_images:
                gr.Examples(examples=example_images, inputs=image_input,
                            label="💡 Click an example image to test:")
            btn_submit = gr.Button("🔍 Diagnose", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="AI Prediction & Confidence")
            ratio_output = gr.Textbox(label="Infection Ratio (lesion area / leaf area)")
            advice_output = gr.Markdown(label="Diagnosis & Recommendation")

    with gr.Row():
        leaf_mask_output = gr.Image(label="Leaf Mask", type="numpy")
        lesion_mask_output = gr.Image(label="Lesion Mask", type="numpy")

    btn_submit.click(
        fn=predict_tomato_disease,
        inputs=[image_input],
        outputs=[label_output, ratio_output, advice_output,
                 leaf_mask_output, lesion_mask_output],
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
