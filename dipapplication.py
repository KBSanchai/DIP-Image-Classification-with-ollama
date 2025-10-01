import streamlit as st
import torch
import open_clip
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import requests
import ollama
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit, ImageReader
import os
import uuid
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Load OpenCLIP Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model = model.to(device).eval()

# --- Label List (trimmed for brevity) ---
labels = [
    "airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear", "bed", "bench", "bicycle",
    "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake", "car", "carrot", "cat", "cell phone", "chair",
    "clock", "couch", "cow", "cup", "dining table", "dog", "donut", "elephant", "fire hydrant", "fork", "frisbee", "giraffe",
    "hair drier", "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop", "microwave", "motorcycle", "mouse",
    "orange", "oven", "parking meter", "person", "pizza", "potted plant", "refrigerator", "remote", "sandwich", "scissors",
    "sheep", "sink", "skateboard", "skis", "snowboard", "spoon", "sports ball", "stop sign", "suitcase", "surfboard", "teddy bear",
    "tennis racket", "tie", "toaster", "toilet", "toothbrush", "traffic light", "train", "truck", "tv", "umbrella", "vase",
    "wine glass", "zebra", "mountain", "tree", "river", "flower", "sky", "sun", "moon", "cloud", "rainbow", "desert", "forest",
    "ocean", "city", "building", "bridge", "tower", "castle", "temple", "church", "mosque", "statue", "monument", "fountain",
    "garden", "park", "zoo", "aquarium", "harbor", "beach", "island", "volcano", "waterfall", "cave", "glacier","Smartphone",
    "Laptop", "Tablet", "Smartwatch", "Headphones", "Speaker", "Camera", "Drone", "VR Headset", "Game Console", "Router","wolf","television",
    "monitor", "printer", "scanner", "projector", "keyboard", "mouse", "webcam", "microphone", "charger", "power bank" 
]
text_inputs = tokenizer(labels).to(device)

# --- Ollama Explanation ---
def fetch_ollama_response(image_class):
    try:
        prompt = (
            f"The uploaded image was classified as '{image_class}'. "
            f"Explain the reasoning behind this classification by identifying visual features, patterns, colors, or shapes "
            f"that are typically associated with a '{image_class}'. "
            f"Also mention how such features may help distinguish it from similar categories."
        )
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "No response received.")
    except Exception as e:
        return f"Error fetching response from Ollama: {e}"

# --- PDF Export ---
def generate_pdf(prediction, explanation, image):
    filename = f"report_{uuid.uuid4().hex}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, f"Prediction: {prediction}")

    # Save and insert image
    img_path = f"/tmp/img_{uuid.uuid4().hex}.png"
    image.save(img_path)
    c.drawImage(ImageReader(img_path), 72, 520, width=200, height=200, preserveAspectRatio=True)

    # Wrap explanation text
    wrapped_text = simpleSplit(explanation, 'Helvetica', 12, 460)
    y = 500
    for line in wrapped_text:
        c.drawString(72, y, line)
        y -= 15
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750

    c.save()
    os.remove(img_path)
    return filename


# --- Streamlit UI ---
st.title("🧠 Image Classification with DIP + Enhancement + Ollama + PDF Export")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # --- DIP Filter ---
    st.subheader("🧪 Apply Digital Image Processing Filter")
    dip_filter = st.selectbox("Choose a DIP filter", [
        "None", "Grayscale", "Edge Detection", "Gaussian Blur", "Sharpen", "Histogram Equalization"
    ])

    # Convert PIL to OpenCV
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Apply DIP filter
    if dip_filter == "Grayscale":
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
    elif dip_filter == "Edge Detection":
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        image_cv = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif dip_filter == "Gaussian Blur":
        image_cv = cv2.GaussianBlur(image_cv, (7, 7), 0)
    elif dip_filter == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image_cv = cv2.filter2D(image_cv, -1, kernel)
    elif dip_filter == "Histogram Equalization":
        img_yuv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        image_cv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Back to PIL
    image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # --- Image Enhancement ---
    st.subheader("🎨 Image Enhancement")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
    color = st.slider("Color (Saturation)", 0.5, 2.0, 1.0, 0.1)

    # Apply enhancements
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    image = ImageEnhance.Color(image).enhance(color)

    st.image(image, caption=f"Processed Image ({dip_filter} + Enhancement)", use_container_width=True)

    # --- Classification ---
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_inputs)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        predicted_label = labels[pred_idx]
        st.success(f"🎯 Prediction: **{predicted_label}** (Confidence: {probs[pred_idx]:.2f})")

    # --- Explanation ---
    with st.spinner("🧠 Generating explanation..."):
        explanation = fetch_ollama_response(predicted_label)
        st.markdown("**Explanation:**")
        st.write(explanation)

    # --- Export to PDF ---
    if st.button("📄 Export Report to PDF"):
        pdf_path = generate_pdf(predicted_label, explanation, image)
        with open(pdf_path, "rb") as file:
            st.download_button("⬇️ Download PDF", file, file_name="classification_report.pdf")

# cd "C:\Users\KBsan\OneDrive\Desktop\DIP Project"
# py -3.10 -m streamlit run dipapplication.py
