# 🧠 Image Classification App with DIP, Enhancements, Ollama, and PDF Export

This is a **Streamlit-based web application** that allows users to upload images, apply digital image processing (DIP) filters, enhance images, classify them using **OpenCLIP**, generate AI-powered explanations via **Ollama (LLaMA 3.2)**, and export the results as a PDF report.

---

## 🚀 Features

1. **Image Upload**
   - Supports PNG, JPG, and JPEG formats.

2. **Digital Image Processing (DIP) Filters**
   - Apply filters such as:
     - Grayscale
     - Edge Detection
     - Gaussian Blur
     - Sharpen
     - Histogram Equalization

3. **Image Enhancement**
   - Adjust:
     - Brightness
     - Contrast
     - Sharpness
     - Color (Saturation)

4. **Image Classification**
   - Uses **OpenCLIP ViT-L-14-quickgelu** for feature extraction and multi-class classification.
   - Predicts from a wide variety of categories (objects, animals, electronics, landscapes, etc.).
   - Displays confidence scores for predictions.

5. **AI-Powered Explanation**
   - Fetches natural language explanations from **Ollama (LLaMA 3.2)**.
   - Describes visual patterns, shapes, and distinguishing features.

6. **PDF Export**
   - Generates a downloadable PDF report containing:
     - Processed image
     - Predicted class and confidence
     - Ollama-generated explanation

---

## 🛠 Technology Stack

- **Frontend:** Streamlit  
- **Deep Learning:** PyTorch, OpenCLIP  
- **Image Processing:** OpenCV, Pillow  
- **AI Explanations:** Ollama (LLaMA 3.2)  
- **PDF Generation:** ReportLab  
- **Containerization (optional):** Docker  

---

## ⚡ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/image-classification-dip.git
   cd image-classification-dip
