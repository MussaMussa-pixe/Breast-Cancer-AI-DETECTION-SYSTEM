# BreastGuard AI — Breast Cancer Detection System

> **AI-powered histopathology image analysis for Invasive Ductal Carcinoma (IDC) detection using ResNet50 deep learning.**

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [API Endpoints](#api-endpoints)
9. [Model Details](#model-details)
10. [Screenshots](#screenshots)
11. [Disclaimer](#disclaimer)
12. [License](#license)

---

## Overview

BreastGuard AI is a full-stack web application that uses a fine-tuned **ResNet50** convolutional neural network to detect **Invasive Ductal Carcinoma (IDC)** from breast tissue histopathology images. The system provides:

- **Instant AI-powered image analysis** with confidence scores
- **Interactive AI chatbot** for patient education and result interpretation
- **Detailed medical analysis reports** with recommended next steps
- **Responsive, modern dark-themed UI** optimized for desktop and mobile

The model was trained on the **IDC Histopathology Dataset** from Kaggle, containing over **277,000** labeled tissue patches.

---

## Features

### Core Functionality
| Feature | Description |
|---------|-------------|
| **Image Upload** | Drag-and-drop or click-to-browse image upload (JPEG, PNG, BMP, TIFF) |
| **AI Prediction** | ResNet50-based binary classification (IDC Positive / IDC Negative) |
| **Confidence Scoring** | Probability-based confidence percentage with risk level assessment |
| **Medical Report** | Auto-generated printable report with clinical notes and recommendations |
| **AI Chat Assistant** | Context-aware chatbot that answers questions about breast cancer, symptoms, treatments, and interprets analysis results |
| **Health Monitoring** | Real-time backend status indicator |

### UI/UX Features
- 🎨 **Glassmorphism design** with animated gradient blobs
- 📱 **Fully responsive** layout (desktop, tablet, mobile)
- ✨ **Smooth animations** and micro-interactions
- 🖨️ **Print-friendly** report generation
- 🌙 **Dark theme** optimized for medical applications

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **FastAPI** | High-performance async web framework |
| **TensorFlow / Keras** | Deep learning model inference |
| **Pillow (PIL)** | Image preprocessing and manipulation |
| **Uvicorn** | ASGI server |
| **Pydantic** | Data validation and serialization |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Vanilla JavaScript (ES6+)** | Client-side logic |
| **CSS3** | Custom styling with CSS variables and animations |
| **HTML5** | Semantic markup |
| **Google Fonts** | Inter + JetBrains Mono typography |

### AI/ML
| Technology | Purpose |
|------------|---------|
| **ResNet50** | Pre-trained CNN architecture (ImageNet weights) |
| **Transfer Learning** | Fine-tuned on IDC histopathology dataset |
| **Pickle** | Model serialization |

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client (Browser)│     │  FastAPI Server │     │  ResNet50 Model │
│                 │     │                 │     │                 │
│  • Upload Image │────▶│  /predict       │────▶│  • Preprocess   │
│  • View Results │     │  • Validate     │     │  • Infer        │
│  • Chat with AI │◄────│  • Preprocess   │     │  • Return Prob  │
│                 │     │  • Return JSON  │     │                 │
│                 │     │                 │     │                 │
│  • Ask Questions│────▶│  /chat          │     │                 │
│  • Get Replies  │◄────│  • Keyword Match│     │                 │
│                 │     │  • Context Aware│     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Data Flow
1. User uploads a histopathology image via drag-and-drop or file browser
2. Frontend sends image to `/predict` endpoint via multipart form data
3. Backend validates file type and preprocesses image (resize → RGB→BGR → mean subtraction)
4. ResNet50 model performs inference and returns probability
5. Backend classifies result (>0.5 = IDC Positive) and generates report
6. Frontend displays result with animated confidence bar and detailed report
7. User can chat with AI assistant about results or general breast cancer questions

---

## Project Structure

```
breastguard-ai/
├── app.py                  # FastAPI backend application
├── index.html              # Main frontend HTML
├── style.css               # Custom CSS styling
├── script.js               # Frontend JavaScript logic
├── scratch.py              # Utility script for model inspection
├── resnet50_model.pkl      # Trained ResNet50 model (not included in repo)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### File Descriptions

| File | Description |
|------|-------------|
| `app.py` | FastAPI server with CORS, model loading with compatibility patches, image preprocessing, prediction endpoint, and chatbot API |
| `index.html` | Single-page application layout with hero section, upload zone, results panel, about section, and chat interface |
| `style.css` | Complete dark-theme styling with CSS variables, animations, responsive breakpoints, and print media queries |
| `script.js` | Client-side logic for drag-and-drop, image preview, API calls, result rendering, and chatbot interactions |
| `scratch.py` | Standalone script to load and inspect the pickled model (useful for debugging compatibility issues) |

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/breastguard-ai.git
cd breastguard-ai
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pillow>=10.0.0
numpy>=1.24.0
tensorflow>=2.14.0
pydantic>=2.5.0
```

### Step 4: Add the Model File
Place your trained `resnet50_model.pkl` file in the project root directory. The model should be a pickled Keras/TensorFlow model compatible with ResNet50 architecture.

> **Note:** The model file is not included in the repository due to size constraints. You can train your own model using the IDC Histopathology Dataset or request access to a pre-trained model.

### Step 5: Run the Server
```bash
python app.py
```

The server will start on `http://localhost:8000`

### Step 6: Access the Application
Open your browser and navigate to:
```
http://localhost:8000
```

---

## Usage

### Analyzing an Image
1. Navigate to the application homepage
2. Drag and drop a histopathology image into the upload zone, or click "Browse Files"
3. Review the image preview and click **"Analyze Image"**
4. Wait for the AI analysis (typically 1-3 seconds)
5. Review the detailed report including:
   - Diagnosis result (IDC Positive / IDC Negative)
   - Confidence score and risk level
   - Clinical AI notes
   - Recommended next steps

### Using the AI Chat Assistant
1. Scroll down to the **BreastGuard AI Assistant** section
2. Type questions about:
   - Your analysis results (e.g., "What does my result mean?")
   - Breast cancer information (e.g., "What is IDC?", "Symptoms", "Treatment options")
   - Next steps and recommendations
3. Click send or press Enter
4. The assistant provides context-aware responses based on your latest analysis

### Printing Reports
- Click the **Print** button in the report header to generate a printer-friendly version
- The print stylesheet removes UI chrome and optimizes layout for paper

---

## API Endpoints

### Health Check
```
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Image Prediction
```
POST /predict
```
**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "prediction": "IDC Positive",
  "confidence": 94.23,
  "probability": 94.23,
  "risk_level": "High",
  "timestamp": "2024-01-15 09:30:45",
  "filename": "tissue_sample.jpg",
  "model": "ResNet50 (Fine-tuned on IDC Histopathology Dataset)",
  "disclaimer": "This result is for research/educational purposes only..."
}
```

### Chatbot
```
POST /chat
```
**Request Body:**
```json
{
  "message": "What are the symptoms?",
  "prediction": "IDC Positive",
  "confidence": 94.23
}
```

**Response:**
```json
{
  "reply": "Common symptoms of breast cancer include: A lump in the breast..."
}
```

---

## Model Details

### Architecture
- **Base Model:** ResNet50 (Residual Network with 50 layers)
- **Pre-training:** ImageNet (1.2M images, 1000 classes)
- **Fine-tuning:** IDC Histopathology Dataset
- **Input Size:** 50×50 pixel patches (preprocessed to match model requirements)
- **Output:** Binary classification (Sigmoid activation)

### Dataset
- **Source:** Kaggle IDC Histopathology Dataset
- **Total Images:** 277,524 patches
- **IDC Negative (Class 0):** 198,738 patches
- **IDC Positive (Class 1):** 78,786 patches
- **Image Size:** 50×50 pixels

### Preprocessing Pipeline
1. Convert to RGB (3 channels)
2. Resize to 50×50 pixels
3. Convert RGB → BGR (OpenCV style)
4. Subtract ImageNet mean values: `[103.939, 116.779, 123.68]`
5. Add batch dimension: `(1, 50, 50, 3)`

### Classification Threshold
- **Probability > 0.5** → IDC Positive
- **Probability ≤ 0.5** → IDC Negative

### Risk Levels
| Confidence | Risk Level |
|------------|------------|
| > 85% | High |
| 50-85% | Moderate |
| < 50% | Low |

---

## Screenshots

> *Screenshots will be added here showing the upload interface, analysis results, and chat assistant.*

---

## Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This application is for **research and educational purposes only**. It does **NOT** provide medical diagnosis, treatment recommendations, or professional medical advice.

- AI predictions are screening tools, not clinical diagnoses
- Always consult a qualified healthcare provider for:
  - Proper diagnosis
  - Treatment decisions
  - Medical guidance
- Do not use this tool as a substitute for professional medical care
- If you have concerns about breast cancer, please see a doctor immediately

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset:** [IDC Histopathology Dataset on Kaggle](https://www.kaggle.com/datasets)
- **Model Architecture:** [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) | [TensorFlow](https://www.tensorflow.org/) | [Keras](https://keras.io/)

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainer.

---

<p align="center">
  <sub>Built with ❤️ for advancing AI in healthcare education.</sub>
</p>
