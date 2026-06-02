import os
import io
import sys
import pickle
import numpy as np
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Force UTF-8 output so emoji/unicode prints don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─── Keras Compatibility Patch ────────────────────────────────────────────────
# The pkl was saved with a Keras build that includes 'quantization_config' in
# Dense config. Patch Dense.__init__ directly so it silently drops unknown args.
try:
    import keras.src.layers.core.dense as _kd_mod

    _OrigDenseInit = _kd_mod.Dense.__init__

    def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
        _OrigDenseInit(self, *args, **kwargs)

    _kd_mod.Dense.__init__ = _patched_dense_init

    # Also patch from_config to strip the key before it reaches __init__
    _OrigDenseFromConfig = _kd_mod.Dense.from_config.__func__

    @classmethod
    def _patched_from_config(cls, config):
        config.pop('quantization_config', None)
        return _OrigDenseFromConfig(cls, config)

    _kd_mod.Dense.from_config = _patched_from_config
    print('[OK] Keras Dense compatibility patch applied.')
except Exception as _patch_err:
    print(f'[WARN] Keras compat patch skipped: {_patch_err}')

# ─── App Init ────────────────────────────────────────────────────────────────
app = FastAPI(title="Breast Cancer Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet50_model.pkl")

model = None

def load_model():
    global model
    # Strategy 1: plain pickle (works when Keras versions match)
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print('[OK] Model loaded via pickle.')
        return
    except Exception as e1:
        print(f'[WARN] pickle load failed: {e1}')

    # Strategy 2: try keras custom_objects with compat Dense
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense as _OrigDense

        class _CompatDense2(_OrigDense):
            @classmethod
            def from_config(cls, config):
                config.pop('quantization_config', None)
                return super().from_config(config)

        with tf.keras.utils.custom_object_scope({'Dense': _CompatDense2}):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        print('[OK] Model loaded via custom_object_scope.')
        return
    except Exception as e2:
        print(f'[WARN] custom_object_scope load failed: {e2}')

    # Strategy 3: tf.keras.models.load_model (if saved as H5/SavedModel inside pkl)
    try:
        import tensorflow as tf
        import tempfile, shutil
        with open(MODEL_PATH, 'rb') as f:
            raw = pickle.load(f)
        if hasattr(raw, 'predict'):
            model = raw
            print('[OK] Model object extracted from pickle.')
            return
    except Exception as e3:
        print(f'[WARN] fallback strategy failed: {e3}')

    print('[ERROR] All model loading strategies failed.')
    model = None

load_model()

# ─── Image Preprocessing ─────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for ResNet50 inference."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((50, 50))
        img_array = np.array(img, dtype=np.float32)
        # ResNet50 preprocess_input (ImageNet mean subtraction, BGR)
        img_array = img_array[..., ::-1]  # RGB -> BGR
        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        img_array = img_array - mean
        img_array = np.expand_dims(img_array, axis=0)  # (1, 50, 50, 3)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# ─── Pydantic Models ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    prediction: str | None = None
    confidence: float | None = None

class ChatResponse(BaseModel):
    reply: str

# ─── Chat Knowledge Base ─────────────────────────────────────────────────────
CANCER_QA = {
    "what is idc": "IDC stands for **Invasive Ductal Carcinoma** — the most common type of breast cancer, accounting for about 80% of all breast cancer diagnoses. It begins in the milk ducts and invades surrounding breast tissue.",
    "what is invasive ductal carcinoma": "**Invasive Ductal Carcinoma (IDC)** is a cancer that starts in the milk ducts of the breast and grows into surrounding breast tissue. It's called 'invasive' because it has broken through the duct wall. It is the most prevalent form of breast cancer.",
    "symptoms": "Common symptoms of breast cancer include:\n• A lump in the breast or underarm\n• Change in breast size or shape\n• Skin dimpling or puckering\n• Nipple discharge (other than breast milk)\n• Redness or flaky skin near the nipple\n• Persistent breast pain\n\n⚠️ Always consult a doctor if you notice any of these symptoms.",
    "treatment": "Treatment options for IDC breast cancer include:\n• **Surgery**: Lumpectomy (removal of tumor) or mastectomy (removal of breast)\n• **Radiation Therapy**: High-energy rays to destroy remaining cancer cells\n• **Chemotherapy**: Drugs to kill cancer cells throughout the body\n• **Hormone Therapy**: For hormone receptor-positive cancers\n• **Targeted Therapy**: Drugs targeting specific cancer cell proteins\n• **Immunotherapy**: Boosting the immune system to fight cancer\n\nTreatment plans are personalized — consult your oncologist.",
    "stages": "Breast cancer has 5 stages:\n• **Stage 0**: Non-invasive (DCIS)\n• **Stage I**: Small tumor, no lymph node spread\n• **Stage II**: Larger tumor or limited lymph node involvement\n• **Stage III**: Larger tumor with more lymph node involvement\n• **Stage IV**: Cancer has spread to distant organs (metastatic)\n\nEarly detection (Stage I/II) has much better outcomes.",
    "survival rate": "Breast cancer survival rates (5-year relative survival):\n• Stage I: ~99%\n• Stage II: ~86%\n• Stage III: ~72%\n• Stage IV: ~28%\n\nEarly detection significantly improves survival outcomes. Regular screening is crucial.",
    "prevention": "Ways to reduce breast cancer risk:\n• Maintain a healthy weight\n• Exercise regularly (150+ min/week)\n• Limit alcohol consumption\n• Avoid smoking\n• Breastfeed if possible\n• Regular screenings and mammograms\n• Know your family history\n• Discuss hormone therapy risks with your doctor",
    "screening": "Breast cancer screening methods:\n• **Mammogram**: X-ray of breast tissue (recommended annually for women 40+)\n• **Clinical Breast Exam**: Physical exam by a healthcare provider\n• **Self-Breast Exam**: Monthly self-check for unusual changes\n• **MRI**: For high-risk individuals\n• **Ultrasound**: Used alongside mammograms for denser breast tissue\n• **Biopsy**: Definitive diagnosis tool",
    "risk factors": "Breast cancer risk factors include:\n• Gender (females at much higher risk)\n• Age (risk increases with age)\n• Family history of breast cancer\n• BRCA1/BRCA2 gene mutations\n• Dense breast tissue\n• Prior radiation therapy to the chest\n• Hormone replacement therapy\n• Obesity and lack of exercise\n• Alcohol consumption\n• Late menopause or early first menstrual period",
    "histopathology": "**Histopathology** is the microscopic examination of tissue samples to diagnose disease. In breast cancer diagnosis:\n• A biopsy sample is taken from suspicious tissue\n• Pathologists examine cells under a microscope\n• They assess cell morphology, arrangement, and invasion patterns\n• Results confirm cancer type, grade, and stage\n\nThis AI model analyzes histopathology slide images (50×50 pixel patches) to detect IDC.",
    "resnet": "This application uses **ResNet50** (Residual Network with 50 layers), a deep convolutional neural network. It was:\n• Pre-trained on ImageNet (1.2M images, 1000 classes)\n• Fine-tuned on the IDC breast histopathology dataset\n• Trained on 50×50 pixel tissue patches labeled as IDC-positive or negative\n• Achieves high accuracy in distinguishing cancerous from normal tissue",
    "accuracy": "The ResNet50 model was trained on the **IDC Histopathology Dataset** (Kaggle), which contains:\n• 277,524 image patches (50×50 pixels)\n• 198,738 IDC-negative patches (class 0)\n• 78,786 IDC-positive patches (class 1)\n• The model uses transfer learning for high accuracy detection",
    "next steps": "If the analysis indicates IDC-positive:\n1. **Do not panic** — AI results are screening tools, not diagnoses\n2. **Schedule a doctor's appointment** immediately\n3. **Bring the report** to share with your physician\n4. **Additional tests** (mammogram, MRI, biopsy) will be ordered\n5. **Seek a specialist** — a breast cancer oncologist or surgeon\n6. **Get a second opinion** if desired\n\nEarly action leads to better outcomes.",
    "disclaimer": "⚠️ **Important Disclaimer**: This AI tool is for **educational and research purposes only**. It is NOT a medical diagnosis. Results should never replace professional medical advice. Always consult a qualified healthcare provider for:\n• Proper diagnosis\n• Treatment decisions\n• Medical guidance\n\nIf you are concerned about breast cancer, please see a doctor immediately.",
    "hello": "Hello! 👋 I'm your Breast Cancer Detection Assistant. I can help you understand:\n• Your analysis results\n• What IDC (Invasive Ductal Carcinoma) is\n• Symptoms and risk factors\n• Treatment options\n• Next steps after diagnosis\n\nWhat would you like to know?",
    "hi": "Hi there! 👋 I'm here to help you understand your analysis results and answer questions about breast cancer. What would you like to know?",
    "help": "I can answer questions about:\n• 🔬 **IDC / Invasive Ductal Carcinoma**\n• 📊 **Your analysis results**\n• ⚠️ **Symptoms** of breast cancer\n• 💊 **Treatment options**\n• 🏥 **Screening methods**\n• 📈 **Survival rates**\n• 🛡️ **Prevention tips**\n• ❓ **Next steps** after diagnosis\n\nJust ask me anything!",
    "thank": "You're welcome! 😊 Remember, if you have any health concerns, please consult a qualified medical professional. Take care!",
    "what should i do": "**Recommended next steps:**\n1. Review your analysis report carefully\n2. Note the confidence percentage and result\n3. **Schedule a medical appointment** regardless of the result\n4. Share this AI report with your doctor (not a replacement for clinical diagnosis)\n5. Undergo proper clinical screening (mammogram, biopsy if needed)\n6. Stay positive — early detection saves lives!\n\nWould you like more information about any specific step?",
}

def get_chat_reply(message: str, prediction: str = None, confidence: float = None) -> str:
    msg = message.lower().strip()
    
    # Context-aware responses
    if prediction and any(word in msg for word in ["result", "my result", "what does it mean", "explain", "analysis"]):
        if prediction == "IDC Positive":
            return f"Your analysis showed **{prediction}** with {confidence:.1f}% confidence. This means the AI detected patterns consistent with Invasive Ductal Carcinoma in the tissue image. ⚠️ **This is NOT a clinical diagnosis.** Please consult a breast cancer specialist immediately for proper evaluation, clinical examination, and if needed, a biopsy for definitive diagnosis. Early action is key to better outcomes."
        else:
            return f"Your analysis showed **{prediction}** with {confidence:.1f}% confidence. This means the AI did not detect strong IDC patterns in the tissue image. However, a negative AI result does **not** guarantee you are cancer-free. Regular screenings and doctor consultations are still very important for your health."
    
    # Keyword matching
    for key, response in CANCER_QA.items():
        if key in msg:
            return response
    
    # Partial keyword matching
    if any(word in msg for word in ["cancer", "tumor", "malignant"]):
        return CANCER_QA["what is idc"]
    if any(word in msg for word in ["treat", "cure", "therapy", "chemo", "surgery"]):
        return CANCER_QA["treatment"]
    if any(word in msg for word in ["symptom", "sign", "lump", "pain"]):
        return CANCER_QA["symptoms"]
    if any(word in msg for word in ["prevent", "avoid", "reduce risk"]):
        return CANCER_QA["prevention"]
    if any(word in msg for word in ["stage", "spread", "metasta"]):
        return CANCER_QA["stages"]
    if any(word in msg for word in ["survive", "survival", "prognosis", "chance"]):
        return CANCER_QA["survival rate"]
    if any(word in msg for word in ["screen", "mammogram", "detect early", "check"]):
        return CANCER_QA["screening"]
    if any(word in msg for word in ["risk", "factor", "gene", "brca"]):
        return CANCER_QA["risk factors"]
    if any(word in msg for word in ["next", "what do i do", "step", "action"]):
        return CANCER_QA["next steps"]
    if any(word in msg for word in ["accurate", "accuracy", "model", "ai", "how does"]):
        return CANCER_QA["accuracy"]
    if any(word in msg for word in ["disclaimer", "warning", "reliable", "trust"]):
        return CANCER_QA["disclaimer"]
    
    return "I'm here to help! I can answer questions about breast cancer, IDC, symptoms, treatments, and your analysis results. Could you please rephrase your question? For example, you can ask about 'symptoms', 'treatment', 'risk factors', or 'what does my result mean'."

# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(os.path.dirname(__file__), "style.css"))

@app.get("/script.js")
async def serve_js():
    return FileResponse(os.path.join(os.path.dirname(__file__), "script.js"))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image.")
    
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    img_array = preprocess_image(image_bytes)
    
    try:
        prediction_raw = model.predict(img_array)
        
        # Handle different output shapes
        if hasattr(prediction_raw, '__len__') and len(prediction_raw.shape) > 1:
            prob = float(prediction_raw[0][0])
        else:
            prob = float(prediction_raw[0])
        
        # Binary classification: > 0.5 = IDC Positive
        if prob > 0.5:
            label = "IDC Positive"
            confidence = prob * 100
            risk_level = "High" if confidence > 85 else "Moderate"
        else:
            label = "Non-Cancerous (IDC Negative)"
            confidence = (1 - prob) * 100
            risk_level = "Low"
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "probability": round(prob * 100, 2),
            "risk_level": risk_level,
            "timestamp": timestamp,
            "filename": file.filename,
            "model": "ResNet50 (Fine-tuned on IDC Histopathology Dataset)",
            "disclaimer": "This result is for research/educational purposes only and does NOT constitute a medical diagnosis."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = get_chat_reply(request.message, request.prediction, request.confidence)
    return ChatResponse(reply=reply)

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
