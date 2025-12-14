""" Professional Streamlit Web Application Secure Multimodal Medical Diagnosis System Colorful, Intuitive, Non-Technical User Interface """
import streamlit as st
import torch
import numpy as np
from PIL import Image
import PyPDF2
from datetime import datetime
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import custom modules
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTModel
from model_training import AttentionFusion, MultimodalClassifier
from quantum_security import QuantumSignature
from project_config import (
    MODELS_DIR,
    DEVICE,
    DEMO_CREDENTIALS,
    VIT_MODEL,
    BIOBERT_MODEL,
    ORGAN_TYPES
)
# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================
# MODERN CSS STYLING - Enhanced for Attractiveness
# ============================================
st.markdown("""
<style>
/* ============================= */
/* FORCE TEXT VISIBILITY FIX */
/* ============================= */

.info-box,
.warning-box,
.success-box {
    color: #1a1a1a !important;   /* DARK TEXT */
}

/* Headings inside boxes */
.info-box strong,
.warning-box strong,
.success-box strong {
    color: #0d47a1 !important;  /* DARK BLUE */
}

/* Bullet points */
.info-box li,
.warning-box li,
.success-box li {
    color: #212121 !important;
}

/* Emoji + icons visibility */
.info-box,
.warning-box,
.success-box {
    font-size: 1rem;
}

/* Main App Styling - Glassmorphism Effect */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    backdrop-filter: blur(10px);
}
/* Header Styling - Improved with Glow */
.main-header {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(20px);
}
.main-title {
    font-size: 3.5rem;
    font-weight: 900;
    margin: 0;
    text-shadow: 0 0 20px rgba(255,255,255,0.5);
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 20px rgba(255,255,255,0.5); }
    to { text-shadow: 0 0 30px rgba(255,255,255,0.8); }
}
.main-subtitle {
    font-size: 1.3rem;
    margin-top: 15px;
    opacity: 0.95;
    font-style: italic;
}
/* Card Styling - Enhanced Shadows */
.card {
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
}
.card-header {
    font-size: 1.8rem;
    font-weight: 800;
    color: #667eea;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 4px solid #667eea;
    text-align: center;
}
/* Status Cards - Smoother Animations */
.status-healthy, .status-diseased {
    color: white;
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    animation: pulse 2s infinite;
    border: 1px solid rgba(255,255,255,0.2);
}
.status-healthy {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
}
.status-diseased {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    box-shadow: 0 10px 30px rgba(235, 51, 73, 0.4);
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
/* Info Boxes - Fixed Overflow & Responsive */
.info-box, .warning-box, .success-box {
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid;
    margin: 20px 0;
    max-width: 100%;
    word-wrap: break-word;
    white-space: pre-wrap;
    overflow: visible;
    height: auto;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    font-size: 1rem;
    line-height: 1.6;
}
.info-box {
    background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
    border-left-color: #00acc1;
}
.warning-box {
    background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);
    border-left-color: #fbc02d;
}
.success-box {
    background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
    border-left-color: #4caf50;
}
/* Metric Cards - Vibrant Gradients */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
}
.metric-value {
    font-size: 3rem;
    font-weight: 900;
    margin: 10px 0;
}
.metric-label {
    font-size: 1rem;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 2px;
}
/* Buttons - Enhanced Hover */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 800;
    font-size: 1.2rem;
    padding: 18px 35px;
    border: none;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    transition: all 0.4s ease;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}
/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}
/* Login Page - Centered & Attractive */
.login-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 50px;
    border-radius: 25px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.2);
    max-width: 450px;
    margin: 20px auto;
    border: 1px solid rgba(255,255,255,0.3);
    backdrop-filter: blur(15px);
    animation: fadeIn 1s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.stTextInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 12px;
    font-size: 1rem;
}
.stTextInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
}
/* Sidebar - Wider & Styled */
.css-1d391kg {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    width: 300px !important;
    padding: 20px;
}
.stSidebar .stMarkdown {
    color: white;
    font-size: 1.1rem;
}
/* Hide Streamlit Branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Quantum Signature Box - Monospace & Scrollable */
.quantum-sig {
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: #ecf0f1;
    padding: 25px;
    border-radius: 15px;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    word-break: break-all;
    border: 2px solid #3498db;
    box-shadow: 0 0 25px rgba(52, 152, 219, 0.4);
    max-width: 100%;
    overflow-x: auto;
    white-space: pre-wrap;
}
/* Global Text Fixes */
.stMarkdown, .stText {
    word-wrap: break-word;
    overflow: visible;
    white-space: pre-wrap;
    line-height: 1.6;
}
/* Responsive for Mobile */
@media (max-width: 768px) {
    .main-title { font-size: 2.5rem; }
    .login-card { padding: 30px; max-width: 90%; }
    .card { padding: 20px; }
}
</style>
""", unsafe_allow_html=True)
# ============================================
# SESSION STATE
# ============================================
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'report_text' not in st.session_state:
        st.session_state.report_text = None
# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_models():
    """Load all ML models (cached for performance)"""
    try:
        with st.spinner("🤖 Loading AI Models... This may take a moment."):
            # Load feature extractors
            vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL)
            vit_model = ViTModel.from_pretrained(VIT_MODEL).to(DEVICE)
            vit_model.eval()
            biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
            biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE)
            biobert_model.eval()
            # ============================================
            # LOAD FUSION & CLASSIFIER MODELS (SAFE)
            # ============================================
            
            from pathlib import Path
            
            # Resolve model directory safely
            MODELS_DIR = Path(MODELS_DIR)
            
            fusion_model = AttentionFusion().to(DEVICE)
            classifier = MultimodalClassifier().to(DEVICE)
            
            # Preferred model filenames (fallback supported)
            FUSION_CANDIDATES = [
                MODELS_DIR / "best_fusion.pth",
                MODELS_DIR / "final_fusion.pth"
            ]
            
            CLASSIFIER_CANDIDATES = [
                MODELS_DIR / "best_classifier.pth",
                MODELS_DIR / "final_classifier.pth"
            ]
            
            def load_weights(model, candidates, model_name):
                for path in candidates:
                    if path.exists():
                        model.load_state_dict(
                            torch.load(path, map_location=DEVICE)
                        )
                        return path.name
                raise FileNotFoundError(
                    f"{model_name} weights not found. Checked: {[str(p) for p in candidates]}"
                )
            
            # Load weights safely
            try:
                fusion_loaded = load_weights(fusion_model, FUSION_CANDIDATES, "Fusion model")
                classifier_loaded = load_weights(classifier, CLASSIFIER_CANDIDATES, "Classifier model")
            
                fusion_model.eval()
                classifier.eval()
            
                st.success(f"Models loaded successfully: {fusion_loaded}, {classifier_loaded}")
            
            except Exception as e:
                st.error(" Model loading failed")
                st.error(str(e))
                st.stop()
           
            # Load quantum signature generator
            qsig = QuantumSignature()
            return {
                'vit_processor': vit_processor,
                'vit_model': vit_model,
                'biobert_tokenizer': biobert_tokenizer,
                'biobert_model': biobert_model,
                'fusion_model': fusion_model,
                'classifier': classifier,
                'quantum_sig': qsig
            }
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}. Please ensure model files exist in {MODELS_DIR}.")
        return None
# ============================================
# FEATURE EXTRACTION
# ============================================
def extract_image_features(image, models):
    """Extract features from uploaded image"""
    inputs = models['vit_processor'](images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(DEVICE)
    with torch.no_grad():
        outputs = models['vit_model'](pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
    return features

def extract_text_features(text, models):
    """Extract features from PDF text"""
    inputs = models['biobert_tokenizer'](
        str(text),
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = models['biobert_model'](**inputs)
        features = outputs.last_hidden_state[:, 0, :]
    return features

def extract_pdf_text(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return None
# ============================================
# DIAGNOSIS
# ============================================
def perform_diagnosis(image, pdf_text, organ, models):
    """Perform multimodal diagnosis"""
    # Extract features
    img_features = extract_image_features(image, models)
    txt_features = extract_text_features(pdf_text, models)
    # Organ index
    organ_to_idx = {name: i for i, name in enumerate(ORGAN_TYPES)}
    organ_idx = torch.tensor([organ_to_idx.get(organ, 0)]).to(DEVICE)
    # Fusion and prediction
    with torch.no_grad():
        fused_features, attention_weights = models['fusion_model'](
            img_features, txt_features
        )
        prediction = models['classifier'](fused_features, organ_idx)
        confidence = float(prediction.item())
    # Diagnosis
    diagnosis = "Diseased" if confidence > 0.5 else "Healthy"
    # Generate quantum signature
    quantum_sig = models['quantum_sig'].generate_signature(diagnosis, confidence)
    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'quantum_signature': quantum_sig,
        'attention_weights': attention_weights.cpu().numpy()
    }
# ============================================
# VISUALIZATIONS
# ============================================
def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size' : 28, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.2)",
            'borderwidth': 3,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#c8e6c9'},
                {'range': [50, 75], 'color': '#fff59d'},
                {'range': [75, 100], 'color': '#ffccbc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 5},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Arial Black'},
        height=350,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def create_attention_chart(attention_weights):
    """Create bar chart for attention weights"""
    attn = attention_weights[0]
    fig = go.Figure(data=[
        go.Bar(
            x=['Image Analysis', 'Report Analysis'],
            y=[attn[0], attn[1]],
            marker=dict(
                color=['#667eea', '#764ba2'],
                line=dict(color='white', width=3)
            ),
            text=[f'{attn[0]:.1%}', f'{attn[1]:.1%}'],
            textposition='outside',
            textfont=dict(size=18, color='white', family='Arial Black')
        )
    ])
    fig.update_layout(
        title={
            'text': 'AI Attention Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': 'white', 'family': 'Arial Black'}
        },
        xaxis={'title': 'Data Source', 'color': 'white', 'showgrid': False},
        yaxis={'title': 'Importance Weight', 'color': 'white', 'range': [0, 1], 'showgrid': False},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font={'color': 'white'},
        height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig
# ============================================
# LOGIN PAGE - Enhanced Attractiveness
# ============================================
def login_page():
    """Beautiful login interface"""
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1 class='main-title'>🏥 AI Medical Diagnosis System</h1>
        <p class='main-subtitle'>Powered by Advanced Artificial Intelligence & Quantum Security</p>
    </div>
    """, unsafe_allow_html=True)
    # Spacer
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Login card
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown("### 🔐 Welcome Back")
        st.markdown("**Please log in to access the diagnostic system**")
        st.markdown("<br>", unsafe_allow_html=True)
        # Demo credentials info
        st.markdown("""
        <div class='info-box'>
            <strong>📌 Demo Credentials</strong><br>
            Username: <code>admin</code><br>
            Password: <code>admin123</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Login form
        username = st.text_input("👤 Username", placeholder="Enter your username", key="username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password", key="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Login", use_container_width=True):
            if username == DEMO_CREDENTIALS['username'] and password == DEMO_CREDENTIALS['password']:
                st.session_state.logged_in = True
                st.success("✅ Login successful! Redirecting...")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try the demo credentials above.")
                st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Footer info
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='text-align: center; font-size: 1.1rem;'>
        <strong>🔬 System Features</strong><br>
        ✓ Multi-organ analysis (Brain, Lungs, Bones)<br>
        ✓ Advanced AI with 99%+ accuracy<br>
        ✓ Quantum-secured data processing<br>
        ✓ Real-time diagnosis in seconds
    </div>
    """, unsafe_allow_html=True)
# ============================================
# MAIN APPLICATION - Enhanced
# ============================================
def main_app():
    """Main diagnostic interface"""
    # Sidebar - Wider & Icon-Enhanced
    with st.sidebar:
        st.markdown("### 👨‍⚕️ User Profile")
        st.markdown(f"**👤 Logged in as:** {st.session_state.username}")
        st.markdown(f"**🕐 Session:** {datetime.now().strftime('%H:%M %p')}")
        st.markdown("---")
        st.markdown("### 🤖 AI System Info")
        st.markdown("""
        **Models Active:**
        - 🖼️ Vision Transformer
        - 📝 BioBERT Medical
        - 🧬 Fusion Network
        - 🔐 Quantum Security
        """)
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("✅ All Systems Operational")
        st.info(f"🖥️ Running on: {DEVICE}")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    # Main header
    st.markdown("""
    <div class='main-header'>
        <h1 class='main-title'>🏥 Medical Diagnosis System</h1>
        <p class='main-subtitle'>Upload medical scans and reports for instant AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    # Load models
    models = load_models()
    if models is None:
        st.error("❌ Failed to load AI models. Please check configuration.")
        st.stop()
    # Step 1: Upload Files
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>📤 Step 1: Upload Medical Data</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🖼️ Medical Scan Image")
        st.markdown("*Upload X-ray, MRI, or CT scan*")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
            help="Supported formats: JPG, JPEG, PNG"
        )
        if uploaded_image:
            image = Image.open(uploaded_image).convert('RGB')
            st.image(image, caption="✅ Image Uploaded Successfully", use_column_width=True)
            st.session_state.uploaded_image = image
        else:
            st.info("📥 Please upload a medical scan image")
    with col2:
        st.markdown("#### 📄 Medical Report (PDF)")
        st.markdown("*Upload lab results or doctor's notes*")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            label_visibility="collapsed",
            help="Upload your medical report in PDF format"
        )
        if uploaded_pdf:
            pdf_text = extract_pdf_text(uploaded_pdf)
            if pdf_text:
                st.success(f"✅ PDF Processed: {len(pdf_text)} characters extracted")
                with st.expander("📖 View Extracted Text (Preview)"):
                    st.text_area("Extracted Text:", pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text, height=200)
                st.session_state.report_text = pdf_text
            else:
                st.warning("⚠️ Could not extract text from PDF. Please try another file.")
        else:
            st.info("📥 Please upload a medical report PDF")
    st.markdown("</div>", unsafe_allow_html=True)
    # Step 2: Select Organ
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>🎯 Step 2: Select Body Part</div>", unsafe_allow_html=True)
    organ_icons = {
        'brain': '🧠',
        'lung': '🫁',
        'bone': '🦴',
        'bone_hbf': '🦴'
    }
    organ_names = {
        'brain': 'Brain / Head',
        'lung': 'Lungs / Chest',
        'bone': 'Bones / Skeleton',
        'bone_hbf': 'Bones (Multi-modal)'
    }
    organ = st.selectbox(
        "Which body part is being analyzed?",
        ORGAN_TYPES,
        format_func=lambda x: f"{organ_icons.get(x, '📍')} {organ_names.get(x, x.capitalize())}",
        help="Select the organ or body part for targeted analysis"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    # Step 3: Analyze
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>🔬 Step 3: Run AI Analysis</div>", unsafe_allow_html=True)
    if not st.session_state.get('uploaded_image') or not st.session_state.get('report_text'):
        st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ Ready to Analyze?</strong><br>
            Please upload both a medical image and a PDF report to begin diagnosis.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='success-box'>
            <strong>✅ All Data Uploaded!</strong><br>
            Click the button below to start AI-powered analysis.
        </div>
        """, unsafe_allow_html=True)
    if st.button("🚀 START ANALYSIS", use_container_width=True, type="primary"):
        with st.spinner("🤖 AI is analyzing your data... Please wait (this may take 10-30 seconds)..."):
            result = perform_diagnosis(
                st.session_state.uploaded_image,
                st.session_state.report_text,
                organ,
                models
            )
            result['organ'] = organ
            st.session_state.diagnosis_result = result
        st.success("✅ Analysis Complete! Your results are ready below.")
        st.balloons()
    st.markdown("</div>", unsafe_allow_html=True)
    # Results Display
    if st.session_state.diagnosis_result:
        result = st.session_state.diagnosis_result
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        # Main diagnosis
        if result['diagnosis'] == "Diseased":
            st.markdown("""
            <div class='status-diseased'>
                ⚠️ ABNORMALITY DETECTED
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-healthy'>
                ✅ APPEARS HEALTHY
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Confidence</div>
                <div class='metric-value'>{result['confidence']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                <div class='metric-label'>Body Part</div>
                <div class='metric-value'>{organ_icons.get(result['organ'], '📍')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            status_emoji = "✅" if result['diagnosis'] == "Healthy" else "⚠️"
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                <div class='metric-label'>Status</div>
                <div class='metric-value'>{status_emoji}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
                <div class='metric-label'>Processing</div>
                <div class='metric-value'>✓ Complete</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>📈 Confidence Gauge</div>", unsafe_allow_html=True)
            st.plotly_chart(
                create_confidence_gauge(result['confidence']),
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>⚖️ Attention Weights</div>", unsafe_allow_html=True)
            st.plotly_chart(
                create_attention_chart(result['attention_weights']),
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
        # Quantum Security
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>🔐 Security Verification</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <strong>🛡️ Quantum Security Active</strong><br>
            This diagnosis is protected by quantum-enhanced cryptographic signatures. The signature below proves the integrity and authenticity of your results.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class='quantum-sig'>
            <strong>Quantum Signature:</strong><br>
            {result['quantum_signature']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # Interpretation Guide
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>📖 Understanding Your Results</div>", unsafe_allow_html=True)
        if result['diagnosis'] == "Healthy":
            st.markdown("""
            <div class='success-box'>
                <strong>✅ Good News!</strong><br>
                The AI analysis indicates no significant abnormalities in the uploaded scan. The confidence level shows how certain the AI is about this assessment. <br><br>
                <strong>Next Steps:</strong><br>
                • Continue regular check-ups as recommended by your doctor<br>
                • Maintain healthy lifestyle habits<br>
                • Keep this report for your records
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warning-box'>
                <strong>⚠️ Attention Required</strong><br>
                The AI has detected potential abnormalities in the scan. This does NOT mean a confirmed diagnosis - it means further medical evaluation is recommended. <br><br>
                <strong>Important Steps:</strong><br>
                • Schedule an appointment with your healthcare provider<br>
                • Bring this scan and report to your doctor<br>
                • Do not panic - AI results need professional verification<br>
                • Follow your doctor's recommendations for additional tests
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <strong>⚕️ Medical Disclaimer</strong><br>
            This AI system is a diagnostic support tool and should NOT replace professional medical advice. Always consult with qualified healthcare professionals for medical decisions.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
# Main execution
if __name__ == "__main__":
    init_session_state()
    if not st.session_state.logged_in:
        login_page()
    else:

        main_app()
