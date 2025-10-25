import streamlit as st
import tensorflow as tf
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import os
import tempfile

# Page config
st.set_page_config(
    page_title="Deepfake Detection App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --bg-color: #0f172a;
        --card-bg: #1e293b;
    }
    
    /* Custom styling */
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 3rem;
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #6366f1;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px;
    }
    
    h1 {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h3 {
        color: #6366f1;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Device setup for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= MODEL LOADING =============
@st.cache_resource
def load_image_model():
    """Load the image deepfake detection model"""
    model_path = os.path.join("models", "image.h5")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_video_model():
    """Load the video deepfake detection model"""
    model_path = os.path.join("models", "video.pth")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    
    # Define model architecture (ResNet18)
    def get_model():
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load models
image_model = load_image_model()
video_model = load_video_model()

# ============= PREPROCESSING =============
def preprocess_image(img_array):
    """Preprocess image for the image model"""
    # Get model input shape
    target_size = image_model.input_shape[1:3]  # (height, width)
    
    # Handle None or invalid dimensions - default to 224x224 if needed
    if target_size[0] is None or target_size[1] is None or target_size[0] <= 0 or target_size[1] <= 0:
        target_size = (224, 224)
    
    # Resize (note: cv2.resize expects (width, height) not (height, width)
    img_resized = cv2.resize(img_array, (target_size[1], target_size[0]))
    
    # Convert to float32 and preprocess
    img_float = img_resized.astype("float32")
    
    try:
        img_preprocessed = tf.keras.applications.xception.preprocess_input(img_float)
    except Exception:
        img_preprocessed = img_float / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    return img_batch

def get_video_transforms():
    """Get video preprocessing transforms"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ============= PREDICTION FUNCTIONS =============
def predict_image(image_file):
    """Predict if an image is deepfake"""
    # Read image
    img = cv2.imread(image_file)
    if img is None:
        raise ValueError("Could not read image file")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    img_preprocessed = preprocess_image(img_rgb)
    
    # Predict
    pred = image_model.predict(img_preprocessed, verbose=0)
    
    # Interpret output
    if pred.shape[-1] == 1:  # Binary (sigmoid)
        prob = float(pred[0][0])
    else:  # Categorical (softmax)
        prob = float(pred[0][1]) if pred.shape[-1] >= 2 else float(pred[0][0])
    
    label = "DEEPFAKE" if prob >= 0.5 else "REAL"
    
    return label, prob, img_rgb

def predict_video(video_file, transform):
    """Predict if a video is deepfake - processes ALL frames like video_test.py"""
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        st.error("Error: Cannot open video file.")
        return None, None, None
    
    frame_count = 0
    real_votes = 0
    fake_votes = 0
    all_predictions = []
    sample_frame = None
    
    # Process ALL frames (no sampling) - exactly like video_test.py
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save first frame for display
        if frame_count == 0:
            sample_frame = rgb
        
        img_t = transform(rgb).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            outp = video_model(img_t)
            prob = torch.softmax(outp, dim=1)[0].cpu().numpy()
            pred = prob.argmax()
        
        label = "REAL" if pred == 0 else "DEEPFAKE"
        conf = prob[pred]
        
        all_predictions.append({
            'frame': frame_count,
            'label': label,
            'confidence': float(conf)
        })
        
        # Count predictions for overall final result
        if label == "REAL":
            real_votes += 1
        else:
            fake_votes += 1
        
        frame_count += 1
    
    cap.release()
    
    # Final decision based on majority vote
    final_label = "REAL" if real_votes > fake_votes else "DEEPFAKE"
    final_confidence = max(real_votes, fake_votes) / (real_votes + fake_votes) if (real_votes + fake_votes) > 0 else 0
    
    results = {
        'label': final_label,
        'confidence': final_confidence,
        'real_votes': real_votes,
        'fake_votes': fake_votes,
        'total_frames': frame_count
    }
    
    return results, sample_frame, all_predictions

# ============= UI =============
# Header with gradient effect
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1>🎭 Deepfake Detection Application</h1>
    <p style="font-size: 1.2rem; color: #64748b; margin-top: -1rem;">
        AI-Powered Media Authenticity Verification
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for file upload
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0;">📤 Upload Files</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video file",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
    help="Upload an image (jpg, png) or video (mp4, avi, mov) file to analyze"
)

# Add info about supported formats
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: #1e293b; padding: 1rem; border-radius: 10px; border-left: 4px solid #6366f1;">
    <h4 style="color: #6366f1; margin-top: 0;">ℹ️ Supported Formats</h4>
    <ul style="color: #cbd5e1; font-size: 0.9rem;">
        <li><strong>Images:</strong> JPG, PNG</li>
        <li><strong>Videos:</strong> MP4, AVI, MOV</li>
    </ul>
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    # Determine file type
    file_ext = uploaded_file.name.split('.')[-1].lower()
    is_video = file_ext in ['mp4', 'avi', 'mov']
    
    # Display preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📂 Uploaded File")
        if is_video:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
                <h3 style="color: white; margin-top: 0;">📹 {uploaded_file.name}</h3>
                <p style="margin-bottom: 0;">Video file ready for analysis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white;">
                <h3 style="color: white; margin-top: 0;">🖼️ {uploaded_file.name}</h3>
                <p style="margin-bottom: 0;">Image file ready for analysis</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Process based on file type
    if is_video:
        # Video processing
        with col2:
            st.markdown("### 🔍 Analysis Results")
            with st.spinner("🔄 Processing video... This may take a moment."):
                transform = get_video_transforms()
                results, sample_frame, all_predictions = predict_video(tmp_file_path, transform)
        
        if results:
            # Display sample frame
            if sample_frame is not None:
                st.image(sample_frame, caption="Sample Frame from Video", use_container_width=True)
            
            # Display results
            st.markdown("### 📊 Detection Results")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                if results['label'] == "DEEPFAKE":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                        <h2 style="color: white; margin: 0;">⚠️ DEEPFAKE</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">Confidence: {results['confidence']*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                        <h2 style="color: white; margin: 0;">✅ REAL</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">Confidence: {results['confidence']*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_result2:
                st.metric("Real Frames", results['real_votes'])
                st.metric("Fake Frames", results['fake_votes'])
            
            with col_result3:
                st.metric("Total Frames Analyzed", results['total_frames'])
            
            # Progress bar
            st.markdown("### 📈 Frame-by-Frame Analysis")
            real_percent = (results['real_votes'] / results['total_frames'] * 100) if results['total_frames'] > 0 else 0
            fake_percent = (results['fake_votes'] / results['total_frames'] * 100) if results['total_frames'] > 0 else 0
            
            st.progress(0.5)
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                st.markdown(f"**Real Frames**: {real_percent:.1f}%")
            with col_bar2:
                st.markdown(f"**Deepfake Frames**: {fake_percent:.1f}%")
    
    else:
        # Image processing
        with col2:
            st.markdown("### 🔍 Analysis Results")
            with st.spinner("🔄 Processing image..."):
                label, prob, img_rgb = predict_image(tmp_file_path)
        
        # Display image and results
        st.image(img_rgb, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
        
        st.markdown("### 📊 Detection Results")
        
        col_result1, col_result2 = st.columns([1, 1])
        
        with col_result1:
            if label == "DEEPFAKE":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="color: white; margin: 0;">⚠️ DEEPFAKE</h2>
                    <p style="font-size: 1rem; margin: 0.5rem 0 0 0;">Probability: {prob:.4f} ({prob*100:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="color: white; margin: 0;">✅ REAL</h2>
                    <p style="font-size: 1rem; margin: 0.5rem 0 0 0;">Probability: {1-prob:.4f} ({(1-prob)*100:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            st.metric("Confidence Score", f"{prob:.4f}")
            # Confidence bar
            if prob >= 0.5:
                st.progress(prob)
                st.caption("Deepfake Confidence")
            else:
                st.progress(1 - prob)
                st.caption("Real Confidence")
    
    # Clean up temp file
    os.unlink(tmp_file_path)
    
else:
    # Show instructions when no file is uploaded
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2 style="color: white; text-align: center; margin-top: 0;">👈 Get Started</h2>
        <p style="text-align: center; font-size: 1.2rem; margin-bottom: 0;">
            Upload an image or video file from the sidebar to begin deepfake detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions in cards
    col_instr1, col_instr2 = st.columns(2)
    
    with col_instr1:
        st.markdown("""
        <div style="background: #1e293b; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #6366f1; height: 100%;">
            <h3 style="color: #6366f1; margin-top: 0;">📖 How to Use</h3>
            <ol style="color: #cbd5e1; line-height: 1.8;">
                <li>Upload an image or video from the sidebar</li>
                <li>Wait for the AI to analyze your media</li>
                <li>View the detection results instantly</li>
                <li>Check confidence scores and details</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col_instr2:
        st.markdown("""
        <div style="background: #1e293b; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #8b5cf6; height: 100%;">
            <h3 style="color: #8b5cf6; margin-top: 0;">🎯 How It Works</h3>
            <ul style="color: #cbd5e1; line-height: 1.8;">
                <li><strong>Images:</strong> Xception-based deep learning model</li>
                <li><strong>Videos:</strong> ResNet18 frame-by-frame analysis</li>
                <li><strong>Results:</strong> Confidence scores & detailed metrics</li>
                <li><strong>Fast:</strong> GPU acceleration when available</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Features")
    feature_cols = st.columns(4)
    features = [
        ("🖼️", "Image Support", "JPG, PNG formats"),
        ("🎬", "Video Support", "MP4, AVI, MOV"),
        ("⚡", "Fast Processing", "Real-time analysis"),
        ("🎯", "High Accuracy", "Advanced AI models")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                <h2 style="margin: 0.5rem 0;">{icon}</h2>
                <h4 style="margin: 0.5rem 0; color: white;">{title}</h4>
                <p style="margin: 0; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #64748b;">
    <p style="font-size: 1.1rem; margin: 0.5rem 0;">
        🎭 <strong>Deepfake Detection App</strong> | Powered by AI & Streamlit
    </p>
    <p style="font-size: 0.9rem; margin: 0.5rem 0;">
        Advanced deep learning models for media authenticity verification
    </p>
</div>
""", unsafe_allow_html=True)
