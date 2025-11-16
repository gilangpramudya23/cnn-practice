import streamlit as st
from ultralytics import YOLO
import supervision as sv
from collections import Counter
import cv2
import numpy as np
from PIL import Image

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Construction Safety Detection - YOLO",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 Construction Safety Detection System (YOLOv12)")
st.markdown("Upload an image to detect safety equipment violations")

# ===========================
# LOAD MODEL (Cached)
# ===========================
@st.cache_resource
def load_yolo_model():
    """Load YOLO model (runs only once)"""
    with st.spinner('🔄 Loading YOLO model... Please wait'):
        model = YOLO('best.pt')  # Your model file
    st.success('✅ Model loaded successfully!')
    return model

# Load model
model = load_yolo_model()

# ===========================
# FUNCTION: COUNT SAFETY EQUIPMENT
# ===========================
def count_safety_equipment(detections, class_names):
    """Count helmets, vests, and violations"""
    
    detection_counts = Counter()
    
    for class_id in detections.class_id:
        class_name = class_names[class_id]
        detection_counts[class_name] += 1
    
    # Safety equipment counts
    helmet_count = detection_counts.get('Hardhat', 0) + detection_counts.get('helmet', 0)
    no_helmet_count = detection_counts.get('NO-Hardhat', 0) + detection_counts.get('no-helmet', 0)
    vest_count = detection_counts.get('Safety Vest', 0) + detection_counts.get('vest', 0)
    no_vest_count = detection_counts.get('NO-Safety Vest', 0) + detection_counts.get('no-vest', 0)
    person_count = detection_counts.get('Person', 0) + detection_counts.get('person', 0)
    
    return {
        'helmet': helmet_count,
        'no_helmet': no_helmet_count,
        'vest': vest_count,
        'no_vest': no_vest_count,
        'person': person_count
    }

# ===========================
# FUNCTION: RUN DETECTION
# ===========================
def detect_objects(image, confidence_threshold=0.5):
    """Run YOLO detection on uploaded image"""
    
    # Convert PIL to numpy array (BGR for OpenCV)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Run YOLO inference
    results = model(image_bgr, conf=confidence_threshold, verbose=False)[0]
    
    # Convert to supervision format
    detections = sv.Detections.from_ultralytics(results).with_nms()
    
    # Count safety equipment
    class_names = model.names
    counts = count_safety_equipment(detections, class_names)
    
    # Annotate image
    annotated_image = annotate_image_with_summary(image_bgr, detections, counts)
    
    # Convert back to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, detections, counts

# ===========================
# SIDEBAR: SETTINGS
# ===========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(
    f"""
    **Model:** YOLOv12
    **Classes:** {len(model.names)}
    
    **Detects:**
    - 🪖 Helmets / Hard hats
    - 🦺 Safety vests
    - ❌ Violations
    """
)

# Show class names
with st.sidebar.expander("🏷️ Class Names"):
    for idx, name in model.names.items():
        st.text(f"{idx}: {name}")

# ===========================
# MAIN APP: IMAGE UPLOAD
# ===========================

uploaded_file = st.file_uploader(
    "📤 Upload Construction Site Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image to detect safety equipment"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    
    # Run detection
    with st.spinner('🔍 Detecting objects...'):
        annotated_image, detections, counts = detect_objects(image, confidence_threshold)
    
    with col2:
        st.subheader("🎯 Detection Results")
        st.image(annotated_image, use_container_width=True)
    
    # ===========================
    # DISPLAY DETAILED COUNTS
    # ===========================
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    # Metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric("👷 People", counts['person'])
    
    with metric_col2:
        st.metric("🪖 With Helmet", counts['helmet'])
    
    with metric_col3:
        st.metric("❌ No Helmet", counts['no_helmet'],
                  delta="Violation" if counts['no_helmet'] > 0 else None,
                  delta_color="inverse")
    
    with metric_col4:
        st.metric("🦺 With Vest", counts['vest'])
    
    with metric_col5:
        st.metric("❌ No Vest", counts['no_vest'],
                  delta="Violation" if counts['no_vest'] > 0 else None,
                  delta_color="inverse")
    
    # Calculate compliance
    total_people = max(
        counts['person'],
        counts['helmet'] + counts['no_helmet'],
        counts['vest'] + counts['no_vest']
    )
    
    if total_people > 0:
        st.markdown("---")
        st.subheader("📈 Compliance Rate")
        
        helmet_compliance = (counts['helmet'] / total_people) * 100
        vest_compliance = (counts['vest'] / total_people) * 100
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.metric("🪖 Helmet Compliance", f"{helmet_compliance:.1f}%")
            st.progress(helmet_compliance / 100)
        
        with comp_col2:
            st.metric("🦺 Vest Compliance", f"{vest_compliance:.1f}%")
            st.progress(vest_compliance / 100)
        
        # Overall status
        st.markdown("---")
        total_violations = counts['no_helmet'] + counts['no_vest']
        
        if total_violations == 0 and total_people > 0:
            st.success("✅ **FULLY COMPLIANT** - All workers have proper safety equipment!")
        elif total_violations <= total_people * 0.2:
            st.warning("⚠️ **MOSTLY COMPLIANT** - Minor violations detected")
        else:
            st.error("❌ **NON-COMPLIANT** - Safety violations detected!")
    
    # Show detailed detections
    with st.expander("🔍 View Detailed Detections"):
        st.write(f"Total objects detected: **{len(detections)}**")
        
        if len(detections) > 0:
            detection_data = []
            
            # FIXED: Correct way to iterate through supervision Detections
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]
                
                detection_data.append({
                    "#": i + 1,
                    "Class": model.names[class_id],
                    "Confidence": f"{confidence:.2%}",
                    "BBox": f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                })
            
            import pandas as pd
            df = pd.DataFrame(detection_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected")

else:
    # Show instructions when no image uploaded
    st.info("👆 Upload an image to start detection")
    
    st.markdown("### 🎯 How to use:")
    st.markdown("""
    1. Click **'Browse files'** button above
    2. Select a construction site image (JPG, JPEG, or PNG)
    3. Wait for YOLO to process (~1-2 seconds)
    4. View detected objects and safety compliance metrics
    5. Adjust confidence threshold in sidebar if needed
    """)
    
    st.markdown("### ✨ Features:")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**🎯 Real-time Detection**")
        st.write("Instant object detection on upload")
    
    with col_b:
        st.markdown("**📊 Safety Metrics**")
        st.write("Automatic compliance calculation")
    
    with col_c:
        st.markdown("**🔧 Adjustable Settings**")
        st.write("Control detection sensitivity")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🦺 Construction Safety Detection System | Powered by YOLOv12</p>
    </div>
    """,
    unsafe_allow_html=True

)

