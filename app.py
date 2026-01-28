import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import os
import zipfile
import io

st.set_page_config(page_title="IA Kart Tracker", layout="wide")

# --- CSS compact ---
st.markdown("<style>.stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }</style>", unsafe_allow_html=True)

st.title("Système de Traque de Kart")

# --- Initialisation Session ---
if 'snapshots' not in st.session_state:
    st.session_state.snapshots = [] # Images (RGB)
if 'detections_count' not in st.session_state:
    st.session_state.detections_count = 0

# --- Sidebar : Config ---
with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Modèle (.pt)", value="runs/detect/train3/weights/best.pt")
    conf_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.4, 0.05)
    
    if st.button("Charger Modèle"):
        if os.path.exists(model_path):
            st.session_state['model'] = YOLO(model_path)
            st.success("Modèle Actif")
        else:
            st.error("Fichier introuvable")

    if st.button("Reset Session"):
        st.session_state.snapshots = []
        st.session_state.detections_count = 0
        st.rerun()

# --- Zone Statistiques (Point 3) ---
col_stat1, col_stat2, col_stat3 = st.columns(3)
metric_frame = col_stat1.empty()
metric_det = col_stat2.empty()
metric_status = col_stat3.empty()

# --- Zone Principale ---
col_vid, col_gal = st.columns([3, 1])
with col_vid:
    video_placeholder = st.empty()
    uploaded_video = st.file_uploader("Source Vidéo", type=['mp4', 'avi'])
with col_gal:
    st.subheader("Captures")
    gallery_placeholder = st.empty()

# --- Logique de Téléchargement (Point 4) ---
download_placeholder = st.empty()

def create_zip(images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images):
            # Conversion RGB -> BGR pour encodage correct par OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ret, buf = cv2.imencode(".jpg", img_bgr)
            if ret:
                zf.writestr(f"detection_{i:04d}.jpg", buf.tobytes())
    return zip_buffer

# --- Moteur de Traitement ---
if uploaded_video is not None and 'model' in st.session_state:
    
    # Lecture par flux (Fix erreur 500)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    while True:
        chunk = uploaded_video.read(4 * 1024 * 1024)
        if not chunk: break
        tfile.write(chunk)
    tfile.flush()
    tfile.close()
    
    vf = cv2.VideoCapture(tfile.name)
    total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_counter = 0
    stop_button = st.button("Interrompre Analyse")
    
    # Boucle de lecture
    while vf.isOpened() and not stop_button:
        ret, frame = vf.read()
        if not ret:
            break
        
        # Inférence
        results = st.session_state['model'].predict(frame, verbose=False, conf=conf_threshold)
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Mise à jour Vidéo
        video_placeholder.image(annotated_frame_rgb, channels="RGB")
        
        # Mise à jour Stats
        metric_frame.metric("Frames Traitées", f"{frame_counter}/{total_frames}")
        
        # Logique de détection
        if len(results[0].boxes) > 0:
            st.session_state.detections_count += 1
            metric_det.metric("Karts Détectés", st.session_state.detections_count)
            metric_status.info("⚠️ DETECTION EN COURS")
            
            # Capture (limité à 1 toutes les 10 frames)
            if frame_counter % 10 == 0:
                st.session_state.snapshots.append(annotated_frame_rgb)
                # Mise à jour Galerie
                with gallery_placeholder.container():
                    st.image(annotated_frame_rgb, use_container_width=True)
        else:
            metric_status.success("R.A.S")

        frame_counter += 1

    vf.release()
    os.remove(tfile.name) # Nettoyage

    # Affichage du bouton de téléchargement si détections
    if st.session_state.snapshots:
        zip_file = create_zip(st.session_state.snapshots)
        download_placeholder.download_button(
            label="⬇️ Télécharger les Frames (ZIP)",
            data=zip_file.getvalue(),
            file_name="detections_kart.zip",
            mime="application/zip"
        )

elif uploaded_video and 'model' not in st.session_state:
    st.error("Modèle non chargé.")