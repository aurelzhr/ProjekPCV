import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mediapipe as mp

# Fungsi deteksi wajah dan analisis kulit
def analyze_skin(image: Image.Image):
    image = np.array(image.convert('RGB'))
    height, width = image.shape[:2]
    scale = 500 / max(height, width)
    resized = cv2.resize(image, (int(width*scale), int(height*scale)))

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = face_detection.process(resized)

    if not results.detections:
        return None, "❌ Tidak ada wajah terdeteksi."

    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    x, y, w, h = int(bboxC.xmin * resized.shape[1]), int(bboxC.ymin * resized.shape[0]), int(bboxC.width * resized.shape[1]), int(bboxC.height * resized.shape[0])

    # Ambil ROI wajah
    face_roi = resized[y:y+h, x:x+w]

    # Buat mask seluruh wajah (tanpa landmark deteksi mata/bibir)
    mask = np.ones_like(face_roi) * 255

    # Ambil LAB
    lab = cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)

    # Masking sederhana (asumsikan seluruh wajah sebagai kulit)
    skin_pixels = lab.reshape(-1, 3)

    # K-Means clustering untuk warna dominan
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(skin_pixels)
    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    a_mean = dominant[1]
    l_value = dominant[0]
    undertone = "Warm" if a_mean > 130 else "Cool"
    skin_tone = "Light" if l_value > 70 else "Medium" if l_value > 45 else "Dark"

    return {
        'resized': resized,
        'masked': face_roi,
        'dominant': dominant,
        'l_value': l_value,
        'a_mean': a_mean,
        'undertone': undertone,
        'skin_tone': skin_tone,
        'box': (x, y, w, h)
    }, None

def get_makeup_recommendation(skin_tone, undertone):
    recommendations = {
        "Warm": "🍑 Rekomendasi: warna peach, coral, golden brown, bronze, warm pink, oranye brick",
        "Cool": "💙 Rekomendasi: warna mauve, rose pink, soft berry, cool red, plum, lavender",
        "Neutral": "🌸 Rekomendasi: hampir semua warna cocok, bisa coba nude pink, beige, rose brown"
    }
    if undertone in recommendations:
        return recommendations[undertone]
    return "🌈 Rekomendasi tidak ditemukan."

# Streamlit UI
st.title("💄 Project PCV (Deteksi Undertone Kulit)")
st.write("Deteksi undertone kulit Anda untuk personalisasi warna makeup.")

uploaded_file = st.file_uploader("📷 Upload gambar wajah Anda", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Wajah Diupload", use_container_width=True)
    with st.spinner("🔍 Menganalisis kulit..."):
        result, error = analyze_skin(image)

    if error:
        st.error(error)
    elif result:
        st.success("✅ Analisis Berhasil!")

        col1, col2 = st.columns(2)
        img_with_box = result['resized'].copy()
        x, y, w, h = result['box']
        cv2.rectangle(img_with_box, (x,y), (x+w,y+h), (0,255,0), 2)
        col1.image(img_with_box, caption="Deteksi Wajah")
        col2.image(result['masked'], caption="Area Kulit (Kasar)")

        st.markdown("---")
        st.subheader("🎨 Hasil Analisis")
        st.write(f"**Skin Tone**: {result['skin_tone']} (L = {result['l_value']:.1f})")
        st.write(f"**Undertone**: {result['undertone']} (a* = {result['a_mean']:.1f})")

        recommendation = get_makeup_recommendation(result['skin_tone'], result['undertone'])
        st.markdown(f"**💋 Rekomendasi Warna Makeup:**\n{recommendation}")

        fig, ax = plt.subplots()
        ax.barh(["Lightness (L)", "Red-Green (a*)", "Blue-Yellow (b*)"], result['dominant'], color=['gray','red','blue'])
        ax.set_xlim(0, 255)
        ax.set_title("Dominant LAB Color")
        st.pyplot(fig)
else:
    st.info("⬆️ Silakan upload gambar terlebih dahulu.")
