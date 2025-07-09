from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

@app.route("/")
def home():
    return open("index.html", "r", encoding="utf-8").read()

@app.route("/analyze", methods=["POST"])
def analyze_skin():
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada gambar diupload"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    path = os.path.join("static", file.filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "Gagal baca gambar"}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = 500 / max(h, w)
    resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify({"error": "Wajah tidak terdeteksi"}), 400

    x, y, w, h = faces[0]
    face = resized[y:y+h, x:x+w]

    # Perkiraan area cepat bukan kulit: atas dan bawah
    mask = np.ones_like(resized) * 255
    roi_top = int(y + 0.1*h)
    roi_bottom = int(y + 0.9*h)
    cv2.rectangle(mask, (0,0), (resized.shape[1], roi_top), (0,0,0), -1)
    cv2.rectangle(mask, (0,roi_bottom), (resized.shape[1], resized.shape[0]), (0,0,0), -1)

    masked = cv2.bitwise_and(resized, mask)
    face_masked = masked[y:y+h, x:x+w]

    lab = cv2.cvtColor(face_masked, cv2.COLOR_RGB2LAB)
    skin = lab.reshape(-1,3)

    if len(skin)==0:
        return jsonify({"error": "Area kulit kosong"}), 400

    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(skin)
    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    L, a, b = dominant
    undertone = "Warm" if a > 130 else "Cool"
    skin_tone = "Light" if L > 70 else "Medium" if L > 45 else "Dark"

    return jsonify({"skin_tone": skin_tone, "undertone": undertone, "L": round(float(L),1), "a": round(float(a),1), "b": round(float(b),1)})

if __name__ == "__main__":
    app.run(debug=True)
