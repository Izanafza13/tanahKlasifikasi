import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = "model_sv.pkl"      # model Soft Voting hasil training
# Catatan: DT, RF, XGBoost, dan Voting berbasis pohon TIDAK butuh scaling,
# jadi scaler sengaja tidak dipakai di sini.

FEATURE_NAMES = [
    "suhu_tanah",
    "kelembapan_tanah",
    "suhu_udara",
    "kelembapan_udara",
    "ph_tanah",
]

FEATURE_LABELS = {
    "suhu_tanah": "Suhu Tanah (°C)",
    "kelembapan_tanah": "Kelembapan Tanah (%)",
    "suhu_udara": "Suhu Udara (°C)",
    "kelembapan_udara": "Kelembapan Udara (%)",
    "ph_tanah": "pH Tanah",
}

# Range default buat slider/input, sesuaikan sama data kamu
# Default disetel mendekati batas S1 (Sangat Sesuai) / S2 (Cukup Sesuai)
# berdasarkan kriteria umum kesesuaian lahan padi (acuan literatur,
# mis. Djaenudin dkk.). Sesuaikan lagi kalau kamu punya threshold
# asli dari proses labeling data training model ini.
FEATURE_RANGES = {
    # Default ini SUDAH divalidasi langsung ke model_sv.pkl (weighted soft
    # voting dt+rf+xgb): menghasilkan P(S1)=35.2% vs P(S2)=28.1% - dua
    # kelas teratas & berdekatan, jadi hasilnya pas di perbatasan S1/S2.
    "suhu_tanah": (15.0, 40.0, 31.0),        # (min, max, default)
    "kelembapan_tanah": (0.0, 100.0, 65.0),
    "suhu_udara": (15.0, 40.0, 31.0),
    "kelembapan_udara": (0.0, 100.0, 65.0),
    "ph_tanah": (3.0, 9.0, 5.5),
}

LABEL_DESC = {
    "S1": "Sangat Sesuai",
    "S2": "Cukup Sesuai",
    "S3": "Sesuai Marginal",
    "N": "Tidak Sesuai",
}

# Fallback: dipakai KALAU model_sv.pkl tidak menyimpan label_encoder,
# sehingga hasil predict() masih berupa angka bukan teks.
# Kelas di model ini dinomori mulai dari 1 (bukan 0): 1=S1, 2=S2, 3=S3, 4=N
CLASS_INDEX_MAP = {
    1: "S1",
    2: "S2",
    3: "S3",
    4: "N",
}

st.set_page_config(page_title="SoilML - Prediksi Kesesuaian Lahan", page_icon="🌾", layout="centered")


# =========================================================
# LOAD MODEL (di-cache biar gak reload tiap interaksi)
# =========================================================
class ManualSoftVotingModel:
    """
    Wrapper buat dict hasil training soft voting manual berisi:
      { "dt": DecisionTreeClassifier, "rf": RandomForestClassifier,
        "xgb": XGBClassifier, "w_dt": float, "w_rf": float, "w_xgb": float }

    PENTING: xgb dilatih dengan label 0-indexed (kelas 0,1,2,3), sedangkan
    dt & rf dilatih dengan label asli (kelas 1,2,3,4 = S1,S2,S3,N). Karena
    urutan kelas di kedua encoding sama-sama urut naik (posisi ke-0 = S1,
    posisi ke-1 = S2, dst), kolom predict_proba dari xgb dan dt/rf sudah
    selaras secara posisi - tinggal dijumlahkan dengan bobotnya.
    """

    def __init__(self, dt, rf, xgb, w_dt, w_rf, w_xgb):
        self.dt, self.rf, self.xgb = dt, rf, xgb
        self.w_dt, self.w_rf, self.w_xgb = w_dt, w_rf, w_xgb
        # kelas final memakai penomoran dt/rf (1,2,3,4) karena itu label asli
        self.classes_ = dt.classes_

    def predict_proba(self, X):
        p_dt = self.dt.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        return self.w_dt * p_dt + self.w_rf * p_rf + self.w_xgb * p_xgb

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


def _extract_model_and_extras(obj):
    """
    Menangani beberapa kemungkinan bentuk isi model_sv.pkl:
    1) Objek model langsung (punya .predict) -> dipakai apa adanya.
    2) Dict soft-voting manual berisi dt/rf/xgb + bobotnya -> dibungkus
       jadi ManualSoftVotingModel supaya ketiga model & bobotnya benar2
       dipakai (bukan cuma salah satu model saja).
    3) Dict generik lain -> coba cari objek yang punya .predict di dalamnya,
       plus label_encoder & feature_names kalau ada.
    """
    if hasattr(obj, "predict"):
        return obj, None, None

    if isinstance(obj, dict):
        # Kasus soft voting manual (dt/rf/xgb + bobot)
        if all(k in obj for k in ["dt", "rf", "xgb", "w_dt", "w_rf", "w_xgb"]):
            model_obj = ManualSoftVotingModel(
                obj["dt"], obj["rf"], obj["xgb"],
                obj["w_dt"], obj["w_rf"], obj["w_xgb"],
            )
            return model_obj, None, None

        label_encoder = None
        feature_names = None
        model_obj = None

        # coba key yang umum dipakai dulu
        common_model_keys = ["model", "best_model", "clf", "classifier", "voting", "soft_voting", "estimator"]
        for k in common_model_keys:
            if k in obj and hasattr(obj[k], "predict"):
                model_obj = obj[k]
                break

        # kalau belum ketemu, scan semua value, cari yang punya .predict
        if model_obj is None:
            for k, v in obj.items():
                if hasattr(v, "predict"):
                    model_obj = v
                    break

        # cari label encoder kalau ada
        for k in ["label_encoder", "le", "encoder"]:
            if k in obj:
                label_encoder = obj[k]
                break

        # cari daftar nama fitur kalau ada
        for k in ["feature_names", "features", "columns"]:
            if k in obj:
                feature_names = obj[k]
                break

        return model_obj, label_encoder, feature_names

    return None, None, None


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, f"File model tidak ditemukan: {MODEL_PATH}"
    try:
        raw = joblib.load(MODEL_PATH)
        model_obj, label_encoder, feature_names = _extract_model_and_extras(raw)

        if model_obj is None:
            keys_info = list(raw.keys()) if isinstance(raw, dict) else str(type(raw))
            return None, None, None, (
                f"File berhasil dimuat tapi tidak ditemukan objek model (yang punya .predict) di dalamnya. "
                f"Isi file: {keys_info}"
            )
        return model_obj, label_encoder, feature_names, None
    except Exception as e:
        return None, None, None, f"Gagal load model: {e}"


model, label_encoder, saved_feature_names, model_error = load_model()

# =========================================================
# UI
# =========================================================
st.title("🌾 Prediksi Kesesuaian Lahan Padi")
st.caption("SoilML - Fuzzy Mamdani + Ensemble ML (Soft Voting)")

if model_error:
    st.error(model_error)
    st.info("Pastikan file `model_sv.pkl` sudah diupload/berada satu folder dengan `app.py` di Codespace ini.")
    st.stop()

st.success("Model berhasil dimuat ✅")

st.subheader("Input 5 Fitur Sensor")

col1, col2 = st.columns(2)
input_values = {}

with col1:
    input_values["suhu_tanah"] = st.number_input(
        FEATURE_LABELS["suhu_tanah"],
        min_value=FEATURE_RANGES["suhu_tanah"][0],
        max_value=FEATURE_RANGES["suhu_tanah"][1],
        value=FEATURE_RANGES["suhu_tanah"][2],
        step=0.1,
    )
    input_values["suhu_udara"] = st.number_input(
        FEATURE_LABELS["suhu_udara"],
        min_value=FEATURE_RANGES["suhu_udara"][0],
        max_value=FEATURE_RANGES["suhu_udara"][1],
        value=FEATURE_RANGES["suhu_udara"][2],
        step=0.1,
    )
    input_values["ph_tanah"] = st.number_input(
        FEATURE_LABELS["ph_tanah"],
        min_value=FEATURE_RANGES["ph_tanah"][0],
        max_value=FEATURE_RANGES["ph_tanah"][1],
        value=FEATURE_RANGES["ph_tanah"][2],
        step=0.1,
    )

with col2:
    input_values["kelembapan_tanah"] = st.number_input(
        FEATURE_LABELS["kelembapan_tanah"],
        min_value=FEATURE_RANGES["kelembapan_tanah"][0],
        max_value=FEATURE_RANGES["kelembapan_tanah"][1],
        value=FEATURE_RANGES["kelembapan_tanah"][2],
        step=0.1,
    )
    input_values["kelembapan_udara"] = st.number_input(
        FEATURE_LABELS["kelembapan_udara"],
        min_value=FEATURE_RANGES["kelembapan_udara"][0],
        max_value=FEATURE_RANGES["kelembapan_udara"][1],
        value=FEATURE_RANGES["kelembapan_udara"][2],
        step=0.1,
    )

st.divider()

if st.button("🔍 Prediksi", type="primary", use_container_width=True):
    # Susun dataframe sesuai urutan fitur training
    X_input = pd.DataFrame([[input_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    try:
        pred = model.predict(X_input)[0]

        # Kalau prediksi masih berupa angka (0,1,2,3) dan ada label_encoder, decode ke S1/S2/S3/N
        decoded = False
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            try:
                pred = label_encoder.inverse_transform([pred])[0]
                decoded = True
            except Exception:
                pass

        # Fallback: kalau belum ke-decode dan hasilnya masih angka, pakai CLASS_INDEX_MAP manual
        if not decoded and isinstance(pred, (int, np.integer, float, np.floating)):
            pred = CLASS_INDEX_MAP.get(int(pred), str(pred))

        label = str(pred)
        desc = LABEL_DESC.get(label, "")

        st.subheader("Hasil Prediksi")
        st.markdown(f"### Kelas: **{label}** - {desc}")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

st.divider()
with st.expander("ℹ️ Info fitur yang dipakai model"):
    st.write(pd.DataFrame({
        "Fitur": [FEATURE_LABELS[f] for f in FEATURE_NAMES],
        "Kolom": FEATURE_NAMES,
    }))
