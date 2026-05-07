import streamlit as st

from utils import (
    leer_matricula,
    gestionar_parking,
    despedida_tts
)

st.title("Parking Inteligente AWS")

uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["png", "jpg", "jpeg"]
)

accion = st.selectbox(
    "Acción",
    ["entrar", "salir"]
)

if uploaded_file:

    matricula = leer_matricula(uploaded_file)

    if matricula == "NO_DETECTADA":
        st.error("No se detectó matrícula")
        st.stop()

    st.success(f"Matrícula: {matricula}")

    mensaje, ok = gestionar_parking(
        matricula,
        accion
    )

    st.write(mensaje)

    if ok:

        audio_path = despedida_tts(
            matricula,
            accion
        )

        st.audio(audio_path)