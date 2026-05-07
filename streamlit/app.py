import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from utils import generar_saludo_voz, mensaje_error_aws, procesar_imagen


st.set_page_config(page_title="Parking Parlanchin", page_icon="P", layout="centered")

st.title("Parking Parlanchin")
st.write("Sube una imagen de la matricula para reconocerla y registrar entrada o salida.")

uploaded_file = st.file_uploader(
    "Imagen de la matricula",
    type=("jpg", "jpeg", "png"),
    accept_multiple_files=False,
)

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Imagen cargada", use_container_width=True)

    if st.button("Reconocer y registrar", type="primary"):
        with st.spinner("Analizando matricula..."):
            try:
                matriculas, evento = procesar_imagen(image_bytes)
            except (BotoCoreError, ClientError, NoCredentialsError) as error:
                st.error(mensaje_error_aws(error))
                st.stop()
            except Exception as error:
                st.error(f"No se pudo procesar la imagen: {error}")
                st.stop()

        if not matriculas or not evento:
            st.warning("No se reconocio ninguna matricula en la imagen.")
            st.stop()

        confianza = matriculas[0].confidence
        st.success(f"Matricula reconocida: {evento.matricula}")
        st.metric("Confianza", f"{confianza:.2f}%")

        if evento.tipo == "entrada":
            st.info("El vehiculo no estaba dentro del parking. Se registro la entrada.")
            st.write(f"Hora de entrada: **{evento.fecha_hora}**")
        else:
            st.info("El vehiculo ya estaba dentro del parking. Se registro la salida.")
            st.write(f"Hora de salida: **{evento.fecha_hora}**")

            if evento.ultimo_evento and evento.ultimo_evento.get("fecha_hora"):
                st.write(f"Ultima entrada: **{evento.ultimo_evento['fecha_hora']}**")

        try:
            audio_saludo = generar_saludo_voz(evento)
            if audio_saludo:
                st.audio(audio_saludo, format="audio/mp3", autoplay=True)
        except (BotoCoreError, ClientError, NoCredentialsError) as error:
            st.warning(f"No se pudo generar el saludo con Polly: {mensaje_error_aws(error)}")

        if len(matriculas) > 1:
            st.caption("Otras posibles matriculas detectadas")
            for matricula in matriculas[1:]:
                st.write(f"- {matricula.text} ({matricula.confidence:.2f}%)")
else:
    st.info("Carga una imagen JPG o PNG para empezar.")
