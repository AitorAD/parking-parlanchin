import configparser
import os
import re
import sqlite3
import boto3

# =========================
# AWS CONFIG
# =========================

CREDENTIALS_PATH = "../.aws/credentials"

if not os.path.exists(CREDENTIALS_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo de credenciales en: {os.path.abspath(CREDENTIALS_PATH)}"
    )

AWS_REGION = "us-east-1"

aws_config = configparser.ConfigParser()
aws_config.read(CREDENTIALS_PATH)

PROFILE = aws_config.sections()[0]

AWS_ACCESS_KEY = aws_config[PROFILE].get("aws_access_key_id")
AWS_SECRET_KEY = aws_config[PROFILE].get("aws_secret_access_key")
AWS_SESSION_TOKEN = aws_config[PROFILE].get("aws_session_token", None)

def crear_cliente(servicio):
    return boto3.client(
        servicio,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_REGION
    )

# =========================
# CLIENTES AWS
# =========================

rekognition = crear_cliente("rekognition")
polly = crear_cliente("polly")
dynamodb = crear_cliente("dynamodb")

# =========================
# DYNAMO DB
# =========================

TABLE_NAME = "parking"

def crear_tabla_si_no_existe():

    tablas = dynamodb.list_tables()["TableNames"]

    if TABLE_NAME not in tablas:

        dynamodb.create_table(
            TableName=TABLE_NAME,

            KeySchema=[
                {
                    'AttributeName': 'matricula',
                    'KeyType': 'HASH'
                }
            ],

            AttributeDefinitions=[
                {
                    'AttributeName': 'matricula',
                    'AttributeType': 'S'
                }
            ],

            BillingMode='PAY_PER_REQUEST'
        )

        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=TABLE_NAME)

        print("Tabla creada")

crear_tabla_si_no_existe()

# =========================
# OCR
# =========================

def leer_matricula(imagen):

    image_bytes = imagen.getvalue()

    try:

        response = rekognition.detect_text(
            Image={
                'Bytes': image_bytes
            }
        )

        print(response)

    except Exception as e:

        import traceback

        print("========== ERROR AWS ==========")
        print(type(e))
        print(e)
        traceback.print_exc()

        raise

    texto_total = ""

    for texto in response["TextDetections"]:

        if texto["Type"] == "LINE":
            texto_total += texto["DetectedText"] + " "

    matricula = extraer_matricula_valida(texto_total)

    return matricula if matricula else "NO_DETECTADA"

# =========================
# REGEX MATRÍCULAS
# =========================

def extraer_matricula_valida(texto):

    texto = re.sub(r'[^A-Z0-9]', '', texto.upper())

    patrones = [
        r'\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}',
        r'[A-Z]{2}\d{2}[A-Z]{3}',
        r'[A-Z]{2}\d{3}[A-Z]{2}',
        r'[A-Z]{2}\d{2}[A-Z]{2}',
    ]

    for patron in patrones:
        match = re.search(patron, texto)

        if match:
            return match.group()

    return None

# =========================
# PARKING
# =========================

def get_estado(matricula):

    response = dynamodb.get_item(
        TableName=TABLE_NAME,
        Key={
            'matricula': {'S': matricula}
        }
    )

    item = response.get("Item")

    if item:
        return item["estado"]["S"]

    return "fuera"

def actualizar_estado(matricula, nuevo_estado):

    dynamodb.put_item(
        TableName=TABLE_NAME,
        Item={
            'matricula': {'S': matricula},
            'estado': {'S': nuevo_estado}
        }
    )

def gestionar_parking(matricula, accion):

    estado = get_estado(matricula)

    if estado == "dentro" and accion == "entrar":
        return "Ya estás dentro", False

    elif estado == "dentro" and accion == "salir":
        actualizar_estado(matricula, "fuera")
        return "Salida permitida", True

    elif estado == "fuera" and accion == "entrar":
        actualizar_estado(matricula, "dentro")
        return "Entrada permitida", True

    else:
        return "No puedes salir si no has entrado", False

# =========================
# DETECTAR PAÍS
# =========================

def detectar_pais_matricula(matricula):

    if re.match(r'^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$', matricula):
        return "es"

    elif re.match(r'^[A-Z]{2}\d{2}[A-Z]{3}$', matricula):
        return "uk"

    return "unknown"

# =========================
# TTS AWS POLLY
# =========================

def despedida_tts(matricula, accion):

    pais = detectar_pais_matricula(matricula)

    mensajes = {
        "es": {
            "entrar": "Entrada permitida. Bienvenido",
            "salir": "Salida permitida. Hasta luego"
        },
        "uk": {
            "entrar": "Entry allowed. Welcome",
            "salir": "Exit allowed. Goodbye"
        }
    }

    voces = {
        "es": "Lucia",
        "uk": "Amy"
    }

    texto = mensajes.get(pais, mensajes["uk"])[accion]
    voz = voces.get(pais, "Amy")

    response = polly.synthesize_speech(
        Text=texto,
        OutputFormat="mp3",
        VoiceId=voz
    )

    audio_path = os.path.join(
        os.path.dirname(__file__),
        "despedida.mp3"
    )

    with open(audio_path, "wb") as file:
        file.write(response["AudioStream"].read())

    return audio_path