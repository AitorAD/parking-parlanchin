from __future__ import annotations
import configparser
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


# Configuracion global y utilidades para el reconocimiento de matriculas y registro de eventos en DynamoDB.
ROOT_DIR = Path(__file__).resolve().parents[1]
CREDENTIALS_PATH = ROOT_DIR / ".aws" / "credentials"
AWS_REGION = "us-east-1"
TABLE_NAME = "RegistroParking"
TIMEZONE = ZoneInfo("Europe/Madrid")
POLLY_LANGUAGE_CODE = "es-ES"
POLLY_VOICES = (
    {"VoiceId": "Lucia", "Engine": "neural"},
    {"VoiceId": "Conchita", "Engine": "standard"},
)
# Expresiones regulares para detectar formatos comunes de matriculas españolas.
PLATE_PATTERNS = (
    re.compile(r"\b\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}\b"),
    re.compile(r"\b[A-Z]{1,3}\d{3,4}[A-Z]{0,3}\b"),
)
MIN_PLATE_LENGTH = 5
MAX_PLATE_LENGTH = 9
MIN_PLATE_ASPECT_RATIO = 2.0
MAX_PLATE_ASPECT_RATIO = 12.0

# Las clases de datos representan las detecciones de texto y los eventos de parking.
# Class TextDetection representa un texto detectado en la imagen, con su confianza y características geométricas.
@dataclass(frozen=True)
class TextDetection:
    text: str
    confidence: float
    area: float = 0
    aspect_ratio: float = 0

# Class ParkingEvent representa un evento de entrada o salida en el parking,
# con detalles sobre la matricula, tipo de evento, fecha y hora, y si el vehículo estaba dentro del parking.
@dataclass(frozen=True)
class ParkingEvent:
    matricula: str
    tipo: str
    fecha_hora: str
    estaba_dentro: bool
    ultimo_evento: dict[str, Any] | None

# Las funciones a continuación implementan la lógica para interactuar con AWS, procesar imágenes, detectar matriculas,
# Fuencion para leer credenciales de AWS
def _read_aws_credentials() -> dict[str, str | None]:
    if not CREDENTIALS_PATH.exists():
        return {}

    aws_config = configparser.ConfigParser()
    aws_config.read(CREDENTIALS_PATH)

    if not aws_config.sections():
        return {}

    profile = aws_config.sections()[0]
    return {
        "aws_access_key_id": aws_config[profile].get("aws_access_key_id"),
        "aws_secret_access_key": aws_config[profile].get("aws_secret_access_key"),
        "aws_session_token": aws_config[profile].get("aws_session_token"),
    }

# Fuencion para crear un cliente de AWS para un servicio específico
def crear_cliente(servicio: str):
    credentials = _read_aws_credentials()
    kwargs: dict[str, Any] = {"region_name": AWS_REGION}

    if credentials:
        kwargs.update(credentials)

    return boto3.client(servicio, **kwargs)

# funcion para normalizar texto eliminando caracteres no alfanumericos y convirtiendo a mayusculas
def normalizar_texto(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", texto.upper())

# Funcion para limpiar el texto de la matricula eliminando espacios y guiones, y convirtiendo a mayusculas
def limpiar_texto_matricula(texto: str) -> str:
    return texto.upper().replace(" ", "").replace("-", "")

def calcular_aspect_ratio(geometry: dict[str, Any]) -> float:
    polygon = geometry.get("Polygon", [])
    if len(polygon) >= 4:
        points = [
            (float(point.get("X", 0)), float(point.get("Y", 0)))
            for point in polygon[:4]
        ]
        edges = [
            math.dist(points[index], points[(index + 1) % 4])
            for index in range(4)
        ]
        short_edges = sorted(edges)[:2]
        long_edges = sorted(edges)[-2:]
        avg_short_edge = sum(short_edges) / len(short_edges)
        avg_long_edge = sum(long_edges) / len(long_edges)

        if avg_short_edge:
            return avg_long_edge / avg_short_edge

    box = geometry.get("BoundingBox", {})
    width = float(box.get("Width", 0))
    height = float(box.get("Height", 0))
    return width / height if height else 0

# Funcion para verificar si un texto tiene el formato de una matricula valida, considerando longitud, caracteres y proporcion
def tiene_formato_matricula(texto: str, aspect_ratio: float) -> bool:
    if not MIN_PLATE_LENGTH <= len(texto) <= MAX_PLATE_LENGTH:
        return False

    if not texto.isalnum():
        return False

    if not any(char.isdigit() for char in texto):
        return False

    if not any(char.isalpha() for char in texto):
        return False

    return MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO

# Funcion para extraer una matricula de un texto dado, verificando su formato y aplicando patrones comunes
def extraer_matricula(texto: str, aspect_ratio: float = 0) -> str | None:
    texto_limpio = limpiar_texto_matricula(texto)

    if not tiene_formato_matricula(texto_limpio, aspect_ratio):
        return None

    for pattern in PLATE_PATTERNS:
        match = pattern.search(texto_limpio)
        if match:
            return match.group(0)

    return texto_limpio

# Funcion para detectar textos en una imagen utilizando AWS Rekognition, filtrando por lineas y extrayendo caracteristicas relevantes
def detectar_textos(image_bytes: bytes) -> list[TextDetection]:
    rekognition = crear_cliente("rekognition")
    response = rekognition.detect_text(Image={"Bytes": image_bytes})

    detections: list[TextDetection] = []
    for item in response.get("TextDetections", []):
        if item.get("Type") != "LINE":
            continue

        text = limpiar_texto_matricula(item.get("DetectedText", ""))
        geometry = item.get("Geometry", {})
        box = geometry.get("BoundingBox", {})
        width = float(box.get("Width", 0))
        height = float(box.get("Height", 0))
        area = width * height
        aspect_ratio = calcular_aspect_ratio(geometry)

        if text:
            detections.append(
                TextDetection(
                    text=text,
                    confidence=float(item.get("Confidence", 0)),
                    area=area,
                    aspect_ratio=aspect_ratio,
                )
            )

    return detections

# Funcion para detectar matriculas en una imagen, filtrando las detecciones de texto por formato de matricula y ordenando por area
def detectar_matriculas(image_bytes: bytes) -> list[TextDetection]:
    detections = detectar_textos(image_bytes)
    plates: dict[str, TextDetection] = {}

    for detection in detections:
        plate = extraer_matricula(detection.text, detection.aspect_ratio)
        if not plate:
            continue

        current = plates.get(plate)
        if not current or detection.area > current.area:
            plates[plate] = TextDetection(
                text=plate,
                confidence=detection.confidence,
                area=detection.area,
                aspect_ratio=detection.aspect_ratio,
            )

    return sorted(plates.values(), key=lambda detection: detection.area, reverse=True)

# funcion para crear la tabla de DynamoDB si no existe, definiendo su esquema y esperando a que esté disponible antes de continuar
def crear_tabla_si_no_existe() -> None:
    dynamodb = crear_cliente("dynamodb")

    try:
        dynamodb.describe_table(TableName=TABLE_NAME)
        return
    except dynamodb.exceptions.ResourceNotFoundException:
        pass

    dynamodb.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "matricula", "KeyType": "HASH"},
            {"AttributeName": "fecha_hora", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "matricula", "AttributeType": "S"},
            {"AttributeName": "fecha_hora", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    waiter = dynamodb.get_waiter("table_exists")
    waiter.wait(TableName=TABLE_NAME)

# Funcion para obtener el ultimo evento registrado para una matricula dada, consultando DynamoDB y devolviendo un diccionario con los detalles del evento o None si no hay eventos registrados
def obtener_ultimo_evento(matricula: str) -> dict[str, Any] | None:
    dynamodb = crear_cliente("dynamodb")
    response = dynamodb.query(
        TableName=TABLE_NAME,
        KeyConditionExpression="matricula = :matricula",
        ExpressionAttributeValues={":matricula": {"S": matricula}},
        ScanIndexForward=False,
        Limit=1,
    )

    items = response.get("Items", [])
    if not items:
        return None

    item = items[0]
    return {
        "matricula": item.get("matricula", {}).get("S", matricula),
        "fecha_hora": item.get("fecha_hora", {}).get("S"),
        "tipo": item.get("tipo", {}).get("S", "entrada"),
    }

# esta funcion verifica si una matricula dada tiene un evento de entrada registrado sin una salida posterior, 
# indicando que el vehículo está actualmente dentro del parking
def esta_dentro(matricula: str) -> bool:
    ultimo_evento = obtener_ultimo_evento(matricula)
    return bool(ultimo_evento and ultimo_evento.get("tipo") == "entrada")

# Funcion para registrar una entrada o salida en el parking,
# determinando el tipo de evento basado en el ultimo evento registrado para la matricula y guardando el nuevo evento en DynamoDB
def registrar_entrada_o_salida(matricula: str) -> ParkingEvent:
    crear_tabla_si_no_existe()

    ultimo_evento = obtener_ultimo_evento(matricula)
    estaba_dentro = bool(ultimo_evento and ultimo_evento.get("tipo") == "entrada")
    tipo = "salida" if estaba_dentro else "entrada"
    ahora = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    dynamodb = crear_cliente("dynamodb")
    dynamodb.put_item(
        TableName=TABLE_NAME,
        Item={
            "matricula": {"S": matricula},
            "fecha_hora": {"S": ahora},
            "tipo": {"S": tipo},
        },
    )

    return ParkingEvent(
        matricula=matricula,
        tipo=tipo,
        fecha_hora=ahora,
        estaba_dentro=estaba_dentro,
        ultimo_evento=ultimo_evento,
    )

# esta función convierte una matricula en un formato adecuado para ser leido por Polly,
# separando cada caracter con un espacio para mejorar la pronunciación
def matricula_para_voz(matricula: str) -> str:
    return " ".join(matricula)

# funcion para  crear el texto del saludo
def crear_texto_saludo(evento: ParkingEvent) -> str:
    hora = evento.fecha_hora.split(" ")[-1][:5]
    matricula = matricula_para_voz(evento.matricula)

    if evento.tipo == "entrada":
        return (
            "Bienvenido al Parking Parlanchin. "
            f"Matricula {matricula}. "
            f"Entrada registrada a las {hora}."
        )

    return (
        "Gracias por visitar el Parking Parlanchin. "
        f"Matricula {matricula}. "
        f"Salida registrada a las {hora}. Buen viaje."
    )

# funcion para crear el audio del saludo
def generar_saludo_voz(evento: ParkingEvent) -> bytes:
    polly = crear_cliente("polly")
    texto = crear_texto_saludo(evento)
    ultimo_error: ClientError | None = None

    for voice in POLLY_VOICES:
        try:
            response = polly.synthesize_speech(
                Text=texto,
                OutputFormat="mp3",
                LanguageCode=POLLY_LANGUAGE_CODE,
                **voice,
            )
            audio_stream = response.get("AudioStream")
            return audio_stream.read() if audio_stream else b""
        except ClientError as error:
            ultimo_error = error

    if ultimo_error:
        raise ultimo_error

    return b""

# funcion para procesar una imagen y registrar el evento  ya sea  entrada o salida
def procesar_imagen(image_bytes: bytes) -> tuple[list[TextDetection], ParkingEvent | None]:
    matriculas = detectar_matriculas(image_bytes)
    if not matriculas:
        return matriculas, None

    evento = registrar_entrada_o_salida(matriculas[0].text)
    return matriculas, evento

# Funcion para  generar un mensaje de error segun la excepcion de AWS,
# generalmente para problemas de credenciales o errores en los servicios, devolviendo un mensaje amigable para el usuario
def mensaje_error_aws(error: Exception) -> str:
    if isinstance(error, NoCredentialsError):
        return "No se encontraron credenciales de AWS."

    if isinstance(error, ClientError):
        detail = error.response.get("Error", {})
        code = detail.get("Code", "ClientError")
        message = detail.get("Message", str(error))
        return f"AWS devolvio {code}: {message}"

    if isinstance(error, BotoCoreError):
        return f"No se pudo conectar con AWS: {error}"

    return str(error)
