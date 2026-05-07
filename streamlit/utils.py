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
    re.compile(r"\b[A-Z]{1,3}[A-Z]{1,2}\d{1,4}\b"),
    re.compile(r"\b[A-Z]{2}\d{3}[A-Z]{2}\b"),
    re.compile(r"\b[A-Z]{1,3}\d{3,4}[A-Z]{0,3}\b"),
)
MIN_PLATE_LENGTH = 5
MAX_PLATE_LENGTH = 9
MIN_PLATE_ASPECT_RATIO = 2.0
MAX_PLATE_ASPECT_RATIO = 12.0
MAX_WORD_ANGLE_DIFF = 12.0
MAX_WORD_HEIGHT_DIFF_RATIO = 0.6
MAX_WORD_GAP_FACTOR = 4.5
MAX_WORD_VERTICAL_FACTOR = 1.4
PARTIAL_PLATE_CONFIDENCE_MARGIN = 5.0

# Las clases de datos representan las detecciones de texto y los eventos de parking.
# Class TextDetection representa un texto detectado en la imagen, con su confianza y características geométricas.
@dataclass(frozen=True)
class TextDetection:
    text: str
    confidence: float
    area: float = 0
    aspect_ratio: float = 0

@dataclass(frozen=True)
class TextPart:
    text: str
    confidence: float
    area: float
    aspect_ratio: float
    left: float
    top: float
    width: float
    height: float
    center_x: float
    center_y: float
    angle: float
    axis_length: float
    cross_length: float

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

def normalizar_angulo(angle: float) -> float:
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle

def diferencia_angulo(first: float, second: float) -> float:
    diff = abs(first - second) % 180
    return min(diff, 180 - diff)

def puntos_poligono(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(point.get("X", 0)), float(point.get("Y", 0)))
        for point in geometry.get("Polygon", [])[:4]
    ]

def calcular_aspect_ratio(geometry: dict[str, Any]) -> float:
    points = puntos_poligono(geometry)
    if len(points) >= 4:
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

def extraer_parte_texto(item: dict[str, Any]) -> TextPart | None:
    text = limpiar_texto_matricula(item.get("DetectedText", ""))
    if not text:
        return None

    geometry = item.get("Geometry", {})
    box = geometry.get("BoundingBox", {})
    left = float(box.get("Left", 0))
    top = float(box.get("Top", 0))
    width = float(box.get("Width", 0))
    height = float(box.get("Height", 0))
    points = puntos_poligono(geometry)
    center_x = left + (width / 2)
    center_y = top + (height / 2)
    angle = 0.0
    axis_length = width
    cross_length = height

    if len(points) >= 4:
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        edges = [
            math.dist(points[index], points[(index + 1) % 4])
            for index in range(4)
        ]
        axis_length = sum(sorted(edges)[-2:]) / 2
        cross_length = sum(sorted(edges)[:2]) / 2
        angle = normalizar_angulo(
            math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
        )

    return TextPart(
        text=text,
        confidence=float(item.get("Confidence", 0)),
        area=width * height,
        aspect_ratio=calcular_aspect_ratio(geometry),
        left=left,
        top=top,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
        angle=angle,
        axis_length=axis_length,
        cross_length=cross_length,
    )

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

def proyectar_punto(part: TextPart, angle: float) -> tuple[float, float]:
    radians = math.radians(angle)
    axis_x = math.cos(radians)
    axis_y = math.sin(radians)
    cross_x = -axis_y
    cross_y = axis_x
    axis_position = (part.center_x * axis_x) + (part.center_y * axis_y)
    cross_position = (part.center_x * cross_x) + (part.center_y * cross_y)
    return axis_position, cross_position

def crear_deteccion_agrupada(parts: list[TextPart]) -> TextDetection:
    ordered_parts = sorted(parts, key=lambda part: proyectar_punto(part, parts[0].angle)[0])
    angle = sum(part.angle for part in ordered_parts) / len(ordered_parts)
    axis_spans = []
    cross_lengths = []

    for part in ordered_parts:
        axis_position, _ = proyectar_punto(part, angle)
        axis_spans.append((axis_position - (part.axis_length / 2), axis_position + (part.axis_length / 2)))
        cross_lengths.append(part.cross_length or part.height)

    axis_length = max(end for _, end in axis_spans) - min(start for start, _ in axis_spans)
    cross_length = sum(cross_lengths) / len(cross_lengths)
    aspect_ratio = axis_length / cross_length if cross_length else 0
    confidence = sum(part.confidence for part in ordered_parts) / len(ordered_parts)

    return TextDetection(
        text="".join(part.text for part in ordered_parts),
        confidence=confidence,
        area=axis_length * cross_length,
        aspect_ratio=aspect_ratio,
    )

def es_candidato_palabra(part: TextPart) -> bool:
    return len(part.text) > 1 or any(char.isdigit() for char in part.text)

def estan_en_la_misma_linea(group: list[TextPart], candidate: TextPart) -> bool:
    angle = sum(part.angle for part in group) / len(group)
    avg_height = sum(part.cross_length or part.height for part in group) / len(group)
    candidate_height = candidate.cross_length or candidate.height

    if diferencia_angulo(angle, candidate.angle) > MAX_WORD_ANGLE_DIFF:
        return False

    height_diff_ratio = abs(candidate_height - avg_height) / avg_height if avg_height else 1
    if height_diff_ratio > MAX_WORD_HEIGHT_DIFF_RATIO:
        return False

    group_cross_positions = [proyectar_punto(part, angle)[1] for part in group]
    _, candidate_cross = proyectar_punto(candidate, angle)
    vertical_distance = abs(candidate_cross - (sum(group_cross_positions) / len(group_cross_positions)))
    if vertical_distance > avg_height * MAX_WORD_VERTICAL_FACTOR:
        return False

    ordered_group = sorted(group, key=lambda part: proyectar_punto(part, angle)[0])
    last = ordered_group[-1]
    last_axis = proyectar_punto(last, angle)[0]
    candidate_axis = proyectar_punto(candidate, angle)[0]
    gap = candidate_axis - (last_axis + (last.axis_length / 2))
    max_gap = avg_height * MAX_WORD_GAP_FACTOR
    return gap <= max_gap

def agrupar_palabras_matricula(parts: list[TextPart]) -> list[TextDetection]:
    words = sorted(
        [part for part in parts if es_candidato_palabra(part)],
        key=lambda part: (part.center_y, part.center_x),
    )
    grouped_detections: list[TextDetection] = []

    for start_index, word in enumerate(words):
        group = [word]
        following_words = sorted(words[start_index + 1 :], key=lambda part: proyectar_punto(part, word.angle)[0])

        for candidate in following_words:
            if not estan_en_la_misma_linea(group, candidate):
                continue

            group.append(candidate)
            if len(group) >= 2:
                grouped_detections.append(crear_deteccion_agrupada(group))

    return grouped_detections

def es_subsecuencia(texto_corto: str, texto_largo: str) -> bool:
    if len(texto_corto) >= len(texto_largo):
        return False

    iterator = iter(texto_largo)
    return all(char in iterator for char in texto_corto)

def es_matricula_parcial(detection: TextDetection, complete_detection: TextDetection) -> bool:
    return (
        es_subsecuencia(detection.text, complete_detection.text)
        and complete_detection.confidence >= detection.confidence - PARTIAL_PLATE_CONFIDENCE_MARGIN
    )

def quitar_matriculas_parciales(detections: list[TextDetection]) -> list[TextDetection]:
    filtered_detections: list[TextDetection] = []

    for detection in detections:
        is_partial_plate = any(
            es_matricula_parcial(detection, complete_detection)
            for complete_detection in detections
            if complete_detection is not detection
        )
        if not is_partial_plate:
            filtered_detections.append(detection)

    return filtered_detections

# Funcion para detectar textos en una imagen utilizando AWS Rekognition, filtrando por lineas y extrayendo caracteristicas relevantes
def detectar_textos(image_bytes: bytes) -> list[TextDetection]:
    rekognition = crear_cliente("rekognition")
    response = rekognition.detect_text(Image={"Bytes": image_bytes})

    detections: list[TextDetection] = []
    word_parts: list[TextPart] = []
    for item in response.get("TextDetections", []):
        text_part = extraer_parte_texto(item)
        if not text_part:
            continue

        if item.get("Type") == "LINE":
            detections.append(
                TextDetection(
                    text=text_part.text,
                    confidence=text_part.confidence,
                    area=text_part.area,
                    aspect_ratio=text_part.aspect_ratio,
                )
            )
        elif item.get("Type") == "WORD":
            word_parts.append(text_part)

    return detections + agrupar_palabras_matricula(word_parts)

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

    filtered_plates = quitar_matriculas_parciales(list(plates.values()))

    return sorted(
        filtered_plates,
        key=lambda detection: (detection.confidence, detection.area, len(detection.text)),
        reverse=True,
    )

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
