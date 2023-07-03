import json
import cv2
import numpy as np
import face_recognition as face
from utils import speech_recognizer, PERSONAJES, ACTIVE_CAM

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.default(self, obj)


def obtener_face_encoding():
    # Configurar la captura de video
    cap = cv2.VideoCapture(ACTIVE_CAM)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        reference_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reference_locations = face.face_locations(reference_rgb, model='hog')
        reference_encodings = face.face_encodings(reference_rgb, reference_locations, model='small')

        if len(reference_encodings) > 0:            
            # Detener la captura de video y cerrar la ventana
            cap.release()
            cv2.destroyAllWindows()
            
            return reference_encodings

        # Mostrar el frame con las caras detectadas
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

    
def estructurar_jugador(nombre, preferencias,history, encoding_cara):
    jugador = {
        "nombre": nombre,
        "preferencias": preferencias,
        "historial": history,
        "encoding_cara": encoding_cara
    }
    return jugador  


def guardar_jugador(jugador):

    # Cargar los datos existentes del archivo JSON si existe
    try:
        with open("jugadores.json", 'r') as archivo:
            datos_jugadores = json.load(archivo)
    except FileNotFoundError:
        datos_jugadores = {}
        
    # Agregar el nuevo jugador a los datos existentes
    datos_jugadores.append(jugador)
        
    with open("jugadores.json", 'w') as archivo:
        json.dump(datos_jugadores, archivo, indent=4, cls=NumpyArrayEncoder)        


def registrar_jugador():
    face_enco = obtener_face_encoding()

    nombre = speech_recognizer("Dime tu nombre.") 
    
    personaje = None
    while personaje not in PERSONAJES:
        personaje = speech_recognizer("Que personaje quieres jugar " + str(PERSONAJES))    
    preferencias = {
        "idioma" : "sp",
        "personaje" : personaje
    }
    history = []
    jugador = estructurar_jugador(nombre,preferencias,history, face_enco)
    guardar_jugador(jugador)
    return nombre
    

def cargar_jugadores():
    try:
        with open("jugadores.json", 'r') as archivo:
            datos_jugadores = json.load(archivo)
            return datos_jugadores
    except FileNotFoundError:
        return []
    
def obtener_encodings_jugadores(jugadores):
    encodings_jugadores = []
    for jugador in jugadores:
        encodings_jugadores.append(np.array(jugador["encoding_cara"]))
    return encodings_jugadores


def reconocer_caras():
    cap = cv2.VideoCapture(ACTIVE_CAM)

    # Cargar los jugadores y sus encodings
    jugadores = cargar_jugadores()
    encodings_jugadores = obtener_encodings_jugadores(jugadores)
    
    
    nombre_jugador = None  # Inicializar como None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face.face_locations(rgb_frame, model='hog')
        face_encodings = face.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings) > 0:
            print("Reconociendo caras...")
            
            for face_encoding in face_encodings:
                # Comparar el encoding de la cara detectada con los encodings de los jugadores
                for i, encoding_jugador in enumerate(encodings_jugadores):
                    resultados_comparacion = face.compare_faces(encoding_jugador, face_encoding)
                    if True in resultados_comparacion:
                        nombre_jugador = jugadores[i]["nombre"]
                        return nombre_jugador # Salir del bucle interno si se encuentra un nombre
                if nombre_jugador is not None:
                    return nombre_jugador
            
            if nombre_jugador is None:
                print("La cara detectada no coincide con ningún jugador.")
                break
        else:
            print("No se han detectado caras.")

        
    cap.release()
    cv2.destroyAllWindows()
    return nombre_jugador


def obtener_preferencias(nombre):
    try:
        with open("jugadores.json", 'r') as archivo:
            datos_jugadores = json.load(archivo)
            
        for jugador in datos_jugadores:
            if jugador["nombre"] == nombre:
                return jugador["preferencias"]
        
        print("No se encontraron preferencias para el jugador:", nombre)
        return None
    
    except FileNotFoundError:
        print("No se encontró el archivo de jugadores.")
        return None
    
def obtener_historial(nombre):
    try:
        with open("jugadores.json", 'r') as archivo:
            datos_jugadores = json.load(archivo)
            
        for jugador in datos_jugadores:
            if jugador["nombre"] == nombre:
                return jugador["historial"]
        
        print("No se encontraron preferencias para el jugador:", nombre)
        return None
    
    except FileNotFoundError:
        print("No se encontró el archivo de jugadores.")
        return None
    
def insertar_tiempo(nombre, tiempo):
        with open("jugadores.json", 'r') as archivo:
            datos_jugadores = json.load(archivo)
        
        for jugador in datos_jugadores:
            if jugador["nombre"] == nombre:
                jugador["historial"].append(tiempo)
                break
            
        
        with open("jugadores.json", 'w') as archivo:
            json.dump(datos_jugadores, archivo, indent=4, cls=NumpyArrayEncoder)
        
    
def obtener_caracteristicas_jugador(nombre_buscar):
    jugadores = cargar_jugadores()
    for jugador in jugadores:
        if jugador["nombre"] == nombre_buscar:
            return jugador
    return None  # Si el jugador no se encuentra, devuelve None