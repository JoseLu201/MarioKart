
import threading
import speech_recognition as sr
import cv2

ACTIVE_CAM = 0

# Lista de los personajes disponibles
PERSONAJES = ["Mario", "Luigi"]


def speech_recognizer(text):
    rec = sr.Recognizer()
    mic = sr.Microphone()
    print(text)
    with mic as source:
        rec.adjust_for_ambient_noise(source, duration=0.5)
        audio = rec.listen(source)
    texto = None
    texto = rec.recognize_google(audio)

    return texto


def speech_recognizer_thread(result_holder, stop_flag):
    rec = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        rec.adjust_for_ambient_noise(source, duration=0.5)
        print("Ajuste de ruido completado")

    while not stop_flag.is_set():
        with mic as source:
            audio = rec.listen(source, timeout=3)

        try:
            texto = rec.recognize_google(audio)
            result_holder.append(texto)
        except sr.UnknownValueError:
            pass


def load_car(pref):
    car = cv2.imread('aruci/'+pref+'bg.png', cv2.IMREAD_UNCHANGED)
    car = cv2.resize(car, None, fx=0.3, fy=0.3)
    return car

# from PIL import Image
#
# def redimensionar_imagen(img1, img2):
#    with Image.open(img1) as imagen1:
#        width1, height1 = imagen1.size
#
#    with Image.open(img2) as imagen2:
#        imagen2_resized = imagen2.resize((width1, height1))
#
#    # Guardar la imagen redimensionada en un archivo
#    imagen2_resized.save("imagen2_resized.png")
