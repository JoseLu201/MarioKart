import cv2
import numpy as np
import os
import random
import face_recognition as face
import time
from utils import speech_recognizer, load_car, ACTIVE_CAM, PERSONAJES,HABILIDADES, speech_recognizer_thread
from game_menu import  end_menu, ini_menu_minima_interaccion
import json
import gestion_usuarios as gu
import threading



if os.path.exists('camara.py'):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()
    
mtx = camara.cameraMatrix
dist = camara.distCoeffs

# Imagenes de mario y pantalla de victoria
#victory = cv2.imread('aruci/victory.png', cv2.IMREAD_UNCHANGED)

#Le indicamos el diccionario sobre el que vamos a trabajar
DIC = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parametros = cv2.aruco.DetectorParameters()


# Funcion que crea los puntos de un cubo
def create_axis(size):
    return np.float32([ [-size/2, -size/2, 0],
                        [-size/2,  size/2, 0],
                        [ size/2,  size/2, 0],
                        [ size/2, -size/2, 0],
						[-size/2, -size/2, size],
                        [-size/2,  size/2, size],
                        [ size/2,  size/2, size],
                        [ size/2, -size/2, size]])
    


# Dibuajomos un cubo en un las coordenadas de un frame, tambien podemos modificar el 
# tamaño y rotacion de dicho cubo.
# Simulan los cubos del mario Kart
def draw_cube(framerecortado, rvecs, tvecs,coordinates=(0,0,0), scale=1,angle=None):
    axis = create_axis(1)
    axis *= scale
    
    if angle != None:
        angle = np.radians(angle)
        
    # Matriz de rotación en el eje X
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

    # Matriz de rotación en el eje Y
    rotation_matrix_y = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

    # Combinar las matrices de rotación
    combined_rotation_matrix = np.dot(rotation_matrix_x, rotation_matrix_y)

    # Aplicar la rotación a los puntos del cubo
    rotated_axis = np.dot(axis, combined_rotation_matrix)

    # Matriz de rotación en el eje Z
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # Aplicar la rotación a los puntos del cubo
    rotated_axis = np.dot(axis, rotation_matrix)
    
    x, y, z = coordinates

    # Add the coordinate values to the axis points
    # Añadimos las coordenadas a los puntos del eje
    rotated_axis[:, 0] += x
    rotated_axis[:, 1] += y
    rotated_axis[:, 2] += z
    
    # Projectamos dichos puntos sobre el plano
    imgpts, jac = cv2.projectPoints(rotated_axis, rvecs, tvecs, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)   
    
    side1 = framerecortado.copy()
    side2 = framerecortado.copy()
    side3 = framerecortado.copy()
    side4 = framerecortado.copy()
    side5 = framerecortado.copy()
    side6 = framerecortado.copy()
    
    # Draw the bottom side (over the marker)
    side1 = cv2.drawContours(side1, [imgpts[:4]], -1, (0,255,0), -2)
    # Draw the top side (opposite of the marker)
    side2 = cv2.drawContours(side2, [imgpts[4:]], -1, (0,255,0), -2)
    # Draw the right side vertical to the marker
    side3 = cv2.drawContours(side3, [np.array([imgpts[0], imgpts[1], imgpts[5],imgpts[4]])], -1, (0,255,0), -2)
    # Draw the left side vertical to the marker
    side4 = cv2.drawContours(side4, [np.array([imgpts[2], imgpts[3], imgpts[7],imgpts[6]])], -1, (0,255,0), -2)
    # Draw the front side vertical to the marker
    side5 = cv2.drawContours(side5, [np.array([imgpts[1], imgpts[2], imgpts[6],imgpts[5]])], -1, (0,255,0), -2)
    # Draw the back side vertical to the marker
    side6 = cv2.drawContours(side6, [np.array([imgpts[0], imgpts[3], imgpts[7],imgpts[4]])], -1, (0,255,0), -2)
    
    # Pintamos los lados
    framerecortado = cv2.addWeighted(side1, 0.1, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side2, 0.1, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side3, 0.1, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side4, 0.1, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side5, 0.1, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side6, 0.1, framerecortado, 0.9, 0)
    
    # Remarcamos las aristas
    framerecortado = cv2.drawContours(framerecortado, [imgpts[:4]], -1, (0,255,0), 2)
    for i, j in zip(range(4), range(4, 8)):
        framerecortado = cv2.line(framerecortado, tuple(imgpts[i]), tuple(imgpts[j]), (0,255,0), 2)
    framerecortado = cv2.drawContours(framerecortado, [imgpts[4:]], -1, (0,255,0), 2)            
    
    return framerecortado

# Esta funcion es parecida a la anterior, simplemente pintamos como un arco de 3 lados a partir de un cubo
# Simulando los checkpoints del juego
def draw_checkpoint(framerecortado, rvecs, tvecs, scale=2):
    axis = create_axis(1)
    #axis[:, -1] /=2
                
    axis *= scale
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)   
    
    #side1 = framerecortado.copy()
    side2 = framerecortado.copy()
    side3 = framerecortado.copy()
    side4 = framerecortado.copy()

    side2 = cv2.drawContours(side2, [imgpts[4:]], -1, (255, 0, 0), -2)
    # Draw the right side vertical to the marker
    side3 = cv2.drawContours(side3, [np.array([imgpts[0], imgpts[1], imgpts[5],imgpts[4]])], -1, (255, 0, 0), -2)
    # Draw the left side vertical to the marker
    side4 = cv2.drawContours(side4, [np.array([imgpts[2], imgpts[3], imgpts[7],imgpts[6]])], -1, (255, 0, 0), -2)

    framerecortado = cv2.addWeighted(side2, 0.9, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side3, 0.9, framerecortado, 0.9, 0)
    framerecortado = cv2.addWeighted(side4, 0.9, framerecortado, 0.9, 0)
    
    framerecortado = cv2.drawContours(framerecortado, [np.array([imgpts[0], imgpts[1], imgpts[5],imgpts[4]])], -1, (255, 0, 0), 2)
    framerecortado = cv2.drawContours(framerecortado, [np.array([imgpts[2], imgpts[3], imgpts[7],imgpts[6]])], -1, (255, 0, 0), 2)
    framerecortado = cv2.drawContours(framerecortado, [imgpts[4:]], -1, (255, 0, 0), 2)            
    
    return framerecortado

def generar_secuencia_aleatoria(n):
    numeros = list(range(n))
    return random.sample(numeros, len(numeros))
    

# Parte del reconocimiento facial, necesitamos la imagen de referencia del jugador que 
# podra acceder al juego
def init_recognition(reference_image_path, capture_width=640, capture_height=480, max_faces=1):
    # Cargar la cara de referencia
    reference_image = cv2.imread(reference_image_path)
    reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_locations = face.face_locations(reference_rgb, model='hog')
    reference_encodings = face.face_encodings(reference_rgb, reference_locations, model='small')

    # Configurar la captura de video
    cap = cv2.VideoCapture(ACTIVE_CAM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=0)
        face_encodings = face.face_encodings(rgb_frame, face_locations, model='small')

        for (top, right, bottom, left), face_encoding in zip(face_locations[:max_faces], face_encodings[:max_faces]):
            matches = face.compare_faces(reference_encodings, face_encoding)
            #distance = face.face_distance(reference_encodings, face_encoding)
            
            #cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            #cv2.putText(frame, f"Error: {distance[0]:.3f}", (left, bottom + 20), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        
            if any(matches):
                cap.release()
                cv2.destroyAllWindows()
                return True

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

#Funcion que calcula coordenadas del mario del primer plano
def calcular_coordenadas(frame, image):
    hframe, wframe, _ = frame.shape
    himg, wimg, _ = image.shape

    # Calcular las coordenadas x e y para la imagen
    x = (wframe - wimg) // 2
    y = hframe - himg - 5  # Se puede ajustar el valor de desplazamiento vertical

    return x, y

# dibujamos a mario en unas coordenadas x,y eliminando el fondo de la imagen
def draw_mario(frame, image, x, y):
    bg = frame
    fg = image[:, :, 0:3]
    hfg, wfg, _ = fg.shape
    alfa = image[:, :, 3]
    afla = 255 - alfa

    alfa = cv2.cvtColor(alfa, cv2.COLOR_GRAY2BGR) / 255
    afla = cv2.cvtColor(afla, cv2.COLOR_GRAY2BGR) / 255

    mezcla = bg
    mezcla[y:y + hfg, x:x + wfg] = mezcla[y:y + hfg, x:x + wfg] * afla + fg * alfa
    return mezcla

# Funcion que dibuja las lineas con texto, principalmente usada para la distancia al marcador
def draw_line_with_text(frame, point1, point2, text):
    # Dibujar la línea entre los dos puntos
    cv2.line(frame, point1, point2, (0, 255, 0), 2)
    
    # Calcular la posición del texto
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_position = ((point1[0] + point2[0] - text_size[0]) // 2, (point1[1] + point2[1] - text_size[1]) // 2)
    
    # Dibujar el texto encima de la línea
    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

def compute_center(framerecortado, rvecs, tvecs):
    marker_center = np.squeeze(cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), rvecs, tvecs, mtx, dist)[0])
    screen_height, screen_width, _ = framerecortado.shape
    bottom_center = np.array([screen_width // 2, screen_height])

    marker_center = tuple(marker_center.astype(int))
    bottom_center = tuple(bottom_center.astype(int))
    return marker_center, bottom_center


def init_game():
    
    #Menu de jugadores 
    player = ini_menu_minima_interaccion()

    pref = gu.obtener_preferencias(player)
    # Cargamos la imagen del coche con el que se correrá
    car = load_car(pref['personaje'])
    
    # Esto es el orden en el que el jugador tendra que enconrar los checkpoints para acabar el juego.
    marker_order = [0]
    #marker_order = generar_secuencia_aleatoria(3)
    
    current_marker_index = 0  # Índice del marcador actual
    start_time = 0  # Tiempo de inicio del juego
    
    cap = cv2.VideoCapture(ACTIVE_CAM)

    if cap.isOpened():
        hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Tamaño del frame de la cámara: ", wframe, "x", hframe)
    
        matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (wframe,hframe), 1, (wframe,hframe))
        roi_x, roi_y, roi_w, roi_h = roi
        
        final = False
        angle = 0
        rotation_speed = 5
        completed_markers = set()
        duracion_info_inicio = 10
        info = False
        dist_to_score = 10
            
        result_holder = []
        stop_flag = threading.Event()
        voice_thread = threading.Thread(target=speech_recognizer_thread, args=(result_holder,stop_flag))
        voice_thread.start()
         
        #while not "start" in speech_recognizer("Hola " + player + " pronuncia start para comenzar "):
        #   print("Pronuncia start para comenzar")
        
        start_time = time.time()  
        while not final:
            #Leemos el frame en formato BGR
            ret, framebgr = cap.read()
        
            if ret:
                # Aquí procesamos el frame, quitando la distorision
                framerectificado = cv2.undistort(framebgr, mtx, dist, None, matrix)

                #Ajustamos y creamos el frame recortado sin distorsion
                framerecortado = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                
                # Donde ira mario
                x, y = calcular_coordenadas(framerecortado, car)
                
                #Esta parte sera para indicarle al jugador el orden en el que tendra que encontrar los checkpoints.
                # Si pulsa s, skip, empezamos a jugar, sino esperamos a que el tiempo de info acabe.
                tiempo_actual = int(time.time() - start_time)
                if tiempo_actual < duracion_info_inicio and not info :
                    texto = f"Orden: {marker_order}"
                    cv2.putText(framerecortado, texto, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
                    if cv2.waitKey(1) == ord('s'):
                        info = True
                        continue    
                else:
                    info = True 
                    #Detectamos los marcadores del frame recortado            
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(framerecortado, DIC, parameters=parametros)
                                        
                    if len(corners) > 0:
                        
                        alpha = 0.2  # Factor de atenuación del brillo
                        framerecortado = cv2.convertScaleAbs(framerecortado, alpha=alpha, beta=0)
                        
                        for i in range(len(ids)):
                                
                            # Calculamos las rotaciones del marcador respecto de la camara                                 
                            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, mtx, dist)

                            marker_center ,bottom_center = compute_center(framerecortado, rvecs, tvecs)
                            
                            # Obtener la distancia entre la cámara y el marcador ArUco
                            distance = np.linalg.norm(tvecs)
                            dist_text = "Dist.: {:.2f} ".format(distance)
                            framerecortado = draw_line_with_text(framerecortado, marker_center, bottom_center, dist_text)
                            
                            if ids is not None:
                                for i in range(len(ids)):
                                    marker_id = ids[i][0]
                            
                                    # Si el marcador actual coincide con el siguiente en la lista y pasamos por el, puntuamos
                                    if marker_id == marker_order[current_marker_index] and current_marker_index < len(marker_order) and distance < dist_to_score:
                                        # Marcar el marcador como completado
                                        completed_markers.add(marker_id)
                                        current_marker_index = (current_marker_index + 1) % len(marker_order)
                            
                            framerecortado = draw_checkpoint(framerecortado, rvecs, tvecs,3)
                        
                            framerecortado = draw_cube(framerecortado, rvecs, tvecs,(1.0,0.0,0.0),  scale=0.3, angle=angle)
                            framerecortado = draw_cube(framerecortado, rvecs, tvecs,(0.0,0.0,0.0),  scale=0.3, angle=angle)
                            framerecortado = draw_cube(framerecortado, rvecs, tvecs,(-1.0,0.0,0.0), scale=0.3, angle=angle)
                            
                            angle = (angle + rotation_speed)%360

                    framerecortado = draw_mario(framerecortado, car, x, y)
                    
                    # Si completamos el juego pantalla de victoria y tiempo en completarlo
                    if len(completed_markers) == len(marker_order):
                        end_menu(player, 'aruci/victory.png', elapsed_time)
                        final = True
                        stop_flag.set()
                    else:        
                        # Tiempo de jugeo actual y checkpoints visitados
                        cv2.putText(framerecortado, f"Visitados : {completed_markers}", (10, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        elapsed_time = int(time.time() - start_time)
                        cv2.putText(framerecortado, f"Time: {elapsed_time} s", (wframe - 150, hframe- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  

                
                cv2.imshow("RECORTADO", framerecortado)
                
                if result_holder:
                    texto_reconocido = result_holder.pop(0)
                    print("Texto reconocido:", texto_reconocido)
                    if texto_reconocido in PERSONAJES:
                        car = load_car(texto_reconocido)
                        gu.insertar_coche_preferencia(player,texto_reconocido)
                    
                    if "activate" in texto_reconocido.lower():
                        if texto_reconocido.split(' ')[1] in HABILIDADES:
                            print("Activando Habilidad TURBO    ")

                    
                if cv2.waitKey(1) == ord(' '):
                    final = True
                    stop_flag.set()
            else:
                final = True
        voice_thread.join()
        
    else:
        print("No se pudo acceder a la cámara.")

if __name__ == "__main__":      
    init_game()
