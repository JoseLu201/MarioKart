
import pygame
import gestion_usuarios as gu

from utils import speech_recognizer

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Menú de Jugadores")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
added = 4

# Fuente del texto
font = pygame.font.Font(None, 36)


def draw_text(text, x, y):
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(topleft=(x, y))
    window.blit(text_surface, text_rect)

''' 
# def ini_menu():

#     menu_phrases = [
#         "¿Eres un jugador nuevo (Di NEW PLAYER)?",
#         "¿O eres un jugador existente (Di cualquier cosa diferente)?"
#     ]

#     # Calcular el alto total de los rectángulos del menú
#     total_rect_height = len(menu_phrases) * (font.get_linesize() + 10)

#     # Bucle principal del juego
#     running = True
#     while running:
#         # Limpiar la pantalla
#         window.fill(WHITE)

#         # Calcular la posición y el tamaño de cada rectángulo del menú
#         rect_width = window_width - 40
#         rect_height = total_rect_height / len(menu_phrases)
#         rect_x = 20
#         rect_y = (window_height - total_rect_height) / 2

#         # Dibujar los rectángulos y el texto del menú
#         for phrase in menu_phrases:
#             pygame.draw.rect(window, GRAY, pygame.Rect(rect_x, rect_y, rect_width, rect_height))
#             draw_text(phrase, rect_x + 10, rect_y + 5)
#             rect_y += rect_height + 10

#         # Actualizar la pantalla
#         pygame.display.flip()
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             else:
#                 t = speech_recognizer("tell us")
#                 print(t)
#                 if "new player" in t:
#                     print("¡Eres un jugador nuevo!")
#                     running = False
#                     user = gu.registrar_jugador()
#                     break
#                 else:
#                     print("¡Eres un jugador existente!")
#                     running = False
#                     user = gu.reconocer_caras()
#                     print("Hola" + user)
#                     break

#     # Cerrar Pygame
#     pygame.quit()
#     return user
'''

# Esta menu reconocera al jugador, le dara su informacion y rapidamente comenzará el juego
def ini_menu_minima_interaccion():
    menu_phrases = [
        "Bienvenido a Mario Kart"
    ]

    # Calcular el alto total de los rectángulos del menú
    total_rect_height = len(menu_phrases)+added * (font.get_linesize() + 10)

    # Bucle principal del juego
    jugador_detectado = False
    running = True
    while running:
        # Limpiar la pantalla
        window.fill(WHITE)

        # Calcular la posición y el tamaño de cada rectángulo del menú
        rect_width = window_width - 40
        rect_height = total_rect_height / len(menu_phrases)+added
        rect_x = 20
        rect_y = (window_height - total_rect_height) / 2

        # Dibujar los rectángulos y el texto del menú
        for phrase in menu_phrases:
            pygame.draw.rect(window, GRAY, pygame.Rect(
                rect_x, rect_y, rect_width, rect_height))
            draw_text(phrase, rect_x + 10, rect_y + 5)
            rect_y += rect_height + 10
        pygame.display.flip()

        if not jugador_detectado:
            user = gu.reconocer_caras()
            if user is None:
                print("¡Eres un jugador nuevo!")
                running = False
                user = gu.registrar_jugador()

            else:
                print("¡Eres un jugador existente!")
                print("Hola " + user)
                print("Mostrando datos...")
                carac = gu.obtener_caracteristicas_jugador(user)
                menu_phrases.append("Jugador: " + str(carac['nombre']))
                menu_phrases.append("Personaje: " + str(carac['preferencias']['personaje']))
                menu_phrases.append("Historial de tiempos: " + str(gu.obtener_historial(user)))
                pygame.display.flip()
            jugador_detectado = True
            menu_phrases.append("Espacio para comenzar")

        # Actualizar la pantalla

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False

    # Cerrar Pygame
    pygame.quit()
    return user


def end_menu(player, img, elapsed_time):
    # Inicializar Pygame
    pygame.init()

    # Configuración de la ventana
    window_width = 800
    window_height = 600
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Ventana de Victoria")

    # Cargar la imagen de fondo
    background_image = pygame.image.load(img)
    background_image = pygame.transform.scale(
        background_image, (window_width, window_height))

    # Fuente del texto
    font = pygame.font.Font(None, 48)

    # Bucle principal del juego
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False

        seconds = elapsed_time

        # Limpiar la pantalla
        window.fill(WHITE)

        # Dibujar la imagen de fondo
        window.blit(background_image, (0, 0))

        # Dibujar el texto del temporizador
        text = font.render(f"Tiempo: {int(seconds)} segundos", True, BLACK)
        text_rect = text.get_rect(
            center=(window_width // 2, window_height - 30))
        window.blit(text, text_rect)

        # Actualizar la pantalla
        pygame.display.flip()

    # Actualizo la base de datos de tiempo del jugador
    print("Guardando resultado...")
    gu.insertar_tiempo(player, elapsed_time)
    # Cerrar
    print("Resultado guardado con exito")
    pygame.quit()
