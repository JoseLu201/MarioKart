import pygame
import sys
from utils import speech_recognizer
import gestion_usuarios as gu

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

# Fuente del texto
font = pygame.font.Font(None, 36)

def draw_text(text, x, y):
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(topleft=(x, y))
    window.blit(text_surface, text_rect)

# Frases del menú
    def menu():
    menu_phrases = [
        "¿Eres un jugador nuevo (N)?",
        "¿O eres un jugador existente (E)?"
    ]

    # Calcular el alto total de los rectángulos del menú
    total_rect_height = len(menu_phrases) * (font.get_linesize() + 10)

    # Bucle principal del juego
    running = True
    while running:
        

        # Limpiar la pantalla
        window.fill(WHITE)

        # Calcular la posición y el tamaño de cada rectángulo del menú
        rect_width = window_width - 40
        rect_height = total_rect_height / len(menu_phrases)
        rect_x = 20
        rect_y = (window_height - total_rect_height) / 2

        # Dibujar los rectángulos y el texto del menú
        for phrase in menu_phrases:
            pygame.draw.rect(window, GRAY, pygame.Rect(rect_x, rect_y, rect_width, rect_height))
            draw_text(phrase, rect_x + 10, rect_y + 5)
            rect_y += rect_height + 10

        # Actualizar la pantalla
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                t = speech_recognizer("tell us")
                print(t)
                if "new player" in t:
                    print("¡Eres un jugador nuevo!")
                    running = False
                else:
                    print("¡Eres un jugador existente!")
                    running = False

    # Cerrar Pygame
    pygame.quit()
