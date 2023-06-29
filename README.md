## MarioKart

Para ejecutar simplemente ejecutar el test.py

# Se han añadido un par de mejoras.
Ahora guardaremos a los jugadores en un json por nombre (identificador), tambien guardaremos un par de sus preferencias que las rellenaremos cuando se registre dicho usuario,(idiioma, personaje del coche sobre el que se vera en el juego) junto con su encoding face

El flujo del programa ahora es de la siguente forma, se abrirá un menu de inicio de juego donde se pregunta si quieres crear un jugador o ya estas registrado.
Note: Es recomendable que en vez de hacer esto, primeramente detecte si tu cara esta dentro de la base de datos? asi no tendria que preguntar y directamente entrariamos en el apartado de crear jugador.
Si respondes NEW_PLAYER crearas un nuevo jugador, 
Si respomdes cualquier otra cosa, reconocera tu cara y la buscara en la base de datos, y comienza la ejecucion con tus preferencias

# TODO
- Cambier preferencias
- Manejar errores
- Añadir funcionalidad en general

