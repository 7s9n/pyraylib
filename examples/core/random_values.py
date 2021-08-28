import pyraylib
from pyraylib.colors import (
    RAYWHITE,
    MAROON,
    LIGHTGRAY,
)
def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT), 
        title='pyraylib [core] example - generate random values',
        fps=60,
    )
    # Variable used to count frames
    frames_counter = 0
    # Get a random integer number between -8 and 5 (both included)
    rand_value = pyraylib.get_random_value(-8, 5)

    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        frames_counter += 1
        # Every two seconds (120 frames) a new random value is generated
        if frames_counter > 120:
            rand_value = pyraylib.get_random_value(-8, 5)
            frames_counter = 0

        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        pyraylib.draw_text("Every 2 seconds a new random value is generated:", 130, 100, 20, MAROON)
        pyraylib.draw_text(f"{rand_value}", 360, 180, 80, LIGHTGRAY)
        window.end_drawing()

    # De-Initialization
    # TODO: Unload all loaded data (textures, fonts, audio) here!
    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()