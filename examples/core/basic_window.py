import pyraylib
from pyraylib.colors import (
    LIGHTGRAY,
    RAYWHITE
)

def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window((SCREEN_WIDTH, SCREEN_HEIGHT), 'pyraylib [core] example - basic window')
    # Set our game to run at 60 frames-per-second
    window.set_fps(60)

    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        # TODO: Update your variables here
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        pyraylib.draw_text('Congrats! You created your first window!', 190, 200, 20, LIGHTGRAY)
        window.end_drawing()

    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()