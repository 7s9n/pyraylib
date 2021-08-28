import pyraylib
from pyraylib.colors import (
    GRAY,
    LIGHTGRAY,
    MAROON,
    RAYWHITE
)

def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window((SCREEN_WIDTH, SCREEN_HEIGHT), 'pyraylib [core] example - mouse wheel')
    # Set our game to run at 60 frames-per-second
    window.set_fps(60)
    box = pyraylib.Rectangle(
        SCREEN_WIDTH / 2 - 40,
        SCREEN_HEIGHT/2 - 40,
        80,
        80,
    )
    scroll_speed = 2
    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        box.y = pyraylib.get_mouse_wheel_move() * scroll_speed
        
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        pyraylib.draw_text("Use mouse wheel to move the cube up and down!", 10, 10, 20, GRAY)
        pyraylib.draw_text(f"Box position Y: {box.y}", 10, 40, 20, LIGHTGRAY)
        box.draw(color=MAROON)
        window.end_drawing()

    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()
