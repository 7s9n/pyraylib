import pyraylib
from pyraylib.colors import (
    DARKGRAY,
    DARK_BLUE,
    LIME,
    MAROON,
    RAYWHITE
)

def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window((SCREEN_WIDTH, SCREEN_HEIGHT), 'pyraylib [core] example - mouse input')
    # Set our game to run at 60 frames-per-second
    window.set_fps(60)
    circle = pyraylib.Circle(
        center=pyraylib.Vector2(-100., -100.),
        radius=40,
        color=DARK_BLUE
    )
    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        circle.center = pyraylib.get_mouse_position()

        if pyraylib.is_mouse_button_pressed(pyraylib.MouseButton.LEFT_BUTTON):
            circle.color = MAROON
        elif pyraylib.is_mouse_button_pressed(pyraylib.MouseButton.MIDDLE_BUTTON):
            circle.color = LIME
        elif pyraylib.is_mouse_button_pressed(pyraylib.MouseButton.RIGHT_BUTTON):
            circle.color = DARK_BLUE
        
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        pyraylib.draw_text("move ball with mouse and click mouse button to change color", 10, 10, 20, DARKGRAY)
        circle.draw()
        window.end_drawing()

    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()