import pyraylib
from pyraylib.colors import (
    DARKGRAY,
    MAROON,
    RAYWHITE
)

def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window((SCREEN_WIDTH, SCREEN_HEIGHT), 'pyraylib [core] example - keyboard input')
    # Set our game to run at 60 frames-per-second
    window.set_fps(60)
    circle = pyraylib.Circle(
        center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        radius=50,
        color=MAROON
    )
    speed = 2.0
    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        if pyraylib.is_key_down(pyraylib.Keyboard.RIGHT):
            circle.move(speed, 0) # circle.x += speed
        if pyraylib.is_key_down(pyraylib.Keyboard.LEFT):
            circle.move(-speed, 0) # circle.x -= speed
        if pyraylib.is_key_down(pyraylib.Keyboard.UP):
            circle.move(0, -speed) # circle.y -= speed
        if pyraylib.is_key_down(pyraylib.Keyboard.DOWN):
            circle.move(0, speed) # circle.y += speed
        
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        pyraylib.draw_text("move the ball with arrow keys", 10, 10, 20, DARKGRAY)
        circle.draw()
        window.end_drawing()

    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()