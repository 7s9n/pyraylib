import pyraylib
from pyraylib.colors import (
    DARKGRAY,
    RAYWHITE,
    LIGHTGRAY,
    GRAY,
)

def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT), 
        title='pyraylib [core] example - drop files',
        fps=60
    )
    dropped_files = []
    color = pyraylib.Color(LIGHTGRAY)
    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        if pyraylib.is_file_dropped():
            dropped_files = pyraylib.get_dropped_files()
        
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)

        if dropped_files:
            pyraylib.draw_text("Dropped files:", 100, 40, 20, DARKGRAY)
            for idx , file in enumerate(dropped_files):
                if idx & 1 == 0:
                    rect = pyraylib.Rectangle(0, 85 + 40* idx, SCREEN_WIDTH, 40).draw(color.fade(0.5))
                else:
                    rect = pyraylib.Rectangle(0, 85 + 40* idx, SCREEN_WIDTH, 40).draw(color.fade(0.3))
                pyraylib.draw_text(file, 120, 100 + 40 * idx, 10, GRAY)
            pyraylib.draw_text("Drop new files...", 100, 110 + 40 * len(dropped_files), 20, DARKGRAY)
        else:
            pyraylib.draw_text("Drop your files to this window!", 100, 40, 20, DARKGRAY)
        window.end_drawing()

    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()