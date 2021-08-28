import pyraylib
from pyraylib.colors import (
    BLUE,
    DARKGREEN,
    GREEN,
    RAYWHITE,
    LIGHTGRAY,
    GRAY,
    MAROON,
    DARKBLUE,
    PURPLE,
)
from enum import IntEnum
def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT), 
        title='pyraylib [core] example - basic screen manager',
        fps=60,
    )
    class GameScreen(IntEnum):
        """
        Window Status
        """
        LOGO = 0,
        TITLE = 1,
        GAMEPLAY = 2,
        ENDING = 3,

    current_screen = GameScreen.LOGO

    # TODO: Initialize all required variables and load all required data here!

    frames_counter = 0 # Useful to count frames
    rect = pyraylib.Rectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        if current_screen == GameScreen.LOGO:
            # TODO: Update LOGO screen variables here!
            frames_counter += 1 # Count frames
            # Wait for 2 seconds (120 frames) before jumping to TITLE screen
            if frames_counter > 120:
                current_screen = GameScreen.TITLE
                del frames_counter
        
        elif current_screen == GameScreen.TITLE:
            # TODO: Update TITLE screen variables here!

            # Press enter to change to GAMEPLAY screen
            if pyraylib.is_key_pressed(pyraylib.Keyboard.ENTER) \
                or pyraylib.is_key_pressed(pyraylib.Keyboard.TAB):
                current_screen = GameScreen.GAMEPLAY

        elif current_screen == GameScreen.GAMEPLAY:
            # TODO: Update GAMEPLAY screen variables here!

            # Press enter to change to ENDING screen
            if pyraylib.is_key_pressed(pyraylib.Keyboard.ENTER) \
                or pyraylib.is_key_pressed(pyraylib.Keyboard.TAB):
                current_screen = GameScreen.ENDING

        elif current_screen == GameScreen.ENDING:
            # TODO: Update ENDING screen variables here!

            # Press enter to return to TITLE screen
            if pyraylib.is_key_pressed(pyraylib.Keyboard.ENTER) \
                or pyraylib.is_key_pressed(pyraylib.Keyboard.TAB):
                current_screen = GameScreen.TITLE
        
        else:
            pass
        
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        
        if current_screen == GameScreen.LOGO:
            # TODO: Draw LOGO screen here!
            pyraylib.draw_text("LOGO SCREEN", 20, 20, 40, LIGHTGRAY)
            pyraylib.draw_text("WAIT for 2 SECONDS...", 290, 220, 20, GRAY)

        elif current_screen == GameScreen.TITLE:
            # TODO: Draw TITLE screen here!
            rect.draw(GREEN)
            pyraylib.draw_text("TITLE SCREEN", 20, 20, 40, DARKGREEN)
            pyraylib.draw_text("PRESS ENTER or TAP to JUMP to GAMEPLAY SCREEN", 120, 220, 20, DARKGREEN)

        elif current_screen == GameScreen.GAMEPLAY:
            # TODO: Draw GAMEPLAY screen here!
            rect.draw(PURPLE)
            pyraylib.draw_text("GAMEPLAY SCREEN", 20, 20, 40, MAROON)
            pyraylib.draw_text("PRESS ENTER or TAP to JUMP to ENDING SCREEN", 130, 220, 20, MAROON)

        elif current_screen == GameScreen.ENDING:
            # TODO: Draw ENDING screen here!
            rect.draw(BLUE)
            pyraylib.draw_text("ENDING SCREEN", 20, 20, 40, DARKBLUE)
            pyraylib.draw_text("PRESS ENTER or TAP to RETURN to TITLE SCREEN", 120, 220, 20, DARKBLUE)

        window.end_drawing()

    # De-Initialization
    # TODO: Unload all loaded data (textures, fonts, audio) here!
    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()