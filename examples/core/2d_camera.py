from typing import List
import pyraylib
from pyraylib.colors import (
    GREEN,
    RAYWHITE,
    RED,
)
def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    MAX_BUILDINGS: int = 100
    window = pyraylib.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT), 
        title='pyraylib [core] example - 2d camera',
        fps=60,
    )
    player: pyraylib.Rectangle = pyraylib.Rectangle(400, 280, 40, 40)
    buildings: List[pyraylib.Rectangle] = []
    build_colors: List[pyraylib.Color] = []

    spacing = 0
    
    for i in range(MAX_BUILDINGS):
        rect = pyraylib.Rectangle()
        rect.width = pyraylib.get_random_value(50, 200)
        rect.height = pyraylib.get_random_value(100, 800)
        rect.x = -6000. + spacing
        rect.y = SCREEN_HEIGHT - 130.0 - rect.height
        spacing += rect.width
        buildings.append(rect)
        build_colors.append(pyraylib.get_random_color())
    
    camera = pyraylib.Camera2D(
        offset=(SCREEN_WIDTH/2., SCREEN_HEIGHT/2.),
        target=(player.x + 20., player.y + 20.),
        rotation=0.,
        zoom=1.,
    )

    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        # Player movement
        if pyraylib.is_key_down(pyraylib.Keyboard.RIGHT):
            player.x += 2
        elif pyraylib.is_key_down(pyraylib.Keyboard.LEFT):
            player -= 2

        # Camera target follows player
        camera.target = (player.x + 20, player.y + 20)

        # Camera rotation controls
        if pyraylib.is_key_down(pyraylib.Keyboard.A):
            camera.rotation -= 1
        elif pyraylib.is_key_down(pyraylib.Keyboard.S):
            camera.rotation += 1
        
        # Limit camera rotation to 80 degrees (-40 to 40)
        if camera.rotation > 40:
            camera.rotation = 40
        elif camera.rotation < -40:
            camera.rotation = -40
        
        # Camera zoom controls
        camera.zoom += pyraylib.get_mouse_wheel_move() * 0.05

        if camera.zoom > 3.:
            camera.zoom = 3.
        elif camera.zoom < 0.1:
            camera.zoom = 0.1
        
        # Camera reset (zoom and rotation)
        if pyraylib.is_key_pressed(pyraylib.Keyboard.R):
            camera.zoom = 1.
            camera.rotation = 0.
            
        # Draw
        window.begin_drawing()
        window.clear_background(RAYWHITE)
        camera.begin_mode()
        for i in range(MAX_BUILDINGS):
            buildings[i].draw(build_colors[i])
        player.draw(RED)
        pyraylib.draw_line((camera.target.x, -SCREEN_HEIGHT*10) , (camera.target.x, SCREEN_HEIGHT*10), GREEN)
        pyraylib.draw_line((-SCREEN_WIDTH*10, camera.target.y), (SCREEN_WIDTH*10, camera.target.y), GREEN)
        camera.end_mode()
        pyraylib.draw_text("SCREEN AREA", 640, 10, 20, RED)
        window.end_drawing()

    # De-Initialization
    # TODO: Unload all loaded data (textures, fonts, audio) here!
    # Close window and OpenGL context
    window.close()

if __name__ == '__main__':
    main()