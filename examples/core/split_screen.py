import pyraylib
from pyraylib.colors import (
    BEIGE,
    BLACK,
    BLUE,
    BROWN,
    DARKGRAY,
    GREEN,
    RED,
    SKY_BLUE,
    WHITE,
)


# Scene drawing
def draw_scene(texture_grid, camera_player_1, camera_player_2):
    count = 5
    spacing = 4

    # Grid of cube trees on a plane to make a "world"
    pyraylib.draw_plane((0, 0, 0 ), (50, 50), BEIGE) # Simple world plane

    for x in range(-count*spacing, (count*spacing)+1, spacing):
        for z in range(-count*spacing, (count*spacing)+1, spacing):
            pyraylib.draw_cube_texture(texture_grid, (x, 1.5, z), 1, 1, 1, GREEN)
            pyraylib.draw_cube_texture(texture_grid, (x, 0.5, z), 0.25, 1, 0.25, BROWN)
    
    # Draw a cube at each player's position
    pyraylib.draw_cube(camera_player_1.position, 1, 1, 1, RED)
    pyraylib.draw_cube(camera_player_2.position, 1, 1, 1, BLUE)

def main():
    # Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 450
    window = pyraylib.Window(
        size=(SCREEN_WIDTH, SCREEN_HEIGHT), 
        title='pyraylib [core] example - split screen',
        fps=60
    )
    # Generate a simple texture to use for trees
    img = pyraylib.Image.checked(256, 256, 32, 32, DARKGRAY, WHITE)
    texture_grid = img.get_texture().set_filter(
        pyraylib.TextureFilter.ANISOTROPIC_16X
    ).set_wrap(pyraylib.TextureWrap.CLAMP)
    img.unload()

    # Setup player 1 camera and screen
    camera_player_1 = pyraylib.Camera(
        fovy=45.,
        up=(0, 1., 0),
        target=(0, 1., 0),
        position=(0, 1., -3.)
    )
    screen_player_1 = pyraylib.RenderTexture.load(SCREEN_WIDTH // 2, SCREEN_HEIGHT)

    # Setup player two camera and screen
    camera_player_2 = pyraylib.Camera(
        fovy=45.,
        up=(0, 1., 0),
        target=(0, 3., 0),
        position=(-3., 3., 0)
    )
    screen_player_2 = pyraylib.RenderTexture.load(SCREEN_WIDTH // 2, SCREEN_HEIGHT)
    
    # Build a flipped rectangle the size of the split view to use for drawing later
    split_screen_rect = pyraylib.Rectangle(
        0., 
        0., 
        screen_player_1.texture.width, 
        -screen_player_1.texture.height
        )
    # Main game loop
    while window.is_open(): # Detect window close button or ESC key
        # Update
        """
        If anyone moves this frame, how far will they move based on the time since the last frame
        this moves thigns at 10 world units per second, regardless of the actual FPS
        """
        offset_this_frame = 10. * window.get_frame_time()

        # Move Player1 forward and backwards (no turning)
        if pyraylib.is_key_down(pyraylib.Keyboard.W):
            camera_player_1.position.z += offset_this_frame
            camera_player_1.target.z += offset_this_frame
        elif pyraylib.is_key_down(pyraylib.Keyboard.S):
            camera_player_1.position.z -= offset_this_frame
            camera_player_1.target.z -= offset_this_frame
        
        # Move Player2 forward and backwards (no turning)
        if pyraylib.is_key_down(pyraylib.Keyboard.UP):
            camera_player_2.position.x += offset_this_frame
            camera_player_2.target.x += offset_this_frame
        elif pyraylib.is_key_down(pyraylib.Keyboard.DOWN):
            camera_player_2.position.x -= offset_this_frame
            camera_player_2.target.x -= offset_this_frame

        # Draw
        # Draw Player1 view to the render texture
        # I can also use with statement to open and close render texture and camera3d mode
        with screen_player_1 as sp1:
            window.clear_background(SKY_BLUE)
            with camera_player_1 as cp1:
                draw_scene(texture_grid, camera_player_1, camera_player_2)
            pyraylib.draw_text("PLAYER1 W/S to move", 0, 0, 20, RED)
        
        # Draw Player2 view to the render texture
        # without using with statement
        screen_player_2.begin_mode()
        window.clear_background(SKY_BLUE)
        camera_player_2.begin_mode()
        draw_scene(texture_grid, camera_player_1, camera_player_2)
        camera_player_2.end_mode()
        pyraylib.draw_text("PLAYER2 UP/DOWN to move", 0, 0, 20, BLUE)
        screen_player_2.end_mode()

        # Draw both views render textures to the screen side by side
        window.begin_drawing()
        window.clear_background(BLACK)
        screen_player_1.texture.draw_rec(split_screen_rect, ( 0, 0), WHITE)
        screen_player_2.texture.draw_rec(split_screen_rect, (SCREEN_WIDTH / 2., 0.), WHITE)
        window.end_drawing()

    # De-Initialization
    screen_player_1.unload() # Unload render texture
    screen_player_2.unload() # Unload render texture
    texture_grid.unload()    # Unload texture
    window.close()           # Close window and OpenGL context

if __name__ == '__main__':
    main()