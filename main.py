from pyraylib.colors import RED
import pyraylib

window = pyraylib.Window((800, 450), 'Hussein')

while window.is_open():
    window.begin_drawing()
    window.clear_background(RED)
    window.end_drawing()

window.close()