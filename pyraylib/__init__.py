from colorsys import hls_to_rgb, hsv_to_rgb, rgb_to_hls, rgb_to_hsv, rgb_to_yiq, yiq_to_rgb
from functools import wraps
import sys
import os
from pathlib import Path
from enum import IntFlag
from multipledispatch import dispatch

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union ,
    Tuple,
)

from ctypes import (
    POINTER,
    Structure,
    byref,
    c_bool,
    c_byte,
    c_char,
    c_char_p,
    c_double,
    c_short,
    c_ubyte,
    c_float,
    c_ushort,
    c_void_p,
    c_int,
    c_uint,
    cdll,
)
from pyraylib.colors import *

_lib_filename = {
    'win32':'raylib.dll',
    'linux':'libraylib.so',
    'darwin': 'libraylib.3.7.0.dylib',
}

_platform = sys.platform
lib_name = _lib_filename[_platform]
main_mod = sys.modules['__main__']

env_path = Path(os.environ['RAYLIB_PATH']) if 'RAYLIB_PATH' in os.environ else None
file_path = Path(__file__).parent / 'lib'
inside_package = file_path.parent
main_path = Path(main_mod.__file__).parent 

if env_path and env_path.exists():
    RAYLIB_PATH = env_path
if (file_path / lib_name).exists():
    RAYLIB_PATH = file_path
elif (main_path / lib_name).exists():
    RAYLIB_PATH = main_path
elif (inside_package / lib_name).exists():
    RAYLIB_PATH = inside_package
else:
    raise Exception(
        """
        Cannot find "{}" in these search paths:\n
        __file__ folder: "{}"\n
        '__main__ folder: "{}"\n
        os.environ["RAYLIB_PATH"] -> "{}"
        """.format(
            lib_name ,
            str(file_path),
            str(main_path),
            str(env_path) if env_path else "NOT SET"
        )
    )


__all__ = (
    'Window'
    'draw_text'
)

_rl = cdll.LoadLibrary(str(RAYLIB_PATH / lib_name))

"""
utility functions and types
"""
Bool = c_bool
VoidPtr = c_void_p
CharPtr = c_char_p
CharPtrPtr = POINTER(c_char_p)
UCharPtr = POINTER(c_ubyte)
IntPtr = POINTER(c_int)
UIntPtr = POINTER(c_uint)
FloatPtr = POINTER(c_float)
UShortPtr = POINTER(c_ushort)
Char = c_char
UChar = c_ubyte
Byte = c_byte
Short = c_short
Int = c_int
Float = c_float
UInt = c_uint
Double = c_double
Struct = Structure
Number = Union[int , float]
Seq = Sequence[Number]
IntSequence = Sequence[int]
FloatSequence = Sequence[float]
VectorN = Union[Seq , 'Vector2' , 'Vector3' , 'Vector4']

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def python_wrapper(funcname: str, argtypes: List[Any]= None, restype: Any= None):
    wrap_function(funcname,restype=restype,argtypes=argtypes)
    def decorator(function):
        @wraps(function)
        def wrapper_function(*args, **kwargs):
            value = function(*args, **kwargs)
            return value
        return wrapper_function
    return decorator

def wrap_function(funcname, argtypes= None, restype= None):
    """Simplify wrapping ctypes functions"""
    func = _rl.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

def _to_int(value: float):
    return int(value) if isinstance(value , float) else value

def _to_float(value: int):
    return float(value) if isinstance(value , int) else value

def _to_byte_str(value: str):
    if not isinstance(value, (str, bytes)):
        value = str(value)
    return value.encode('utf-8', 'ignore')

def _to_str(value: bytes):
    return value.decode('utf-8' , 'ignore') if isinstance(value , bytes) else value

def _to_color(seq: Seq)-> 'Color':
    if isinstance(seq , Color):
        return seq
    else:
        r , g , b , a = (_to_float(v) for v in seq)
        return Color(r , g , b , a)

def _vec2(seq: FloatSequence)-> 'Vector2':
    if isinstance(seq , Vector2):
        return seq
    else:
        x , y = (_to_float(v) for v in seq)
        return Vector2(x , y)

def _vec3(seq: FloatSequence)-> 'Vector3':
    if isinstance(seq , Vector3):
        return seq
    else:
        x , y , z = (_to_float(v) for v in seq)
        return Vector3(x , y ,z)

def _vec4(seq: FloatSequence)-> 'Vector4':
    if isinstance(seq , Vector4):
        return seq
    else:
        x , y , z , w = (_to_float(v) for v in seq)
        return Vector4(x, y , z , w)

def _rect(seq: FloatSequence)-> 'Rectangle':
    if isinstance(seq , Rectangle):
        return seq
    else:
        x, y, w, h = (_to_float(v) for v in seq)
        return Rectangle(x , y , w , h)

def _flatten(
    filter_types: Sequence[Type] ,
    *args: Any ,
    map_to: Optional[Type]= None
    )-> List[Any]:
    flatten_list = []
    for value in args:
        if isinstance(value , filter_types):
            flatten_list.append(map_to(value) if map_to else value)
        else:
            flatten_list.extend(_flatten(filter_types , *value , map_to=map_to))

    return flatten_list

"""

Enumerators Definition

"""
class WindowState(IntFlag):
    """System/Window config flags"""
    VSYNC_HINT = 64         #Try enabling V-Sync on GPU
    FULLSCREEN_MODE = 2     #Run program in fullscreen
    RESIZABLE = 4    #Allow resizable window
    UNDECORATED = 8  #Disable window decoration (frame and buttons)
    HIDDEN = 128     #Hide window
    MINIMIZED = 512  #Minimize window (iconify)
    MAXIMIZED = 1024 #Maximize window (expanded to monitor)
    UNFOCUSED = 2048 #Window non focused
    TOPMOST = 4096   #Window always on top
    ALWAYS_RUN = 256 #Allow windows running while minimized
    TRANSPARENT = 16 #Allow transparent framebuffer
    HIGHDPI = 8192   #Support HighDPI
    MSAA_4X_HINT = 32       #Try enabling MSAA 4X
    INTERLACED_HINT = 65536 #Try enabling interlaced video format (for V3D)
    
class Keyboard(IntFlag):
    """Keyboard keys (US keyboard layout)"""
    NULL            = 0
    # Alphanumeric keys
    APOSTROPHE      = 39
    COMMA           = 44
    MINUS           = 45
    PERIOD          = 46
    SLASH           = 47
    ZERO            = 48
    ONE             = 49
    TWO             = 50
    THREE           = 51
    FOUR            = 52
    FIVE            = 53
    SIX             = 54
    SEVEN           = 55
    EIGHT           = 56
    NINE            = 57
    SEMICOLON       = 59
    EQUAL           = 61
    A               = 65
    B               = 66
    C               = 67
    D               = 68
    E               = 69
    F               = 70
    G               = 71
    H               = 72
    I               = 73
    J               = 74
    K               = 75
    L               = 76
    M               = 77
    N               = 78
    O               = 79
    P               = 80
    Q               = 81
    R               = 82
    S               = 83
    T               = 84
    U               = 85
    V               = 86
    W               = 87
    X               = 88
    Y               = 89
    Z               = 90

    # Function keys
    SPACE           = 32
    ESCAPE          = 256
    ENTER           = 257
    TAB             = 258
    BACKSPACE       = 259
    INSERT          = 260
    DELETE          = 261
    RIGHT           = 262
    LEFT            = 263
    DOWN            = 264
    UP              = 265
    PAGE_UP         = 266
    PAGE_DOWN       = 267
    HOME            = 268
    END             = 269
    CAPS_LOCK       = 280
    SCROLL_LOCK     = 281
    NUM_LOCK        = 282
    PRINT_SCREEN    = 283
    PAUSE           = 284
    F1              = 290
    F2              = 291
    F3              = 292
    F4              = 293
    F5              = 294
    F6              = 295
    F7              = 296
    F8              = 297
    F9              = 298
    F10             = 299
    F11             = 300
    F12             = 301
    LEFT_SHIFT      = 340
    LEFT_CONTROL    = 341
    LEFT_ALT        = 342
    LEFT_SUPER      = 343
    RIGHT_SHIFT     = 344
    RIGHT_CONTROL   = 345
    RIGHT_ALT       = 346
    RIGHT_SUPER     = 347
    KB_MENU         = 348
    LEFT_BRACKET    = 91
    BACKSLASH       = 92
    RIGHT_BRACKET   = 93
    GRAVE           = 96

    # Keypad keys
    KP_0            = 320
    KP_1            = 321
    KP_2            = 322
    KP_3            = 323
    KP_4            = 324
    KP_5            = 325
    KP_6            = 326
    KP_7            = 327
    KP_8            = 328
    KP_9            = 329
    KP_DECIMAL      = 330
    KP_DIVIDE       = 331
    KP_MULTIPLY     = 332
    KP_SUBTRACT     = 333
    KP_ADD          = 334
    KP_ENTER        = 335
    KP_EQUAL        = 336
    # Android key buttons
    BACK            = 4
    MENU            = 82
    VOLUME_UP       = 24
    VOLUME_DOWN     = 25

class MouseButton(IntFlag):
    """Mouse buttons"""
    LEFT_BUTTON   = 0
    RIGHT_BUTTON  = 1
    MIDDLE_BUTTON = 2

class MouseCursor(IntFlag):
    """Mouse cursor"""
    CURSOR_DEFAULT       = 0
    CURSOR_ARROW         = 1
    CURSOR_IBEAM         = 2
    CURSOR_CROSSHAIR     = 3
    CURSOR_POINTING_HAND = 4
    CURSOR_RESIZE_EW     = 5     # The horizontal resize/move arrow shape
    CURSOR_RESIZE_NS     = 6     # The vertical resize/move arrow shape
    CURSOR_RESIZE_NWSE   = 7     # The top-left to bottom-right diagonal resize/move arrow shape
    CURSOR_RESIZE_NESW   = 8     # The top-right to bottom-left diagonal resize/move arrow shape
    CURSOR_RESIZE_ALL    = 9     # The omni-directional resize/move cursor shape
    CURSOR_NOT_ALLOWED   = 10    # The operation-not-allowed shape

class GamepadButton(IntFlag):
    """Gamepad buttons"""
    # This is here just for error checking
    BUTTON_UNKNOWN = 0

    # This is normally a DPAD
    BUTTON_LEFT_FACE_UP = 1
    BUTTON_LEFT_FACE_RIGHT = 2
    BUTTON_LEFT_FACE_DOWN = 3
    BUTTON_LEFT_FACE_LEFT = 4

    # This normally corresponds with PlayStation and Xbox controllers
    # XBOX: [Y,X,A,B]
    # PS3: [Triangle,Square,Cross,Circle]
    # No support for 6 button controllers though..
    BUTTON_RIGHT_FACE_UP = 5
    BUTTON_RIGHT_FACE_RIGHT = 6
    BUTTON_RIGHT_FACE_DOWN = 7
    BUTTON_RIGHT_FACE_LEFT = 8

    # Triggers
    BUTTON_LEFT_TRIGGER_1 = 9
    BUTTON_LEFT_TRIGGER_2 = 10
    BUTTON_RIGHT_TRIGGER_1 = 11
    BUTTON_RIGHT_TRIGGER_2 = 12

    # These are buttons in the center of the gamepad
    BUTTON_MIDDLE_LEFT = 13     # PS3 Select
    BUTTON_MIDDLE = 14          # PS Button/XBOX Button
    BUTTON_MIDDLE_RIGHT = 15    # PS3 Start

    # These are the joystick press in buttons
    BUTTON_LEFT_THUMB = 16
    BUTTON_RIGHT_THUMB = 17

class GamepadAxis(IntFlag):
    """Gamepad axis"""
    # Left stick
    GAMEPAD_AXIS_LEFT_X = 0
    GAMEPAD_AXIS_LEFT_Y = 1

    # Right stick
    GAMEPAD_AXIS_RIGHT_X = 2
    GAMEPAD_AXIS_RIGHT_Y = 3

    # Pressure levels for the back triggers
    GAMEPAD_AXIS_LEFT_TRIGGER = 4       # [1..-1] (pressure-level)
    GAMEPAD_AXIS_RIGHT_TRIGGER = 5      # [1..-1] (pressure-level)

class MaterialMapIndex(IntFlag):
    """Material map index"""
    ALBEDO    = 0       # MATERIAL_MAP_DIFFUSE
    METALNESS = 1       # MATERIAL_MAP_SPECULAR
    NORMAL    = 2
    ROUGHNESS = 3
    OCCLUSION = 4
    EMISSION = 5
    HEIGHT = 6
    BRDG = 7
    CUBEMAP = 8            # NOTE: Uses GL_TEXTURE_CUBE_MAP
    IRRADIANCE = 9         # NOTE: Uses GL_TEXTURE_CUBE_MAP
    PREFILTER = 9          # NOTE: Uses GL_TEXTURE_CUBE_MAP

class ShaderLocationIndex(IntFlag):
    """Shader location index"""
    VERTEX_POSITION = 0
    VERTEX_TEXCOORD01 = 1
    VERTEX_TEXCOORD02 = 2
    VERTEX_NORMAL = 3
    VERTEX_TANGENT = 4
    VERTEX_COLOR = 5
    MATRIX_MVP = 6
    MATRIX_VIEW = 7
    MATRIX_PROJECTION = 8
    MATRIX_MODEL = 9
    MATRIX_NORMAL = 10
    VECTOR_VIEW = 11
    COLOR_DIFFUSE = 12
    COLOR_SPECULAR = 13
    COLOR_AMBIENT = 14
    MAP_ALBEDO = 15          # MAP_DIFFUSE
    MAP_METALNESS = 16       # MAP_SPECULAR
    MAP_NORMAL = 17
    MAP_ROUGHNESS = 18
    MAP_OCCLUSION = 19
    MAP_EMISSION = 20
    MAP_HEIGHT = 21
    MAP_CUBEMAP = 22
    MAP_IRRADIANCE = 23
    MAP_PREFILTER = 24
    MAP_BRDF = 24

class ShaderUniformDataType(IntFlag):
    """Shader uniform data type"""
    FLOAT = 0
    VEC2 = 1
    VEC3 = 2
    VEC4 = 3
    INT = 4
    IVEC2 = 5
    IVEC3 = 6
    IVEC4 = 7
    SAMPLER2D = 8

class PixelFormat(IntFlag):
    """
    Pixel formats
    NOTE: Support depends on OpenGL version and platform
    """
    UNCOMPRESSED_GRAYSCALE = 1          # 8 bit per pixel (no alpha)
    UNCOMPRESSED_GRAY_ALPHA = 2         # 8*2 bpp (2 channels)
    UNCOMPRESSED_R5G6B5 = 3             # 16 bpp
    UNCOMPRESSED_R8G8B8 = 4             # 24 bpp
    UNCOMPRESSED_R5G5B5A1 = 5           # 16 bpp (1 bit alpha)
    UNCOMPRESSED_R4G4B4A4 = 6           # 16 bpp (4 bit alpha)
    UNCOMPRESSED_R8G8B8A8 = 7           # 32 bpp
    UNCOMPRESSED_R32 = 8                # 32 bpp (1 channel - float)
    UNCOMPRESSED_R32G32B32 = 9          # 32*3 bpp (3 channels - float)
    UNCOMPRESSED_R32G32B32A32 = 10      # 32*4 bpp (4 channels - float)
    COMPRESSED_DXT1_RGB = 11            # 4 bpp (no alpha)
    COMPRESSED_DXT1_RGBA = 12           # 4 bpp (1 bit alpha)
    COMPRESSED_DXT3_RGBA = 13           # 8 bpp
    COMPRESSED_DXT5_RGBA = 14           # 8 bpp
    COMPRESSED_ETC1_RGB = 15            # 4 bpp
    COMPRESSED_ETC2_RGB = 16            # 4 bpp
    COMPRESSED_ETC2_EAC_RGBA = 17       # 8 bpp
    COMPRESSED_PVRT_RGB = 18            # 4 bpp
    COMPRESSED_PVRT_RGBA = 19           # 4 bpp
    COMPRESSED_ASTC_4x4_RGBA = 20       # 8 bpp
    COMPRESSED_ASTC_8x8_RGBA = 21       # 2 bpp

class TextureFilter(IntFlag):
    """
    Texture parameters: filter mode
    NOTE 1: Filtering considers mipmaps if available in the texture
    NOTE 2: Filter is accordingly set for minification and magnification
    """
    POINT = 0                   # No filter, just pixel aproximation
    BILINEAR = 1                # Linear filtering
    TRILINEAR = 2               # Trilinear filtering (linear with mipmaps)
    ANISOTROPIC_4X = 3          # Anisotropic filtering 4x
    ANISOTROPIC_8X = 4          # Anisotropic filtering 8x
    ANISOTROPIC_16X = 5         # Anisotropic filtering 16x

class TextureWrap(IntFlag):
    """Texture parameters: wrap mode"""
    REPEAT = 0              # Repeats texture in tiled mode
    CLAMP = 1               # Clamps texture to edge pixel in tiled mode
    MIRROR_REPEAT = 2       # Mirrors and repeats the texture in tiled mode
    MIRROR_CLAMP = 3        # Mirrors and clamps to border the texture in tiled mode

class CubemapLayout(IntFlag):
    """Cubemap layouts"""
    AUTO_DETECT = 0            # Automatically detect layout type
    LINE_VERTICAL = 1          # Layout is defined by a vertical line with faces
    LINE_HORIZONTAL = 2        # Layout is defined by an horizontal line with faces
    CROSS_THREE_BY_FOUR = 3    # Layout is defined by a 3x4 cross with cubemap faces
    CROSS_FOUR_BY_THREE = 4    # Layout is defined by a 4x3 cross with cubemap faces
    PANORAMA = 5               # Layout is defined by a panorama image (equirectangular map)

class FontType(IntFlag):
    """Font type, defines generation method"""
    DEFAULT = 0        # Default font generation, anti-aliased
    BITMAP = 1         # Bitmap font generation, no anti-aliasing
    SDF = 2            # SDF font generation, requires external shader

class BlendMode(IntFlag):
    """Color blending modes (pre-defined)"""
    ALPHA = 0               # Blend textures considering alpha (default)
    ADDITIVE = 1            # Blend textures adding colors
    MULTIPLIED = 2          # Blend textures multiplying colors
    ADD_COLORS = 3          # Blend textures adding colors (alternative)
    SUBTRACT_COLORS = 4     # Blend textures subtracting colors (alternative)
    CUSTOM = 5              # Belnd textures using custom src/dst factors (use rlSetBlendMode())

class Gestures(IntFlag):
    """
    Gestures
    NOTE: It could be used as flags to enable only some gestures
    """
    NONE        = 0
    TAP         = 1
    DOUBLETAP   = 2
    HOLD        = 4
    DRAG        = 8
    SWIPE_RIGHT = 16
    SWIPE_LEFT  = 32
    SWIPE_UP    = 64
    SWIPE_DOWN  = 128
    PINCH_IN    = 256
    PINCH_OUT   = 512

class CameraMode(IntFlag):
    """Camera system modes"""
    CUSTOM = 0
    FREE = 1
    ORBITAL = 2
    FIRST_PERSON = 3
    THIRD_PERSON = 4

class CameraProjection(IntFlag):
    """Camera projection"""
    PERSPECTIVE = 0
    ORTHOGRAPHIC = 1

class NPatchLayout(IntFlag):
    """N-patch layout"""
    NINE_PATCH = 0              # Npatch layout: 3x3 tiles
    THREE_PATCH_VERTICAL = 1    # Npatch layout: 1x3 tiles
    THREE_PATCH_HORIZONTAL = 2  # Npatch layout: 3x1 tiles

"""
Structures Definition
"""
class Matrix(Struct):
    """Matrix type (OpenGL style 4x4 - right handed, column major)"""
    _fields_ = [
        ('m0', Float),
		('m4', Float),
		('m8', Float),
		('m12', Float),
		('m1', Float),
		('m5', Float),
		('m9', Float),
		('m13', Float),
		('m2', Float),
		('m6', Float),
		('m10', Float),
		('m14', Float),
		('m3', Float),
		('m7', Float),
		('m11', Float),
		('m15', Float),
    ]

    def trace(self)-> float:
        """Returns the trace of the matrix (sum of the values along the diagonal)"""
        return _rl.MatrixTrace(self)

    def transpose(self)-> 'Matrix':
        """Transposes provided matrix"""
        return _rl.MatrixTranspose(self)

    def invert(self)-> 'Matrix':
        return _rl.MatrixInvert(self)

    def normalize(self)-> 'Matrix':
        """Normalize provided matrix"""
        return _rl.MatrixNormalize(self)

    @staticmethod
    def identity()-> 'Matrix':
        return _rl.MatrixIdentity()
 
    @staticmethod
    def translate(seq: Sequence[float])-> 'Matrix':
        x, y, z = map(float, seq)
        return _rl.MatrixTranslate(
            x,
            y,
            z,
        )
    
    @staticmethod
    def rotate(axis: Union[Sequence[float], 'Vector3'], angle: float)-> 'Matrix':
        """
        Create rotation matrix from axis and angle
        NOTE: Angle should be provided in radians
        """
        return _rl.MatrixRotate(_vec3(_flatten((int, float), axis)), _to_float(angle))

    @staticmethod
    def rotate_x(angle: float)-> 'Matrix':
        """Get x-rotation matrix (angle in radians)"""
        return _rl.MatrixRotateX(_to_float(angle))
    
    @staticmethod
    def rotate_y(angle: float)-> 'Matrix':
        """Get y-rotation matrix (angle in radians)"""
        return _rl.MatrixRotateY(_to_float(angle))

    @staticmethod
    def rotate_z(angle: float)-> 'Matrix':
        """Get z-rotation matrix (angle in radians)"""
        return _rl.MatrixRotateZ(_to_float(angle))

    @staticmethod
    def rotate_xyz(ang: Union[Sequence[float], 'Vector3'])-> 'Matrix':
        """Get xyz-rotation matrix (angles in radians)"""
        return _rl.MatrixRotateXYZ(_vec3(ang))

    @staticmethod
    def rotate_zyx(ang: Union[Sequence[float], 'Vector3'])-> 'Matrix':
        """Get zyx-rotation matrix (angles in radians)"""
        return _rl.MatrixRotateZYX(_vec3(ang))
    
    @dispatch((float, int), (float, int), (float, int))
    def scale(x: float, y: float, z: float)-> 'Matrix':
        """Get scaling matrix"""
        return _rl.MatrixScale(
            _to_float(x),
            _to_float(y),
            _to_float(z),
        )

    @dispatch(Iterable)
    def scale(itr)-> 'Matrix':
        return Matrix.scale(*itr)

    def __add__(self, other: 'Matrix')-> 'Matrix':
        """Add two matrices"""
        return _rl.MatrixAdd(self, other)

    def __sub__(self, other: 'Matrix')-> 'Matrix':
        """Subtract two matrices (self - other)"""
        return _rl.MatrixSubtract(self, other)
    
    def __mul__(self, other: 'Matrix')-> 'Matrix':
        """
        Get two matrix multiplication
        NOTE: When multiplying matrices... the order matters!
        """
        return _rl.MatrixMultiply(self, other)

    def __str__(self) -> str:
        return f"(MATRIX: [{self.m0}, {self.m4}, {self.m8}, {self.m12}]\n[{self.m1}, {self.m5}, {self.m9}, {self.m13}]\n [{self.m2}, {self.m6}, {self.m10}, {self.m14}]\n [{self.m3}, {self.m7}, {self.m11}, {self.m15}]) "

wrap_function('MatrixScale', [Float, Float, Float], Matrix)
wrap_function('MatrixRotateZ', [Float], Matrix)
wrap_function('MatrixRotateY', [Float], Matrix)
wrap_function('MatrixRotateX', [Float], Matrix)
wrap_function('MatrixTranslate', [Float, Float, Float], Matrix)
wrap_function('MatrixMultiply', [Matrix, Matrix], Matrix)
wrap_function('MatrixSubtract', [Matrix, Matrix], Matrix)
wrap_function('MatrixAdd', [Matrix, Matrix], Matrix)
wrap_function('MatrixIdentity', restype=Matrix)
wrap_function('MatrixNormalize', [Matrix], Matrix)
wrap_function('MatrixInvert', [Matrix], Matrix)
wrap_function('MatrixTrace', [Matrix], Float)
wrap_function('MatrixTranspose', [Matrix], Matrix)


class Vector2(Struct):
    _fields_ = [
        ('_x' , Float),
        ('_y' , Float),
    ]

    def __init__(self, x: float= 0, y: float= 0)-> None:
        super(Vector2, self).__init__(
            _to_float(x),
            _to_float(y)
        )
    
    @property
    def x(self)-> float:
        return self._x

    @x.setter
    def x(self, i: float)-> None:
        self._x = i

    @property
    def y(self)-> float:
        return self._y

    @y.setter
    def y(self, i: float)-> None:
        self._y = i  

    def rotate(self, degree: float)-> 'Vector2':
        """Rotate Vector by float in Degrees"""
        return _rl.Vector2Rotate(self, _to_float(degree))

    def dot_product(self, other: 'Vector2')-> float:
        """Calculate two vectors dot product"""
        if not isinstance(other, Vector2):
            raise TypeError("{} must be Vector2, not {}".format(other, other.__class__.__qualname__))
        else:
            return self.x* other.x + self.y * other.y

    def distance(self, other)-> float:
        """Calculate distance between two vectors"""
        if not isinstance(other, Vector2):
            raise TypeError("{} must be Vector2, not {}".format(other, other.__class__.__qualname__))
        return _rl.Vector2Distance(self, other)

    def angle(self, other: 'Vector2')-> float:
        if not isinstance(other, Vector2):
            raise TypeError("{} must be Vector2, not {}".format(other, other.__class__.__qualname__))
        return _rl.Vector2Angle(self, other)

    def __str__(self) -> str:
        return f"({self.x} , {self.y})"

    def __iter__(self)-> Iterator[float]:
        return (self.x , self.y).__iter__()
    
    def __getitem__(self, key: Union[str, int, slice])-> Union[float, Sequence]:
        assert isinstance(key, (str, int, slice)), "KeyTypeError: {} not supported as subscription key.".format(key.__class__.__name__)
        if isinstance(key, (int, slice)):
            return [self.x, self.y][key]
        else:
            return {'x': self.x, 'y': self.y}[key]
    
    def __setitem__(self, key: Union[str, int], value: Number)-> None:
        assert isinstance(key, (str, int)), "KeyTypeError: {} not supported as subscription key.".format(key.__class__.__name__)
        if isinstance(key, int):
            a = [self.x, self.y]
            a[key] = _to_float(value)
            self.x, self.y = a
        else:
            a = {'x': self.x, 'y': self.y}
            assert key in a, "KeyError: invalid key '{}'.".format(key)
            a[key] = value
            self.x, self.y = tuple(a.values())

    def __len__(self)-> int:
        return 2
    
    def __pos__(self)-> 'Vector2':
        return Vector2(+self.x, +self.y)
    
    def __neg__(self)-> 'Vector2':
        return Vector2(-self.x, -self.y)

    def __abs__(self)-> 'Vector2':
        return Vector2(abs(self.x), abs(self.y))

    def __add__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self.x + other, self.y + other)
        else:
            return Vector2(self.x + other[0], self.y + other[1])
    
    def __sub__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self.x - other, self.y - other)
        else:
            return Vector2(self.x - other[0], self.y - other[1])

    def __truediv__(self, other: Union[Sequence[Number], Number, 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self.x / other, self.y / other)
        else:
            return Vector2(self.x / other[0], self.y / other[1])

    def __floordiv__(self, other: Union[Sequence[Number], Number, 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self.x // other, self.y // other)
        else:
            return Vector2(self.x // other[0], self.y // other[1])

    def __mod__(self, other: Union[Sequence[Number], Number, 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self.x % other, self.y % other)
        else:
            return Vector2(self.x % other[0], self.y % other[1])

    def __mul__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            return Vector2(self.x * other, self.y * other)
        else:
            return Vector2(self.x * other[0], self.y * other[1])

    def __iadd__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            self.x += other
            self.y += other
        else:
            self.x += other[0]
            self.y += other[1]
        return self

    def __isub__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            self.x -= other
            self.y -= other
        else:
            self.x -= other[0]
            self.y -= other[1]
        return self
    
    def __itruediv__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            self.x /= other
            self.y /= other
        else:
            self.x /= other[0]
            self.y /= other[1]
        return self
    
    def __ifloordiv__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            self.x //= other
            self.y //= other
        else:
            self.x //= other[0]
            self.y //= other[1]
        return self
    
    def __imod__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            self.x %= other
            self.y %= other
        else:
            self.x %= other[0]
            self.y %= other[1]
        return self
    
    def __imul__(self, other: Union[Number, Sequence[Number], 'Vector2'])-> 'Vector2':
        if isinstance(other, (int, float)):
            self.x *= other
            self.y *= other
        else:
            self.x *= other[0]
            self.y *= other[1]
        return self

    def __eq__(self, other: Union[Sequence[float], 'Vector2'])-> bool:
        return self.x == other[0] and self.y == other.y
    
    def __repr__(self) -> str:
        return "{}({},{})".format(self.__class__.__qualname__, self.x, self.y)

    @classmethod
    def one(cls):
        return cls(1., 1.)

    @classmethod
    def zero(cls):
        return cls(0., 0.)

Vector2Ptr = POINTER(Vector2)
wrap_function('Vector2Rotate', [Vector2, Float], Vector2)
wrap_function('Vector2DotProduct', [Vector2, Vector2], Float)
wrap_function('Vector2Distance', [Vector2, Vector2], Float)
wrap_function('Vector2Angle', [Vector2, Vector2], Float)

class Vector3(Struct):

    _fields_ = [
        ('_x' , Float),
        ('_y' , Float),
        ('_z' , Float),
    ]
    def __init__(self, x: float, y: float, z: float)-> None:
        super(Vector3, self).__init__(
            _to_float(x),
            _to_float(y),
            _to_float(z),
        )
    
    def __str__(self) -> str:
        return "({},{},{})".format(self.x, self.y, self.z)

    def __iter__(self)-> Iterator[float]:
        return (self.x , self.y , self.z).__iter__()
    
    def __getitem__(self, key: Union[str, int, slice])-> Union[float, Sequence]:
        assert isinstance(key, (str, int, slice)), "KeyTypeError: {} not supported as subscription key.".format(key.__class__.__name__)
        if isinstance(key, (int, slice)):
            return [self.x, self.y, self.z][key]
        else:
            return {'x': self.x, 'y': self.y,'z': self.z}[key]
    
    def __setitem__(self, key: Union[str, int], value: Number)-> None:
        assert isinstance(key, (str, int)), "KeyTypeError: {} not supported as subscription key.".format(key.__class__.__name__)
        if isinstance(key, int):
            a = [self.x, self.y, self.z]
            a[key] = _to_float(value)
            self.x, self.y, self.z = a
        else:
            a = {'x': self.x, 'y': self.y, 'z': self.z}
            assert key in a, "KeyError: invalid key '{}'.".format(key)
            a[key] = value
            self.x, self.y, self.z = tuple(a.values())

    @property
    def x(self)-> float:
        return self._x
    
    @x.setter
    def x(self, i: float)-> None:
        self._x = _to_float(i)
    
    @property
    def y(self)-> float:
        return self._y
    
    @y.setter
    def y(self, i: float)-> None:
        self._y = _to_float(i)
    
    @property
    def z(self)-> float:
        return self._z
    
    @z.setter
    def z(self, i: float)-> None:
        self._z = _to_float(i)
    
    def __len__(self)-> int:
        return 3
    
    def __pos__(self)-> 'Vector3':
        return Vector3(+self.x, +self.y, -self.z)
    
    def __neg__(self)-> 'Vector3':
        return Vector3(-self.x, -self.y, +self.z)

    def __abs__(self)-> 'Vector3':
        return Vector3(abs(self.x), abs(self.y), abs(self.z))

    def __add__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            return Vector3(self.x + other, self.y + other, self.z + other)
        else:
            return Vector3(self.x + other[0], self.y + other[1], self.z + other[2])
    
    def __sub__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            return Vector3(self.x - other, self.y - other, self.z - other)
        else:
            return Vector3(self.x - other[0], self.y - other[1], self.z - other[2])

    def __truediv__(self, other: Union[Sequence[Number], Number, 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            return Vector3(self.x / other, self.y / other, self.z / other)
        else:
            return Vector3(self.x / other[0], self.y / other[1], self.z / other[2])

    def __floordiv__(self, other: Union[Sequence[Number], Number, 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            return Vector3(self.x // other, self.y // other, self.z // other)
        else:
            return Vector3(self.x // other[0], self.y // other[1], self.z // other[2])

    def __mod__(self, other: Union[Sequence[Number], Number, 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            return Vector3(self.x % other, self.y % other, self.z % other)
        else:
            return Vector3(self.x % other[0], self.y % other[1], self.z % other[2])

    def __mul__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            return Vector3(self.x * other[0], self.y * other[1], self.z * other[2])

    def __iadd__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            self.x += other
            self.y += other
            self.z += other
        else:
            self.x += other[0]
            self.y += other[1]
            self.z += other[2]
        return self

    def __isub__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            self.x -= other
            self.y -= other
            self.z -= other
        else:
            self.x -= other[0]
            self.y -= other[1]
            self.z -= other[2]
        return self
    
    def __itruediv__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            self.x /= other
            self.y /= other
            self.z /= other
        else:
            self.x /= other[0]
            self.y /= other[1]
            self.z /= other[2]
        return self
    
    def __ifloordiv__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            self.x //= other
            self.y //= other
            self.z //= other
        else:
            self.x //= other[0]
            self.y //= other[1]
            self.z //= other[2]
        return self
    
    def __imod__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            self.x %= other
            self.y %= other
            self.z %= other
        else:
            self.x %= other[0]
            self.y %= other[1]
            self.z %= other[2]
        return self
    
    def __imul__(self, other: Union[Number, Sequence[Number], 'Vector3'])-> 'Vector3':
        if isinstance(other, (int, float)):
            self.x *= other
            self.y *= other
            self.z *= other
        else:
            self.x *= other[0]
            self.y *= other[1]
            self.z *= other[2]
        return self

    def __eq__(self, other: Union[Sequence[float], 'Vector3'])-> bool:
        return self.x == other[0] and self.y == other.y and self.z == other[2]
    
    def __repr__(self) -> str:
        return "{}({},{},{})".format(self.__class__.__qualname__, self.x, self.y, self.z)
    
    @classmethod
    def one(cls)-> 'Vector3':
        return cls(1., 1., 1.)
    
    @classmethod
    def zero(cls)-> 'Vector3':
        return cls(0., 0., 0.)

    def cross_product(self, other: 'Vector3')-> 'Vector3':
        """Calculate two vectors cross product"""
        if not isinstance(other, Vector3):
            raise TypeError("{} must be Vector3, not {}".format(other, other.__class__.__qualname__))
        return _rl.Vector3CrossProduct(self, other)
    
    def dot_product(self, other: 'Vector3')-> float:
        """Calculate two vectors dot product"""
        if not isinstance(other, Vector3):
            raise TypeError("{} must be Vector3, not {}".format(other, other.__class__.__qualname__))
        return _rl.Vector3DotProduct(self, other)

Vector3Ptr = POINTER(Vector3)

wrap_function('Vector3DotProduct', [Vector3, Vector3], Float)
wrap_function('Vector3CrossProduct', [Vector3, Vector3], Vector3)
wrap_function('MatrixRotateXYZ', [Vector3], Matrix)
wrap_function('MatrixRotateZYX', [Vector3], Matrix)
wrap_function('MatrixRotate', [Vector3, Float], Matrix)


    

class Vector4(Struct):
    _fields_ = [
        ('_x' , Float),
        ('_y' , Float),
        ('_z' , Float),
        ('_w' , Float),
    ]

    def __init__(self, x: float, y: float, z: float, w: float):
        super(Vector4, self).__init__(
            _to_float(x),
            _to_float(y),
            _to_float(z),
            _to_float(w),
        )

    @property
    def x(self)-> float:
        return self._x
    
    @x.setter
    def x(self, i: float)-> None:
        self._x = _to_float(i)
    
    @property
    def y(self)-> float:
        return self._y
    
    @y.setter
    def y(self, i: float)-> None:
        self._y = _to_float(i)
    
    @property
    def z(self)-> float:
        return self._z
    
    @z.setter
    def z(self, i: float)-> None:
        self._z = _to_float(i)
    
    @property
    def w(self)-> float:
        return self._z
    
    @w.setter
    def w(self, i: float)-> None:
        self._w = _to_float(i)

    def to_matrix(self)-> 'Matrix':
        return _rl.QuaternionToMatrix(self)

    def to_rectangle(self)-> 'Rectangle':
        return Rectangle(self.x, self.y, self.z, self.w)

    def invert(self)-> 'Vector4':
        return _rl.QuaternionInvert(self)

    def lerp(self, vector4: 'Vector4', amount: float)-> 'Vector4':
        return _rl.QuaternionLerp(self, vector4, amount)

    def nlerp(self, vector4: 'Vector4', amount: float)-> 'Vector4':
        return _rl.QuaternionNlerp(self, vector4, amount)

    def slerp(self, vector4: 'Vector4', amount: float)-> 'Vector4':
        return _rl.QuaternionSlerp(self, vector4, amount)

    def normalize(self)-> 'Vector4':
        return _rl.QuaternionNormalize(self)

    def transform(self, matrix: 'Matrix')-> 'Vector4':
        return _rl.QuaternionTransform(self, matrix)

    @staticmethod
    def identity()-> 'Vector4':
        return _rl.QuaternionIdentity()

    @staticmethod
    def from_vec3_to_vec3(from_: Vector3, to_: Vector3)-> 'Vector4':
        return _rl.QuaternionFromVector3ToVector3(from_, to_)

    @staticmethod
    def from_matrix(matrix: 'Matrix')-> 'Vector4':
        return _rl.QuaternionFromMatrix(matrix)

    @staticmethod
    def from_axis_angle(axis: Vector3, angle: float)-> 'Vector4':
        return _rl.QuaternionFromAxisAngle(axis, angle)

    def __iter__(self)-> Iterator[float]:
        return (self.x , self.y , self.z , self.w).__iter__()

    def __mul__(self, other: 'Vector4') -> 'Vector4':
        return _rl.QuaternionMultiply(self, other)

    def __len__(self)-> float:
        return _rl.QuaternionLength(self)

    def __getitem__(self, key: Union[str, int, slice])-> Union[float, Sequence]:
        assert isinstance(key, (str, int, slice)), "KeyTypeError: {} not supported as subscription key.".format(key.__class__.__name__)
        if isinstance(key, (int, slice)):
            return [self.x, self.y, self.z, self.w][key]
        else:
            return {'x': self.x, 'y': self.y,'z': self.z, 'w': self.w}[key]
    
    def __setitem__(self, key: Union[str, int], value: Number)-> None:
        assert isinstance(key, (str, int)), "KeyTypeError: {} not supported as subscription key.".format(key.__class__.__name__)
        if isinstance(key, int):
            a = [self.x, self.y, self.z, self.w]
            a[key] = _to_float(value)
            self.x, self.y, self.z, self.w = a
        else:
            a = {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}
            assert key in a, "KeyError: invalid key '{}'.".format(key)
            a[key] = value
            self.x, self.y, self.z, self.w = tuple(a.values())

    def __len__(self)-> int:
        return 4

    def __str__(self) -> str:
        return f"({self.x} , {self.y} , {self.z} , {self.w})"
    
    def __repr__(self) -> str:
        return "{}({},{},{},{})".format(self.__class__.__qualname__, self.x, self.y, self.z, self.w)

wrap_function('QuaternionToMatrix', [Vector4], Matrix)
wrap_function('QuaternionInvert', [Vector4], Vector4)
wrap_function('QuaternionMultiply', [Vector4, Vector4], Vector4)
wrap_function('QuaternionLerp', [Vector4, Vector4, Float], Vector4)
wrap_function('QuaternionNlerp', [Vector4, Vector4, Float], Vector4)
wrap_function('QuaternionSlerp', [Vector4, Vector4, Float], Vector4)
wrap_function('QuaternionLength', [Vector4], Float)
wrap_function('QuaternionNormalize', [Vector4], Vector4)
wrap_function('QuaternionTransform', [Vector4, Matrix], Vector4)
wrap_function('QuaternionIdentity', restype=Vector4)
wrap_function('QuaternionFromVector3ToVector3', [Vector3, Vector3], Vector4)
wrap_function('QuaternionFromMatrix', [Matrix], Vector4)
wrap_function('QuaternionFromAxisAngle', [Vector3, Float], Vector4)

Vector4Ptr = POINTER(Vector4)
Quaternion = Vector4

class Color(Struct):
    _fields_  = [
        ('r' , c_ubyte),
        ('g' , c_ubyte),
        ('b' , c_ubyte),
        ('a' , c_ubyte),
    ]
    @dispatch((int, float), (int, float), (int, float), (int, float))
    def __init__(self, r: int, g: int, b: int, a: int)-> None:
        self.__set(
            r,
            g,
            b,
            a,
        )
    
    @dispatch(Iterable)
    def __init__(self, color: Union[Sequence[int], 'Color']) -> None:
        color = _flatten((int, float), *color)
        self.__set(
            *color
        )
    
    @dispatch()
    def __init__(self)-> None:
        self.__set()
    
    @property
    def normalized(self) -> 'Vector4':
        """Gets or sets a normalized Vector4 color."""
        return Vector4(
            self.r / 255.0,
            self.g / 255.0,
            self.b / 255.0,
            self.a / 255.0
        )
    
    @normalized.setter
    def normalized(self, value: Union[Sequence[Number], Vector3, Vector4])-> None:
        value = _flatten((int, float), *value, map_to=float)
        length = len(value)
        if length not in (3, 4):
            raise ValueError("Too many or too few values (expected 3 or 4, not {})".format(length))
        self.r = int(value[0] * 255.0)
        self.g = int(value[1] * 255.0)
        self.b = int(value[2] * 255.0)
        if length == 4:
            self.a = int(value[3] * 255.0)

    @property
    def hsv(self)-> Vector4:
        """Gets a normalized color in HSV colorspace."""
        return Vector4(*rgb_to_hsv(*self.normalized[:3]), self.a / 255.)

    @hsv.setter
    def hsv(self, value: Union[Sequence[Number], Vector3, Vector4])-> None:
        value = _flatten((int, float), *value, map_to=float)
        val_len = len(value)
        if val_len not in (3, 4):
            raise ValueError("Too many or too few values (expected 3 or 4, not {})".format(len(val_len)))
        self.normalized = hsv_to_rgb(*value[:3])

    @property
    def hls(self,)-> Vector4:
        """Gets a normalized color in HLS colorspace."""
        return Vector4(*rgb_to_hls(*self.normalized[:3]), self.a / 255.)

    @hls.setter
    def hls(self, value: Union[Sequence[Number], Vector3, Vector4])-> None:
        value = _flatten((int, float), *value, map_to=float)
        val_len = len(value)
        if val_len not in (3, 4):
            raise ValueError("Too many or too few values (expected 3 or 4, not {})".format(val_len))
        self.normalized = hls_to_rgb(*value[:3])

    @property
    def yiq(self) -> 'Vector4':
        """Gets or sets a normalized color in YIQ colorspace."""
        return Vector4(*rgb_to_yiq(*self.normalized[:3]), self.a / 255.)

    @yiq.setter
    def yiq(self, value: Union[Sequence[Number], Vector3 ,Vector4])-> None:
        value = _flatten((int, float), *value, map_to=float)
        val_len = len(value)
        if val_len not in (3, 4):
            raise ValueError("Too many or too few values (expected 3 or 4, not {})".format(val_len))
        self.normalized = yiq_to_rgb(*value[:3])

    def fade(self , alpha: float)-> 'Color':
        """Returns color with alpha applied, alpha goes from 0.0f to 1.0f"""
        return _rl.Fade(self , _to_float(alpha))

    def alpha(self , alpha: float)-> 'Color':
        """Returns color with alpha applied, alpha goes from 0.0f to 1.0f"""
        return _rl.ColorAlpha(self , _to_float(alpha))

    def to_int(self)-> int:
        """Returns hexadecimal value for a Color"""
        return int(self)

    def __int__(self)-> int:
        """Returns hexadecimal value for a Color"""
        return _rl.ColorToInt(self)

    def __str__(self)-> str:
        return f"({self.r} , {self.g} , {self.b} , {self.a})"

    def __len__(self)-> int:
        return 4

    def __iter__(self)-> Iterator[int]:
        return (self.r, self.g, self.b, self.a).__iter__()

    def __set(self, r: int = 0, g: int = 0, b: int = 0, a: int = 255)-> None:
        super(Color, self).__init__(
            _to_int(r),
            _to_int(g),
            _to_int(b),
            _to_int(a),
        )

ColorPtr = POINTER(Color)
wrap_function('ColorFromHSV', [Float, Float, Float], Color)
wrap_function('ColorFromNormalized', [Vector4], Color)
wrap_function('ColorAlpha', [Color, Float], Color)
wrap_function('Fade', [Color, Float], Color)
wrap_function('ColorToInt', [Color], Int)
    

class Rectangle(Struct):

    _fields_ = [
        ('_x' , c_float),
        ('_y' , c_float),
        ('_width' , c_float),
        ('_height' , c_float),
    ]
    @dispatch()
    def __init__(self)-> None:
        self.__set(0,0,0,0)
        
    @dispatch((int, float), (int, float), (int, float), (int, float))
    def __init__(self, x: float, y: float, width: float, height: float)-> None:
        self.__set(
            x,
            y,
            width,
            height,
        )
    
    @dispatch(Iterable)
    def __init__(self, itr: Union[Sequence[float], Vector4])-> None:
        self.__set(*itr)

    @dispatch(Iterable)
    def move(self, velocity: Union[Sequence[float], Vector2])-> 'Rectangle':
        self.x += velocity[0]
        self.y += velocity[1]
        return self

    @dispatch((int, float), (int, float))
    def move(self, x: int, y: int)-> 'Rectangle':
        self.x += x
        self.y += y
        return self

    def draw(
        self,
        color: Union[Sequence[int], Color] = (0, 0, 0, 255),
        origin: Union[Sequence[float], Vector2] = (0, 0),
        rotation: float = 0,
        line_thick = 1,
        outline = False,
    )-> 'Rectangle':
        """
        Draw a color-filled rectangle with pro parameters
        or Draw rectangle outline with extended parameters if outline set to True
        """
        origin = _vec2(origin)
        color = _to_color(color)
        line_thick = _to_int(line_thick)
        rotation = _to_float(rotation)

        if outline:
            _rl.DrawRectangleLinesEx(
                self,
                line_thick,
                color,
            )
        else:
            _rl.DrawRectanglePro(
                self,
                origin,
                rotation,
                color,
            )
        return self
    
    def draw_rounded(
        self, 
        roundness: float = 0.1, 
        segments: int = 0, 
        color: Union[Sequence[int], Color] = (0, 0, 0, 255),
        line_thick: int = 1,
        outline: bool=False
        )-> 'Rectangle':
        """
        Draw rectangle with rounded edges
        or Draw rectangle with rounded edges outline if outline set to True
        """
        if outline:
            _rl.DrawRectangleRoundedLines(
                self,
                _to_float(roundness),
                _to_int(segments),
                _to_int(line_thick),
                _to_color(color),
            )
        else:
            _rl.DrawRectangleRounded(
                self,
                _to_float(roundness),
                _to_int(segments),
                _to_color(color),
            )
        return self
    
    def draw_gradient(
        self,
        colors: Union[Sequence[int], Color] = (RED, YELLOW), 
        vertical: bool = True
        )-> 'Rectangle':
        """
        Draw a vertical-gradient-filled rectangle if flag set to True
        otherwise, Draw a horizontal-gradient-filled rectangle
        """
        c1 , c2  = [_to_color(color) for color in colors[:2]]
        x, y, w, h = map(int, self)
        if vertical:
            _rl.DrawRectangleGradientV(
                x,
                y,
                w,
                h,
                c1,
                c2,
            )
        else:
            _rl.DrawRectangleGradientH(
                x,
                y,
                w,
                h,
                c1,
                c2,
            )

        return self

    def draw_gradiend_ex(
        self, 
        colors: Union[Sequence[Tuple[int, int, int, int]], Sequence[Color]] = (RED, GREEN, GOLD, BLUE)
        )-> 'Rectangle':
        """Draw a gradient-filled rectangle with custom vertex colors"""
        try:
            vertex_colors = [_to_color(color) for color in colors]
            c1 , c2 , c3 , c4 = vertex_colors
        except:
            raise ValueError('You must provide a sequence of 4 colors.')
        else:
            _rl.DrawRectangleGradientEx(self, c1, c2, c3, c4)
        
        return self

    def to_vector4(self)-> 'Vector4':
        return Vector4(self.x , self.y , self.width , self.height)

    def collide_with(self, shape: Union['Rectangle', 'Circle', Vector2])-> bool:
        if isinstance(shape, Rectangle):
            """Check collision between two rectangles"""
            return _rl.CheckCollisionRecs(self , shape)
        elif isinstance(shape, Circle):
            """Check collision between circle and rectangle"""
            return _rl.CheckCollisionCircleRec(shape.center , shape.radius , self)
        elif isinstance(shape, Vector2):
            """Check if point is inside rectangle"""
            return _rl.CheckCollisionPointRec(shape , self)
        else:
            raise ValueError('You must provide either a circle , rectangle, or point')
    
    def get_collision_rect(self , other: 'Rectangle')-> 'Rectangle':
        """Get collision rectangle for two rectangles collision"""
        if not isinstance(other , Rectangle):
            raise ValueError('other parameter must be of type Rectangle')
        else:
            return _rl.GetCollisionRec(self , other)

    def __set(self, x: float = 0., y: float = 0., w: float = 0., h: float = 0.)-> None:
        super(Rectangle, self).__init__(
            _to_float(x),
            _to_float(y),
            _to_float(w),
            _to_float(h),
        )
    
    @property
    def x(self)-> float:
        return self._x
    
    @x.setter
    def x(self, d: float)-> None:
        self._x = _to_float(d)
    
    @property
    def y(self)-> float:
        return self._y
    
    @y.setter
    def y(self, d: float)-> None:
        self._y = _to_float(d)

    @property
    def width(self)-> float:
        return self._width
    @width.setter
    def width(self, w: float)-> None:
        self._width = _to_float(w)

    @property
    def height(self)-> float:
        return self._height

    @height.setter
    def height(self, h: float)-> None:
        self._height = _to_float(h)

    @property
    def right(self)-> float:
        """Gets or sets the right-most rect coordinate."""
        return self.x + self.width
    
    @right.setter
    def right(self, value: float)-> None:
        self.x = _to_float(value) - self.width
    
    @property
    def bottom(self)-> float:
        """Gets or sets the bottom-most rect coordinate."""
        return self.y + self.height
    
    @bottom.setter
    def bottom(self, value: float)-> None:
        self.y = _to_float(value) - self.height
    
    @property
    def top(self)-> float:
        return self.y

    @top.setter
    def top(self, value: float):
        self.y = value

    @property
    def left(self)-> float:
        return self.x
    
    @left.setter
    def left(self, value: float)-> None:
        self.x = value

    @property
    def size(self)-> 'Vector2':
        """Returns or Sets Vector2 contains width and height of this rectangle"""
        return Vector2(self.width , self.height)

    @size.setter
    def size(self , size: Union[Seq, Vector2])-> None:
        self.width , self.height = size[0] , size[1]

    @property
    def position(self)-> 'Vector2':
        """Returns or Sets Vector2 contains width and height of this rectangle"""
        return Vector2(self.x , self.y)

    @position.setter
    def position(self , pos: Union[Seq, Vector2])-> None:
        pos = _vec2(pos)
        self.x , self.y = pos.x, pos.y


    def __iadd__(self , other: Union[Sequence[Number] , Vector2])-> 'Rectangle':
        """
        """
        other = _flatten((int , float) , other , map_to=float)
        other_len = len(other)
        if other_len != 2:
            raise ValueError('Too many or too few initializers ({} instead of 2)'.format(other_len))
        else:
            self.x += other[0]
            self.y += other[1]
            return self

    def __isub__(self , other: Union[Sequence[Number] , Vector2])-> 'Rectangle':
        other = _flatten((int , float) , other , map_to=float)
        other_len = len(other)
        if other_len != 2:
            raise ValueError('Too many or too few initializers ({} instead of 2)'.format(other_len))
        else:
            self.x -= other[0]
            self.y -= other[1]
            return self

    def __str__(self) -> str:
        return f"({self.x} , {self.y} , {self.width} , {self.height})"

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.x} , {self.y} , {self.width} , {self.height})"

    def __iter__(self)-> Iterator[float]:
        return (self.x , self.y , self.width , self.height).__iter__()

_rl.DrawRectangle.argtypes = [Int , Int ,Int ,Int , Color]
_rl.DrawRectangle.restype = None
_rl.DrawRectanglePro.argtypes = [Rectangle , Vector2, Float , Color]
_rl.DrawRectanglePro.restype = None
_rl.DrawRectangleLinesEx.argtypes = [Rectangle , Int , Color]
_rl.DrawRectangleLinesEx.restype = None
_rl.DrawRectangleRounded.argtypes = [Rectangle , Float , Int , Color]
_rl.DrawRectangleRounded.restype = None
_rl.DrawRectangleRoundedLines.argtypes = [Rectangle , Float , Int , Int , Color]
_rl.DrawRectangleRoundedLines.restype = None
_rl.DrawRectangleGradientV.argtypes = [Int , Int , Int , Int , Color , Color]
_rl.DrawRectangleGradientV.restype = None
_rl.DrawRectangleGradientH.argtypes = [Int , Int , Int , Int , Color , Color]
_rl.DrawRectangleGradientH.restype = None
_rl.CheckCollisionRecs.argtypes = [Rectangle , Rectangle]
_rl.CheckCollisionRecs.restype = Bool
_rl.GetCollisionRec.argtypes = [Rectangle , Rectangle]
_rl.GetCollisionRec.restype = Rectangle
_rl.CheckCollisionPointRec.argtypes = [Vector2 , Rectangle]
_rl.CheckCollisionPointRec.restype = Bool
_rl.CheckCollisionCircleRec.argtypes = [Vector2 , Float , Rectangle]
_rl.CheckCollisionCircleRec.restype = Bool

RectanglePtr = POINTER(Rectangle)

class Circle(object):

    def __init__(
        self, 
        center: Union[Sequence[int], Vector2] = (0, 0), 
        radius: float= 1.0, 
        color: Union[Sequence[int], 
        Vector4, Color]= (0, 0, 0, 255)
        )-> None:
        super().__init__()
        self._center = _vec2(center)
        self._radius = _to_float(radius)
        self._color = _to_color(color)

    @property
    def x(self)-> int:
        return _to_int(self._center.x)
    
    @x.setter
    def x(self, i: int)-> None:
        self._center.x = _to_int(i)
    
    @property
    def y(self)-> int:
        return _to_int(self._center.y)
    
    @y.setter
    def y(self, i: int)-> None:
        self._center.y = _to_int(i)

    @property
    def radius(self)-> float:
        return self._radius
    
    @radius.setter
    def radius(self, r: float)-> None:
        self._radius = _to_float(r)
    
    @property
    def center(self)-> Vector2:
        return self._center

    @center.setter
    def center(self, c: Union[Sequence[float], Vector2])-> None:
        self._center = _vec2(c)

    @property
    def color(self)-> Color:
        return self._color
    
    @color.setter
    def color(self, c: Union[Sequence[int], Vector4, Color]):
        self._color = _to_color(c)

    @property
    def left(self)-> int:
        return self.x - self.radius
    
    @property
    def right(self)-> int:
        return self.x + self.radius

    @property
    def top(self)-> int:
        return self.y - self.radius
    
    @property
    def bottom(self)-> int:
        return self.y + self.radius

    @dispatch(Iterable)
    def move(self, velocity: Union[Sequence[int], Vector2])-> 'Circle':
        self.x += velocity[0]
        self.y += velocity[1]
        return self
    
    @dispatch((int, float), (int, float))
    def move(self, x: int, y: int)-> 'Circle':
        self.x += x
        self.y += y
        return self

    def draw(self, outline=False)-> 'Circle':
        """
        Draw a color-filled circle
        or Draw circle outline if outline set to True
        """
        if outline:
            _rl.DrawCircleLines(
                self.x,
                self.y,
                self.radius,
                self.color,
            )
        else:
            _rl.DrawCircleV(
                self.center,
                self.radius,
                self.color,
            )
        return self

    def draw_sector(self, startAngle: float = 0.0, endAngle: float = 60.0, segments: int = 1, outline: bool=False)-> 'Circle':
        """
        Draw a piece of a circle
        or Draw circle sector outline if outline set to True
        """
        if outline:
            _rl.DrawCircleSectorLines(
                self.center,
                self.radius,
                _to_float(startAngle),
                _to_float(endAngle),
                _to_int(segments),
                self.color
            )
        else:
            _rl.DrawCircleSector(
                self.center,
                self.radius,
                _to_float(startAngle),
                _to_float(endAngle),
                _to_int(segments),
                self.color
            )
        return self

    def draw_gradient(self, colors: Union[List[Sequence[int]], Sequence[Color]] = [RED, YELLOW_GREEN])-> 'Circle':
        c1, c2 = [_to_color(color) for color in colors][:2]
        _rl.DrawCircleGradient(
            self.x,
            self.y,
            self.radius,
            c1,
            c2,
        )
        return self

    def collide_with(self, shape: Union['Circle', Rectangle, Vector2])-> bool:
        if isinstance(shape, Rectangle):
            """Check collision between circle and rectangle"""
            return _rl.CheckCollisionCircleRec(
                self.center,
                self.radius,
                shape
            )
        elif isinstance(shape, Circle):
            """Check collision between two circles"""
            return _rl.CheckCollisionCircles(
                self.center, 
                self.radius, 
                shape.center, 
                shape.radius)
        elif isinstance(shape, Vector2):
           """Check if point is inside circle"""
           return _rl.CheckCollisionPointCircle(
                shape,
                self.center,
                self.radius,
            )
        else:
            raise ValueError('You must provide either a circle , rectangle, or point')

wrap_function('CheckCollisionCircles', [Vector2, Float, Vector2, Float], Bool)
wrap_function('CheckCollisionPointCircle', [Vector2, Vector2, Float], Bool)
wrap_function('CheckCollisionCircleRec', [Vector2, Float, Rectangle], Bool)
wrap_function('DrawCircleSectorLines', [Vector2, Float, Float, Float, Int, Color])
wrap_function('DrawCircleSector', [Vector2, Float, Float, Float, Int, Color])
wrap_function('DrawCircleGradient', [Int, Int, Float, Color, Color])
wrap_function('DrawCircleV', [Vector2, Float, Color])
wrap_function('DrawCircleLines', [Int, Int, Float, Color])

    

class Image(Struct):
    _fields_ = [
        ('_data' , VoidPtr), #Image raw data
        ('_width' , Int),    #Image base width
        ('_height' , Int),   #Image base height
        ('_mipmaps' , Int),  #Mipmap levels, 1 by default
        ('_format' , Int),   #Data format (PixelFormat type)
    ]
    """
    Image type, bpp always RGBA (32bit)
    NOTE: Data stored in CPU memory (RAM)
    """
    def __init__(self, image: 'Image') -> None:
        print('__init__')
        if isinstance(image, Image):
            self.__set(image)
        else:
            raise ValueError('Invalid argument')
       
    # Image manipulation functions
    def copy(self)-> 'Image':
        """Create an image duplicate (useful for transformations)"""
        return _rl.ImageCopy(self)

    def cut(self, rec: Union[Seq, Rectangle])-> 'Image':
        """Create an image from another image piece defined by a rectangle"""
        return _rl.ImageFromImage(self, _rect(rec))

    def convert_to(self, new_format: Union[PixelFormat, int])-> 'Image':
        """Convert image data to desired format"""
        _rl.ImageFormat(byref(self), _to_int(new_format))
        return self

    def to_pot(self, fill: Union[Seq, Color])-> 'Image':
        """Convert image to POT (power-of-two)"""
        _rl.ImageToPOT(byref(self), _to_color(fill))
        return self

    def alpha_mask(self, alphaMask: 'Image')-> 'Image':
        """Apply alpha mask to image"""
        _rl.ImageAlphaMask(byref(self), alphaMask)
        return self

    def crop(self, crop_rect: Union[Seq, Rectangle])-> 'Image':
        """Crop the image to the area defined by a rectangle"""
        _rl.ImageCrop(byref(self), _rect(crop_rect))
        return self

    def alpha_crop(self, threshold: float)-> 'Image':
        """Crop image depending on alpha value"""
        _rl.ImageAlphaCrop(byref(self), _to_float(threshold))
        return self

    def alpha_clear(self, color: Union[Seq, Color], threshold: float)-> 'Image':
        """Clear alpha channel to desired color"""
        _rl.ImageAlphaClear(byref(self), _to_color(color), _to_float(threshold))
        return self

    def alpha_pre_multiply(self)-> 'Image':
        """Premultiply alpha channel"""
        _rl.ImageAlphaPremultiply(byref(self))
        return self

    def resize(self, new_width: int, new_height: int)-> 'Image':
        """Resize image (Bicubic scaling algorithm)"""
        _rl.ImageResize(byref(self), _to_int(new_width), _to_int(new_height))
        return self

    def resize_nn(self, new_width: int, new_height: int)-> 'Image':
        """Resize image (Nearest-Neighbor scaling algorithm)"""
        _rl.ImageResizeNN(byref(self), _to_int(new_width), _to_int(new_height))
        return self

    def resize_canvas(
        self,
        new_width: int,
        new_height: int,
        offsetX: int,
        offsetY: int,
        color: Union[Seq, Color])-> 'Image':
        """Resize canvas and fill with color"""
        _rl.ImageResizeCanvas(
            byref(self),
            _to_int(new_width),
            _to_int(new_height),
            _to_int(offsetX),
            _to_int(offsetY),
            _to_color(color)
        )
        return self

    def gen_mipmaps(self)-> 'Image':
        """Generate all mipmap levels for a provided image"""
        _rl.ImageMipmaps(byref(self))
        return self

    def dither(self, r_bpp: int, g_bpp: int, b_bpp: int, a_bpp: int) -> 'Image':
        """Dither image data to 16bpp or lower (Floyd-Steinberg dithering)"""
        _rl.ImageDither(
            byref(self),
            _to_int(r_bpp),
            _to_int(g_bpp),
            _to_int(b_bpp),
            _to_int(a_bpp))
        return self

    def flip_vertically(self)-> 'Image':
        """Flip image vertically"""
        _rl.ImageFlipVertical(byref(self))
        return self

    def flip_horizontally(self)-> 'Image':
        """Flip image horizontally"""
        _rl.ImageFlipHorizontal(byref(self))
        return self

    def rotate_cw(self)-> 'Image':
        """Rotate image clockwise 90deg"""
        _rl.ImageRotateCW(byref(self))
        return self

    def rotate_ccw(self)-> 'Image':
        """Rotate image counter-clockwise 90deg"""
        _rl.ImageRotateCCW(byref(self))
        return self

    def color_tint(self, color: Union[Seq, Color])-> 'Image':
        """Modify image color: tint"""
        _rl.ImageColorTint(byref(self), _to_color(color))
        return self

    def color_invert(self)-> 'Image':
        """Modify image color: invert"""
        _rl.ImageColorInvert(byref(self))
        return self

    def color_grayscale(self)-> 'Image':
        """Modify image color: grayscale"""
        _rl.ImageColorGrayscale(byref(self))
        return self

    def color_contrast(self, contrast: float)-> 'Image':
        """Modify image color: contrast (-100 to 100)"""
        _rl.ImageColorContrast(byref(self), _to_float(contrast))
        return self

    def color_brightness(self, brightness: int)-> 'Image':
        """Modify image color: brightness (-255 to 255)"""
        _rl.ImageColorBrightness(byref(self), _to_int(brightness))
        return self

    def color_replace(self, color: Union[Seq, Color], replace: Union[Seq, Color])-> 'Image':
        """Modify image color: replace color"""
        _rl.ImageColorReplace(byref(self), _to_color(color), _to_color(replace))
        return self

    def load_colors(self)-> 'ColorPtr':
        """Load color data from image as a Color array (RGBA - 32bit)"""
        return _rl.LoadImageColors(self)

    def unload_colors(self, colors: ColorPtr)-> 'Image':
        """Unload color data loaded with load_colors()"""
        _rl.UnloadImageColors(colors)

    def get_alpha_border(self, threshold: float)-> 'Rectangle':
        """Get image alpha border rectangle"""
        return _rl.GetImageAlphaBorder(self, _to_float(threshold))

    @property
    def size(self)-> Vector2:
        """Retrieve the width and height of the image."""
        return Vector2(self.width, self.height)

    #Image drawing functions
    def draw_pixel(self, pos: Union[Seq, Vector2], color: Union[Seq, Color])-> 'Image':
        """Draw pixel within an image"""
        _rl.ImageDrawPixelV(byref(self), _vec2(pos), _to_color(color))
        return self

    def draw_line(
        self,
        start: Union[Seq, Vector2],
        end: Union[Seq, Vector2],
        color: Union[Seq, Color]
        )-> 'Image':
        """Draw line within an image"""
        _rl.ImageDrawLineV(byref(self), _vec2(start), _vec2(end), _to_color(color))
        return self

    def draw_circle(
        self,
        center: Union[Seq, Vector2],
        radius: int,
        color: Union[Seq, Color]
        )-> 'Image':
        """Draw circle within an image"""
        _rl.ImageDrawCircleV(
            byref(self),
            _vec2(center),
            _to_int(radius),
            _to_color(color),
        )
        return self

    def draw_rectangle(
        self,
        position: Union[Seq, Vector2],
        size: Union[Seq, Vector2],
        color: Union[Seq, Color]
    )-> 'Image':
        """Draw rectangle within an image"""
        _rl.ImageDrawRectangleV(
            byref(self),
            _vec2(position),
            _vec2(size),
            _to_color(color)
        )
        return self

    def draw_rect(
        self,
        rect: Union[Seq, Vector4, Rectangle , Sequence[Vector2]],
        color: Union[Seq , Color]
        )-> 'Image':
        """Draw rectangle within an image"""
        _rl.ImageDrawRectangleRec(
            byref(self),
            _rect(_flatten((float, int), rect, map_to=float)),
            _to_color(color)
        )
        return self

    def draw_rect_lines(
        self,
        rec: Union[Seq, Vector4, Rectangle, Sequence[Vector2]],
        thick: int,
        color: Union[Seq, Color]
        )-> 'Image':
        """Draw rectangle lines within an image"""
        _rl.ImageDrawRectangleLines(
            byref(self),
            _rect(_flatten((int, float), rec, map_to=float)),
            _to_int(thick),
            _to_color(color)
        )
        return self

    def draw_text(
        self,
        text: str,
        pos: Union[Seq, Vector2],
        font_size: int,
        color: Union[Seq, Color]
        )-> 'Image':
        """Draw text (using default font) within an image"""
        pos = _flatten((int, float), pos, map_to=int)
        _rl.ImageDrawText(
            byref(self),
            _to_byte_str(text),
            pos[0],
            pos[1],
            _to_int(font_size),
            _to_color(color))
        return self

    def draw_text_ex(
        self,
        font: 'Font',
        text: str,
        position: Union[Seq, Vector2],
        font_size: float,
        tint: Union[Seq, Color]
        )-> 'Image':
        """Draw text (custom sprite font) within an image"""
        _rl.ImageDrawTextEx(
            byref(self),
            font,
            _to_byte_str(text),
            _vec2(position),
            _to_float(font_size),
            _to_color(tint)
        )
        return self

    def get_texture(self)-> 'Texture':
        """Load texture from image data"""
        return _rl.LoadTextureFromImage(self)

    # Image generation functions
    @staticmethod
    def color(width: int, height: int, color: Union[Sequence[int], Color])-> 'Image':
        """Generate image: plain color"""
        return _rl.GenImageColor(
            _to_int(width),
            _to_int(height),
            _to_color(color)
        )

    @staticmethod
    def gradient_v(
        width: int,
        height: int,
        top: Union[Seq, Color],
        bottom: Union[Seq, Color])-> 'Image':
        """Generate image: vertical gradient"""
        return _rl.GenImageGradientV(
            _to_int(width),
            _to_int(height),
            _to_color(top),
            _to_color(bottom)
            )

    @staticmethod
    def gradient_h(
        width: int,
        height: int,
        left: Union[Seq, Color],
        right: Union[Seq, Color])-> 'Image':
        """Generate image: horizontal gradient"""
        return _rl.GenImageGradientH(
            _to_int(width),
            _to_int(height),
            _to_color(left),
            _to_color(right)
            )

    @staticmethod
    def gradient_radial(
        width: int,
        height: int,
        density: float,
        inner: Union[Color, Seq],
        outer: Union[Color, Seq])-> 'Image':
        """Generate image: radial gradient"""
        return _rl.GenImageGradientRadial(
            _to_int(width),
            _to_int(height),
            _to_float(density),
            _to_color(inner),
             _to_color(outer))

    @staticmethod
    def checked(
        width: int,
        height: int,
        checks_x: int,
        checks_y: int,
        col1: Union[Color, Seq],
        col2: Union[Color, Seq]) -> 'Image':
        """Generate image: checked"""
        return _rl.GenImageChecked(
            _to_int(width),
            _to_int(height),
            _to_int(checks_x),
            _to_int(checks_y),
            _to_color(col1),
            _to_color(col2))

    @staticmethod
    def white_noise(width: int, height: int, factor: float)-> 'Image':
        """Generate image: white noise"""
        return _rl.GenImageWhiteNoise(_to_int(width), _to_int(height), _to_float(factor))

    @staticmethod
    def perlin_noise(
        width: int,
        height: int,
        offset_x: int,
        offset_y: int,
        scale: float)-> 'Image':
        """Generate image: perlin noise"""
        return _rl.GenImagePerlinNoise(
            _to_int(width),
            _to_int(height),
            _to_int(offset_x),
            _to_int(offset_y),
            _to_float(scale)
        )

    @staticmethod
    def cellular(width: int, height: int, tile_size: int) -> 'Image':
        """Generate image: cellular algorithm. Bigger tileSize means bigger cells"""
        return _rl.GenImageCellular(
            _to_int(width),
            _to_int(height),
            _to_int(tile_size)
        )

    @staticmethod
    def text(text: str, font_size: int, color: Union[Seq, Color])-> 'Image':
        """Create an image from text (default font)"""
        return _rl.ImageText(
            _to_byte_str(text),
            _to_int(font_size),
            _to_color(color)
        )

    @staticmethod
    def text_ex(
        font: 'Font',
        text: str,
        font_size: float,
        spacing: float,
        tint: Union[Seq, Color]
        )-> 'Image':
        """Create an image from text (custom sprite font)"""
        return _rl.ImageTextEx(
            font,
            _to_byte_str(text),
            _to_float(font_size),
            _to_float(spacing),
            _to_color(tint)
        )

    @staticmethod
    def get_screen_data()-> 'Image':
        """Get pixel data from screen buffer and return an Image (screenshot)"""
        return _rl.GetScreenData()

    @staticmethod
    def load_from_texture(texture)-> 'Image':
        """Get pixel data from GPU texture and return an Image"""
        return  _rl.GetTextureData(texture)

    @staticmethod
    def load_image(file_name: Union[str , Path])-> 'Image':
        """Load image from file into CPU memory (RAM)"""
        return _rl.LoadImage(_to_byte_str(str(file_name)))

    @staticmethod
    def load_image_anim(file_name: str)-> 'Image':
        """Load image sequence from file (frames appended to image.data)"""
        frames = Int(0)
        img = _rl.LoadImageAnim(_to_byte_str(file_name), byref(frames))
        print(frames)
        return img

    @staticmethod
    def load_image_raw(
        file_name: str,
        width: int,
        height: int,
        format: Union[PixelFormat, int],
        header_size: int)-> 'Image':
        """Load image from RAW file data"""
        return _rl.LoadImageRaw(
            _to_byte_str(file_name),
            _to_int(width),
            _to_int(height),
            _to_int(format),
            _to_int(header_size)
        )

    @staticmethod
    def __from_memory(file_type: str, file_data: List[str], data_size: int)-> 'Image':
        """Load image from memory buffer, fileType refers to extension: i.e. ".png"""
        # data_len = len(file_data)
        # dataArray = UChar * data_len
        # return _rl.LoadImageFromMemory(file_type, dataArray(*file_data), data_size)
        pass

    def unload(self)-> None:
        """Unload image from CPU memory (RAM)"""
        _rl.UnloadImage(self)

    def export(self, filename: Union[str , Path])-> bool:
        """Export image data to file, returns true on success"""
        return _rl.ExportImage(self, _to_byte_str(str(filename)))

    def export_as_code(self, filename: Union[str, Path])-> bool:
        """Export image as code file defining an array of bytes, returns true on success"""
        return _rl.ExportImageAsCode(self, _to_byte_str(filename))

    def __set(self, img)-> None:
        super(Image, self).__init__(
            img.data,
            img.width,
            img.height,
            img.mipmaps,
            img.format
        )

    def clear_background(self, color: Color)-> 'Image':
        _rl.ImageClearBackground(self, color)
        return self
    
    @property
    def width(self)-> int:
        return self._width

    @width.setter
    def width(self, value)-> None:
        self._width = _to_int(value)
    
    @property
    def height(self)-> int:
        return self._height

    @height.setter
    def height(self, value)-> None:
        self._height = _to_int(value)
    
    @property
    def size(self)-> Vector2:
        """Retrieves or sets the width and height of the texture."""
        return Vector2(self.width, self.height)
    
    @size.setter
    def size(self, s: Union[Sequence[int], Vector2])-> None:
        self.width, self.height = s

    @property
    def format(self)-> int:
        return self._format
    
    @format.setter
    def format(self, value: Union[PixelFormat, int])-> None:
        self.convert_to(value)
    
ImagePtr = POINTER(Image)

wrap_function('LoadImage', [CharPtr], Image)
wrap_function('ImageCopy',[Image], Image)
wrap_function('LoadImageRaw', [CharPtr, Int, Int, Int, Int], Image)
wrap_function('LoadImageAnim', [CharPtr, IntPtr], Image)
wrap_function('LoadImageFromMemory', [CharPtr, UCharPtr , Int], Image)
wrap_function('UnloadImage', [Image])
wrap_function('ExportImage', [Image, CharPtr], Bool)
wrap_function('ExportImageAsCode', [Image, CharPtr], Bool)

_rl.GenImageColor.argtypes = [Int, Int, Color]
_rl.GenImageColor.restype = Image
_rl.GenImageGradientV.argtypes = [Int, Int, Color, Color]
_rl.GenImageGradientV.restype = Image
_rl.GenImageGradientH.argtypes = [Int, Int, Color, Color]
_rl.GenImageGradientH.restype = Image
_rl.GenImageGradientRadial.argtypes = [Int, Int, Float, Color, Color]
_rl.GenImageGradientRadial.restype = Image
_rl.GenImageChecked.argtypes = [Int, Int, Int, Int, Color, Color]
_rl.GenImageChecked.restype = Image
_rl.GenImageWhiteNoise.argtypes = [Int, Int, Float]
_rl.GenImageWhiteNoise.restype = Image
_rl.GenImagePerlinNoise.argtypes = [Int, Int, Int, Int, Float]
_rl.GenImagePerlinNoise.restype = Image
_rl.GenImageCellular.argtypes = [Int, Int, Int]
_rl.GenImageCellular.restype = Image
_rl.ImageFromImage.argtypes = [Image, Rectangle]
_rl.ImageFromImage.restype = Image
_rl.ImageText.argtypes = [CharPtr, Int, Color]
_rl.ImageText.restype = Image
_rl.GetScreenData.argtypes = None
_rl.GetScreenData.restype = Image
_rl.ImageClearBackground.argtypes = [ImagePtr, Color]
_rl.ImageClearBackground.restype = None
_rl.ImageToPOT.argtypes = [ImagePtr, Color]
_rl.ImageToPOT.restype = None
wrap_function('ImageFormat', [ImagePtr, Int])
_rl.ImageAlphaMask.argtypes = [ImagePtr, Image]
_rl.ImageAlphaMask.restype = None
_rl.ImageCrop.argtypes = [ImagePtr, Rectangle]
_rl.ImageCrop.restype = None
_rl.ImageAlphaCrop.argtypes = [ImagePtr, Float]
_rl.ImageAlphaCrop.restype = None
_rl.ImageAlphaClear.argtypes = [ImagePtr, Color, Float]
_rl.ImageAlphaClear.restype = None
_rl.ImageAlphaPremultiply.argtypes = [ImagePtr]
_rl.ImageAlphaPremultiply.restype = None
_rl.ImageResize.argtypes = [ImagePtr, Int, Int]
_rl.ImageResize.restype = None
_rl.ImageResizeNN.argtypes = [ImagePtr, Int, Int]
_rl.ImageResizeNN.restype = None
_rl.ImageResizeCanvas.argtypes = [ImagePtr, Int, Int, Int, Int, Color]
_rl.ImageResizeCanvas.restype = None
_rl.ImageMipmaps.argtypes = [ImagePtr]
_rl.ImageMipmaps.restype = None
_rl.ImageDither.argtypes = [ImagePtr, Int, Int, Int, Int]
_rl.ImageDither.restype = None
_rl.ImageFlipVertical.argtypes = [ImagePtr]
_rl.ImageFlipVertical.restype = None
_rl.ImageFlipHorizontal.argtypes = [ImagePtr]
_rl.ImageFlipHorizontal.restype = None
_rl.ImageRotateCW.argtypes = [ImagePtr]
_rl.ImageRotateCW.restype = None
_rl.ImageRotateCCW.argtypes = [ImagePtr]
_rl.ImageRotateCCW.restype = None
_rl.ImageColorTint.argtypes = [ImagePtr, Color]
_rl.ImageColorTint.restype = None
_rl.ImageColorInvert.argtypes = [ImagePtr]
_rl.ImageColorInvert.restype = None
_rl.ImageColorGrayscale.argtypes = [ImagePtr]
_rl.ImageColorGrayscale.restype = None
_rl.ImageColorContrast.argtypes = [ImagePtr, Float]
_rl.ImageColorContrast.restype = None
_rl.ImageColorBrightness.argtypes = [ImagePtr, Int]
_rl.ImageColorBrightness.restype = None
_rl.LoadImageColors.argtypes = [Image]
_rl.LoadImageColors.restype = ColorPtr
_rl.UnloadImageColors.argtypes = [ColorPtr]
_rl.UnloadImageColors.restype = None
_rl.GetImageAlphaBorder.argtypes = [Image, Float]
_rl.GetImageAlphaBorder.restype = Rectangle
_rl.ImageDrawPixelV.argtypes = [ImagePtr, Vector2, Color]
_rl.ImageDrawPixelV.restype = None
_rl.ImageDrawLineV.argtypes = [ImagePtr, Vector2, Vector2, Color]
_rl.ImageDrawLineV.restype = None
_rl.ImageDrawCircle.argtypes = [ImagePtr, Vector2, Int, Color]
_rl.ImageDrawCircle.restype = None
_rl.ImageDrawRectangleV.argtypes = [ImagePtr, Vector2, Vector2, Color]
_rl.ImageDrawRectangleV.restype = None
_rl.ImageDrawRectangleRec.argtypes = [ImagePtr, Rectangle, Color]
_rl.ImageDrawRectangleRec.restype = None
_rl.ImageDrawRectangleLines.argtypes = [ImagePtr, Rectangle, Int, Color]
_rl.ImageDrawRectangleLines.restype = None

    

class Texture(Struct):
    _fields_ = [
        ('_id' , UInt),
        ('_width' , Int),
        ('_height' , Int),
        ('_mipmaps' , Int),
        ('_format' , Int),
    ]
    def __init__(self, texture: 'Texture')-> None:
        if isinstance(texture, Texture):
            self.__set(texture)
        else:
            raise ValueError('Invalid argument')
    
    def gen_texture_mipmaps(self)-> 'Texture':
        """Generate GPU mipmaps for a texture"""
        _rl.GenTextureMipmaps(self)
        return self

    def set_filter(self, filter: Union[TextureFilter, int])-> 'Texture':
        """Set texture scaling filter mode"""
        _rl.SetTextureFilter(self, filter)
        return self

    def set_wrap(self, wrap: Union[TextureWrap, int])-> 'Texture':
        """Set texture wrapping mode"""
        _rl.SetTextureWrap(self, wrap)
        return self

    def unload(self)-> None:
        """Unload texture from GPU memory (VRAM)"""
        return _rl.UnloadTexture(self)

    def get_image(self)-> Image:
        """Get pixel data from GPU texture and return an Image"""
        return _rl.GetTextureData(self)

    @staticmethod
    def from_image(image: Image)-> 'Image':
        """Load texture from image data"""
        if isinstance(image, Image):
            return _rl.LoadTextureFromImage(image)
        else:
            raise TypeError('You must provide an Image object')

    @staticmethod
    def texture_cubemap(image: Image , layout: Union[CubemapLayout, int])-> 'Texture':
        """Load cubemap from image, multiple image cubemap layouts supported"""
        if isinstance(image, Image):
            return _rl.LoadTextureCubemap(image , layout)
        else:
            raise TypeError('You must provide an Image object')

    @staticmethod
    def from_file(filename: Union[str, Path])-> 'Texture':
        """Load texture from file into GPU memory (VRAM)"""
        if isinstance(filename, (str, Path)):
            return _rl.LoadTexture(_to_byte_str(str(filename)))
        else:
            raise TypeError('You must provide a Path or str')

    def draw(
        self,
        position: Union[Seq, Vector2] = (0 , 0),
        color: Union[Seq, Color]=(255 , 255 , 255 , 255),
        rotation: float = 0.0,
        scale: float = 1.0,
        )-> 'Texture':
        """Draw a Texture2D with extended parameters"""
        _rl.DrawTextureEx(
            self,
            _vec2(position),
            _to_float(rotation),
            _to_float(scale),
            _to_color(color))
        return self

    def draw_rec(
        self,
        source: Union[Seq, Rectangle],
        position: Union[Seq, Vector2],
        color: Union[Seq, Color]=(255 , 255 , 255 , 255)
        )-> 'Texture':
        """Draw a part of a texture defined by a rectangle"""
        _rl.DrawTextureRec(self, _rect(source), _vec2(position), _to_color(color))
        return self

    def draw_quad(
        self,
        tiling: Union[Seq, Vector2],
        offset: Union[Seq, Vector2],
        quad: Union[Seq, Rectangle],
        tint: Union[Seq, Color]
        )-> 'Texture':
        """Draw texture quad with tiling and offset parameters"""
        _rl.DrawTextureQuad(
            self,
            _vec2(tiling),
            _vec2(offset), _rect(quad),
            _to_color(tint))
        return self

    def update(self, pixels: Union[VoidPtr, bytes])-> None:
        """Update GPU texture with new data"""
        return _rl.UpdateTexture(self, pixels)

    @property
    def width(self)-> int:
        return self._width

    @width.setter
    def width(self, value)-> None:
        self._width = _to_int(value)
    
    @property
    def height(self)-> int:
        return self._height

    @height.setter
    def height(self, value)-> None:
        self._height = _to_int(value)
    
    @property
    def size(self)-> Vector2:
        """Retrieves or sets the width and height of the texture."""
        return Vector2(self.width, self.height)
    
    @size.setter
    def size(self, s: Union[Sequence[int], Vector2])-> None:
        self.width, self.height = s

    def __set(self , texture):
        super(Texture, self).__init__(
            texture.id,
            texture.width,
            texture.height,
            texture.mipmaps,
            texture.format)

Texture2D = Texture

class TextureCubemap(Texture):
    pass

wrap_function('UnloadTexture', [Texture])
wrap_function('LoadTexture', [CharPtr], Texture)
wrap_function('LoadTextureFromImage', [Image], Texture)
wrap_function('LoadTextureCubemap', [Image , Int], TextureCubemap)
wrap_function('GenTextureMipmaps', [Texture])
wrap_function('GetTextureData', [Texture], Image)
wrap_function('SetTextureFilter', [Texture, Int])
wrap_function('SetTextureWrap', [Texture, Int])
# wrap_function('DrawTexture', [Texture, Int, Int, Color])
# wrap_function('DrawTextureV', [Texture, Vector2, Color])
wrap_function('DrawTextureEx', [Texture, Vector2, Float, Float, Color])
wrap_function('DrawTextureRec', [Texture, Rectangle, Vector2, Color])
wrap_function('DrawTextureQuad', [Texture, Vector2, Vector2, Rectangle, Color])
wrap_function('ImageDrawText', [ImagePtr, CharPtr, Int, Int, Int, Color])


class RenderTexture(Struct):
    """
    RenderTexture type, for texture rendering
    """
    _fields_ = [
        ('_id' , UInt),          #OpenGL framebuffer object id
        ('_texture' , Texture),  #Color buffer attachment texture
        ('_depth' , Texture),    #Depth buffer attachment texture
    ]
    
    def __init__(self, id: int, texture: Texture, depth: Texture)-> None:
        self.__set(id, texture, depth)
        
    def begin_mode(self)-> 'RenderTexture':
        """Initializes render texture for drawing"""
        _rl.BeginTextureMode(self)
        return self

    def end_mode(self)-> 'RenderTexture':
        """Ends drawing to render texture"""
        _rl.EndTextureMode()
        return self

    @classmethod
    def load(cls, width: int, height: int)-> 'RenderTexture':
        """Load texture for rendering (framebuffer)"""
        data = _rl.LoadRenderTexture(_to_int(width), _to_int(height))
        return cls(data.id, data.texture, data.depth)

    def unload(self)-> None:
        """Unload render texture from GPU memory (VRAM)"""
        _rl.UnloadRenderTexture(self)

    def __set(self, id: int, texture: Texture, depth: Texture):
        super(RenderTexture, self).__init__(
            id,
            texture,
            depth,
        )

    @property
    def id(self)-> int:
        return self._id
    
    @id.setter
    def id(self, i: int)-> None:
        self._id = _to_int(i)

    @property
    def texture(self)-> Texture:
        return self._texture
    
    @texture.setter
    def texture(self, t: Texture)-> None:
        if isinstance(t, Texture):
            self._texture = t
    
    @property
    def depth(self)-> Texture:
        return self._depth

    @depth.setter
    def depth(self, d: Texture)-> None:
        if isinstance(d, Texture):
            self._depth = d
    
    def __enter__(self)-> 'RenderTexture':
        self.begin_mode()
        return self
    def __exit__(self, *args)-> None:
        self.end_mode()
wrap_function('LoadRenderTexture', argtypes=[Int, Int], restype=RenderTexture)
wrap_function('UnloadRenderTexture', argtypes=[RenderTexture], restype=None)
wrap_function('BeginTextureMode', [RenderTexture])
wrap_function('EndTextureMode')

RenderTexture2D = RenderTexture


class _NPatchInfo(Struct):
    """
    N-Patch layout info
    """
    _fields_ = [
        ('source' , Rectangle), #Texture source rectangle
        ('left' , Int), #Left border offset
        ('top' , Int), #Top border offset
        ('right' , Int), #Right border offset
        ('bottom' , Int), #Bottom border offset
        ('layout' , Int), #Layout of the n-patch: 3x3, 1x3 or 3x1
    ]

class NPatchInfo(_NPatchInfo):
    pass


class _CharInfo(Struct):
    """
    Font character info
    """
    _fields_ = [
        ('value' , Int), #Character value (Unicode)
        ('offsetX' , Int), #Character offset X when drawing
        ('offsetY' , Int), #Character offset Y when drawing
        ('advanceX' , Int), #Character advance position X
        ('image' , Image), #Character image data
    ]

class CharInfo(_CharInfo):
    pass

CharInfoPtr = POINTER(CharInfo)
   

class Font(Struct):
    """
    Font type, includes texture and charSet array data
    """
    _fields_=[
        ('baseSize' , Int), #Base size (default chars height)
        ('charsCount' , Int), #Number of characters
        ('charsPadding' , Int), #Padding around the chars
        ('texture' , Texture2D), #Characters texture atlas
        ('recs' , RectanglePtr), #Characters rectangles in texture
        ('chars' , CharInfoPtr), #Characters info data
    ]
    # TODO fix this constructor : make it a copy constructor only
    def __init__(
        self,
        data: Union[str , Path, 'Font'] = None,) -> None:
        if data is None:
            self.__set(Font.get_default())
        elif isinstance(data, Font):
            self.__set(data)
        elif isinstance(data, (str, Path)):
            self.__set(Font.__load_font(str(data)))
        else:
            raise ValueError('')

    @staticmethod
    def get_default()-> 'Font':
        """Get the default Font"""
        return _rl.GetFontDefault()

    @staticmethod
    def load_from_file(filename: Union[str, Path])-> 'Font':
        """Load font from file into GPU memory (VRAM)"""
        return _rl.LoadFont(_to_byte_str(str(filename)))

    @staticmethod
    def __loaf_font_ex(
        filename: Union[str, Path],
        font_size: int,
        font_chars: int,
        chars_count: int)-> 'Font':
        """Load font from file with extended parameters"""
        return _rl.LoadFontEx(
            _to_byte_str(str(filename)),
            _to_int(font_size),
            byref(Int(font_chars)),
            _to_int(chars_count),
        )

    def unload(self)-> None:
        return _rl.UnloadFont(self)

    def __set(self, data):
        super(Font, self).__init__(
            data.baseSize,
            data.charsCount,
            data.charsPadding,
            data.texture,
            data.recs,
            data.chars,
        )

wrap_function('GetFontDefault', None, Font)
wrap_function('LoadFont', None , Font)
wrap_function('UnloadFont', [Font])
wrap_function('LoadFontEx', [CharPtr, Int, IntPtr, Int], Font)
class Text(object):
    def __init__(
        self,
        text: str = '',
        font: Font = Font.get_default(),
        size: float = 10.5,
        spacing: float = 0.0,
        color = (0 , 0 , 0 , 255)) -> None:

        self._text = _to_byte_str(text)
        self._font = font
        self._size = _to_float(size)
        self._color = _to_color(color)
        self._spacing = _to_float(spacing)
        super().__init__()

    @property
    def text(self)-> str:
        return self._text

    @text.setter
    def text(self, text):
        if not isinstance(text, str):
            raise ValueError()
        self._text = _to_byte_str(text)

    @property
    def color(self)-> Color:
        return self._color

    @color.setter
    def color(self, color: Union[Seq, Color])-> None:
        self._color = _to_color(color)

    @property
    def size(self)-> float:
        return self._size
    @size.setter
    def size(self, s: float)-> None:
        self._size = _to_float(s)

    def render(self, pos: Union[Seq, Vector2])-> None:
        """Draw text using font and additional parameters"""
        pos = _vec2(pos)
        _rl.DrawTextEx(
            self._font,
            self._text,
            pos,
            self._size,
            self._spacing,
            self._color
        )

    @python_wrapper('DrawTextRec', [Font, CharPtr, Rectangle, Float, Float, Bool, Color])
    def render_in_rec(
        self,
        rect: Union[Seq, Rectangle, Vector4, Sequence[Vector2]],
        word_wrap: bool = True)-> None:
        """Draw text using font inside rectangle limits"""
        _rl.DrawTextRec(
            self._font,
            self._text,
            _rect(rect),
            self._size,
            self._spacing,
            word_wrap,
            self._color
        )

    @property
    @python_wrapper('MeasureTextEx', [Font, CharPtr, Float, Float], Vector2)
    def width(self)-> int:
        return _rl.MeasureTextEx(self._font, self._text, self._size, self._spacing).x
    @property
    def height(self)-> int:
        return _rl.MeasureTextEx(self._font, self._text, self._size, self._spacing).y

# wrap_function('GetFontDefault', Font, None)
# wrap_function('LoadFont', Font, [CharPtr])
_rl.ImageTextEx.argtypes = [Font, CharPtr, Float, Float, Color]
_rl.ImageTextEx.restype = Image
_rl.ImageDrawTextEx.argtypes = [ImagePtr, Font, CharPtr, Vector2, Float, Color]
_rl.ImageDrawTextEx.restype = None

class _Camera3D(Struct):
    _fields_=[
        ('_position' , Vector3), # Camera position
        ('_target' , Vector3), # Camera target it looks-at
        ('_up' , Vector3), # Camera up vector (rotation over its axis)
        ('_fovy' , Float), # Camera field-of-view apperture in Y (degrees) in perspective, used as near plane width in orthographic
        ('_projection' , Int) # Camera projection: CAMERA_PERSPECTIVE or CAMERA_ORTHOGRAPHIC
    ]
    
    @property
    def position(self)-> Vector3:
        return self._position
    
    @position.setter
    def position(self, pos: Union[Sequence[float], Vector3])-> None:
        self._position = _vec3(pos)

    @property
    def target(self)-> Vector3:
        return self._target
    
    @target.setter
    def target(self, t: Union[Sequence[float], Vector3])-> None:
        self._target = _vec3(t)

    @property
    def up(self)-> Vector3:
        return self._up
    
    @up.setter
    def up(self, u: Union[Sequence[float], Vector3])-> None:
        self._up = _vec3(u)

    @property
    def fovy(self)-> float:
        return self._fovy
    
    @fovy.setter
    def fovy(self, f: float)-> None:
        self._fovy = _to_float(f)

    @property
    def projection(self)-> int:
        return self._projection
    
    @projection.setter
    def projection(self, p: int)-> None:
        self._projection = _to_int(p)
    

class Camera3D(_Camera3D):
    """
    Camera type, defines a camera position/orientation in 3d space
    """
    def __init__(
        self, 
        position: Union[Sequence[float], Vector3] = (0.0 , 0.0 , 0.0),
        target: Union[Sequence[float], Vector3] = (0.0 , 0.0 , 0.0),
        up: Union[Sequence[float], Vector3] = (0.0 , 0.0 , 0.0),
        fovy: float = 0.0,
        projection: int = CameraProjection.PERSPECTIVE,
        camera: Optional['Camera3D']= None)-> None:
        if camera and isinstance(camera, Camera3D):
            self.__set(
                camera.position,
                camera.target,
                camera.up,
                camera.fovy,
                camera.projection,
            )
        else:
            self.__set(
                _vec3(position),
                _vec3(target),
                _vec3(up),
                _to_float(fovy),
                _to_int(projection),
            )

    @python_wrapper('BeginMode3D', [_Camera3D])
    def begin_mode(self)-> 'Camera3D':
        """Initializes 3D mode with custom camera (3D)"""
        _rl.BeginMode3D(self)
        return self
    
    @python_wrapper('EndMode3D')
    def end_mode(self)-> 'Camera3D':
        """Ends 3D mode and returns to default 2D orthographic mode"""
        _rl.EndMode3D()
        return self
    
    @python_wrapper('GetCameraMatrix', [_Camera3D], Matrix)
    def get_matrix(self)-> Matrix:
        """Returns camera transform matrix (view matrix)"""
        return _rl.GetCameraMatrix(self)

    @python_wrapper('SetCameraMode', [_Camera3D, Int])
    def set_mode(self, mode: Union[CameraMode, int])-> 'Camera3D':
        """Set camera mode (multiple camera modes available)"""
        _rl.SetCameraMode(self, _to_int(mode))
        return self
    
    @python_wrapper('SetCameraAltControl', [Int])
    def set_alt_controll(self, alt_key: int)-> 'Camera3D':
        """Set camera alt key to combine with mouse movement (free camera)"""
        _rl.SetCameraAltControl(_to_int(alt_key))
        return self
    
    @python_wrapper('SetCameraPanControl', [Int])
    def set_camera_pan_control(key_pan: Union[Keyboard, int])-> None:
        """Set camera pan key to combine with mouse movement (free camera)"""
        _rl.SetCameraPanControl(_to_int(key_pan))

    @python_wrapper('SetCameraSmoothZoomControl', [Int])
    def set_smooth_zoom_controll(self, keySmoothZoom: Union[Keyboard, int])-> 'Camera3D':
        """Set camera smooth zoom key to combine with mouse (free camera)"""
        _rl.SetCameraSmoothZoomControl(_to_int(keySmoothZoom))
        return self

    @python_wrapper('SetCameraMoveControls', [Int, Int, Int, Int, Int, Int])
    def set_move_controll(
        self, 
        keyFront: int, 
        keyBack: int,
        keyRight: int, 
        keyLeft: int, 
        keyUp: int,
        keyDown: int)-> 'Camera3D':
        """Set camera move controls (1st person and 3rd person cameras)"""
        _rl.SetCameraMoveControls(
            _to_int(keyFront),
            _to_int(keyBack),
            _to_int(keyRight),
            _to_int(keyLeft),
            _to_int(keyUp),
            _to_int(keyDown),
        )
        return self
    
    @python_wrapper('UpdateCamera', [POINTER(_Camera3D)])
    def update(self)-> 'Camera3D':
        """Update camera position for selected mode"""
        _rl.UpdateCamera(byref(self))
        return self

    def get_mouse_ray(self, mousePosition: Union[Sequence[float], Vector2])-> 'Ray':
        """Returns a ray trace from mouse position"""
        return _rl.GetMouseRay(_vec2(mousePosition), self)

    @python_wrapper('GetWorldToScreen', [Vector3, _Camera3D])
    def get_world_to_screen(self, position: Union[Sequence[float], Vector3])-> Vector2:
        """Returns the screen space position for a 3d world space position"""
        return _rl.GetWorldToScreen(position, self)
    
    @python_wrapper('DrawBillboard', [_Camera3D, Texture2D, Vector3, Float, Color])
    def draw_billboard(
        self, 
        texture: Texture2D, 
        center: Union[Sequence[float], Vector3],
        size: float,
        color: Union[Sequence[int], Color] = (255 , 255 , 255 , 255))-> 'Camera3D':
        """Draw a billboard texture"""
        _rl.DrawBillboard(
            self,
            texture,
            _vec3(center),
            _to_float(size),
            _to_color(color),
        )
        return self

    @python_wrapper('DrawBillboardRec', [_Camera3D, Texture2D, Rectangle, Vector3, Float, Color])
    def draw_billboard_rec(
        self, 
        texture: Texture2D,
        source: Union[Sequence[float], Rectangle], 
        center: Union[Sequence[float], Vector3],
        size: float,
        color: Union[Sequence[int], Color] = (255 , 255 , 255 , 255))-> 'Camera3D':
        """Draw a billboard texture defined by source"""
        _rl.DrawBillboardRec(
            self,
            texture,
            _rect(source),
            _vec3(center),
            _to_float(size),
            _to_color(color),
        )
        return self
    
    def __set(
        self, 
        position: Union[Sequence[float], Vector3],
        target: Union[Sequence[float], Vector3] ,
        up: Union[Sequence[float], Vector3],
        fovy: float,
        projection: int)-> None:
        super(Camera3D, self).__init__(
            position,
            target,
            up,
            fovy,
            projection
        )

    def __enter__(self)-> 'Camera':
        self.begin_mode()
        return self

    def __exit__(self, *args)-> None:
        self.end_mode()

Camera = Camera3D #Camera type fallback, defaults to Camera3D

class _Camera2D(Struct):
    _fields_ =[
        ('_offset' , Vector2),   # Camera offset (displacement from target)
        ('_target' , Vector2),   # Camera target (rotation and zoom origin)
        ('_rotation' , Float),   # Camera rotation in degrees
        ('_zoom' , Float),       # Camera zoom (scaling), should be 1.0f by default
    ]

    @property
    def offset(self)-> Vector2:
        return self._offset
    
    @offset.setter
    def offset(self, o: Union[Sequence[float], Vector2])-> None:
        self._offset = _vec2(o)
    
    @property
    def target(self)-> Vector2:
        return self._target

    @target.setter
    def target(self, t: Union[Sequence[float], Vector2])-> None:
        self._target = _vec2(t)

    @property
    def rotation(self)-> float:
        return self._rotation
    
    @rotation.setter
    def rotation(self, r: float)-> None:
        self._rotation = _to_float(r)
    
    @property
    def zoom(self)-> float:
        return self._zoom

    @zoom.setter
    def zoom(self, z: float)-> None:
        self._zoom = _to_float(z)

class Camera2D(_Camera2D):
    """Camera2D type, defines a 2d camera"""
    def __init__(
        self, 
        offset: Union[Sequence[float], Vector2],
        target: Union[Sequence[float], Vector2],
        rotation: float,
        zoom: float):
        super(Camera2D, self).__init__(
            _vec2(offset),
            _vec2(target),
            _to_float(rotation),
            _to_float(zoom),
        )
    @python_wrapper('BeginMode2D', [_Camera2D])
    def begin_mode(self)-> 'Camera2D':
        """Initialize 2D mode with custom camera (2D)"""
        _rl.BeginMode2D(self)
        return self
    @python_wrapper('EndMode2D')
    def end_mode(self)-> 'Camera2D':
        """Ends 2D mode with custom camera"""
        _rl.EndMode2D()
        return self

    @python_wrapper('GetCameraMatrix2D', [_Camera2D], Matrix)
    def get_matrix(self)-> Matrix:
        """Returns camera 2d transform matrix"""
        return _rl.GetCameraMatrix2D(self)
    
    @python_wrapper('GetWorldToScreen2D', [Vector2,_Camera2D], Vector2)
    def get_world_to_screen(self, position: Union[Sequence[float], Vector2])-> Vector2:
        """Returns the screen space position for a 2d camera world space position"""
        return _rl.GetWorldToScreen2D(_vec2(position), self)
    
    @python_wrapper('GetScreenToWorld2D', [Vector2,_Camera2D], Vector2)
    def get_screen_to_world(self, position: Union[Sequence[float], Vector2])-> Vector2:
        """Returns the world space position for a 2d camera screen space position"""
        return _rl.GetScreenToWorld2D(_vec2(position), self)
    
class Mesh(Struct):
    """
    Vertex data definning a mesh
    NOTE: Data stored in CPU memory (and GPU)
    """
    _fields_ = [
        ('vertexCount', Int),       #Number of vertices stored in arrays
        ('triangleCount', Int),     #Number of triangles stored (indexed or not)

        #Default vertex data
        ('vertices', FloatPtr),     #Vertex position (XYZ - 3 components per vertex) (shader-location = 0)
        ('texcoords', FloatPtr),    #Vertex texture coordinates (UV - 2 components per vertex) (shader-location = 1)
        ('texcoords2', FloatPtr),   #Vertex second texture coordinates (useful for lightmaps) (shader-location = 5)
        ('normals', FloatPtr),      #Vertex normals (XYZ - 3 components per vertex) (shader-location = 2)
        ('tangents', FloatPtr),     #Vertex tangents (XYZW - 4 components per vertex) (shader-location = 4)
        ('color', UCharPtr),        #Vertex colors (RGBA - 4 components per vertex) (shader-location = 3)
        ('indices', UShortPtr),     #Vertex indices (in case vertex data comes indexed)

        #Animation vertex data
        ('animVertices', FloatPtr), #Animated vertex positions (after bones transformations)
        ('animNormals', FloatPtr),  #Animated normals (after bones transformations)
        ('boneIds', IntPtr),        #Vertex bone ids, up to 4 bones influence by vertex (skinning)
        ('boneWeights', FloatPtr),  #Vertex bone weight, up to 4 bones influence by vertex (skinning)

        #OpenGL identifiers
        ('vaoId', UInt),            #OpenGL Vertex Array Object id
        ('vboId', UIntPtr),         #OpenGL Vertex Buffer Objects id (default vertex data)
    ]
        
    def __init__(self, mesh= None):
        if isinstance(mesh, Mesh):
            self.__set(mesh=mesh)
        elif mesh is None:
            pass
        else:
            raise ValueError('you must provide a mesh object or Try other constructor')

    def __set(self, mesh):
        super(Mesh, self).__init__(
            mesh.vertexCount,
            mesh.triangleCount,
            mesh.vertices,
            mesh.texcoords,
            mesh.texcoords2,
            mesh.normals,
            mesh.tangents,
            mesh.color,
            mesh.indices,
            mesh.animVertices,
            mesh.animNormals,
            mesh.boneIds,
            mesh.boneWeights,
            mesh.vaoId,
            mesh.vboId,
        )
    
    # Mesh generation functions
    @classmethod
    def poly(cls, sides: int, radius: float)-> 'Mesh':
        """Generate polygonal mesh"""
        return cls(_rl.GenMeshPoly(
                _to_int(sides),
                _to_float(radius),
            ))
        

    @classmethod
    def plane(cls, width: float, length: float, resX: int, resZ: int)-> 'Mesh':
        """Generate plane mesh (with subdivisions)"""
        return cls(_rl.GenMeshPlane(
            _to_float(width),
            _to_float(length),
            _to_int(resX),
            _to_int(resZ),
        ))
    
    @classmethod
    def cube(cls, width: float, height: float, length: float)-> 'Mesh':
        """Generate cuboid mesh"""
        return cls(_rl.GenMeshCube(
            _to_float(width),
            _to_float(height),
            _to_float(length),
        ))
    
    @classmethod
    def sphere(cls, radius: float, rings: int, slices: int)-> 'Mesh':
        """Generate sphere mesh (standard sphere)"""
        return cls(_rl.GenMeshSphere(
            _to_float(radius),
            _to_int(rings),
            _to_int(slices),
        ))
    
    @classmethod
    def hemi_sphere(cls, radius: float, rings: int, slices: int)-> 'Mesh':
        """Generate half-sphere mesh (no bottom cap)"""
        return cls(_rl.GenMeshHemiSphere(
            _to_float(radius),
            _to_int(rings),
            _to_int(slices)
        ))
    
    @classmethod
    def cylinder(cls, radius: float, height: float, slices: int)-> 'Mesh':
        """Generate cylinder mesh"""
        return cls(_rl.GenMeshCylinder(
            _to_float(radius),
            _to_float(height),
            _to_int(slices)
        ))

    @classmethod
    def torus(cls, radius: float, size: float, radSeg: int, sides: int)-> 'Mesh':
        """Generate torus mesh"""
        return cls(_rl.GenMeshTorus(
            _to_float(radius),
            _to_float(size),
            _to_int(radSeg),
            _to_int(sides),
        ))
    
    @classmethod
    def knot(cls, radius: float, size: float, radSeg: int, sides: int)-> 'Mesh':
        """Generate trefoil knot mesh"""
        return cls(_rl.GenMeshKnot(
            _to_float(radius),
            _to_float(sides),
            _to_int(radSeg),
            _to_int(sides),
        ))

    @classmethod
    def height_map(cls, heightmap: Image, size: Union[Sequence[float], Vector3])-> 'Mesh':
        """Generate heightmap mesh from image data"""
        return cls(_rl.GenMeshHeightmap(
            heightmap,
            _vec3(size)
        ))
    
    @classmethod
    def cubic_map(cls, cubicmap: Image, cubeSize: Union[Sequence[float], Vector3])-> 'Mesh':
        """Generate cubes-based map mesh from image data"""
        return cls(_rl.GenMeshCubicmap(
            cubicmap,
            _vec3(cubeSize),
        ))

    # Mesh manipulation functions
    
    def bounding_box(self)-> 'BoundingBox':
        """Compute mesh bounding box limits"""
        return _rl.MeshBoundingBox(self)
    
    def tangents(self)-> 'Mesh':
        """Compute mesh tangents"""
        _rl.MeshTangents(byref(self))
        return self
    
    def binormals(self)-> 'Mesh':
        """Compute mesh binormals"""
        _rl.MeshBinormals(byref(self))
        return self

    def unload(self)-> 'Mesh':
        """Unload mesh data from CPU and GPU"""
        try:
            _rl.UnloadMesh(self)
        except OSError:
            print('cannot unload mesh data')
        return self
    
    def export(self, filename: str)-> bool:
        """Export mesh data to file, returns true on success"""
        return _rl.ExportMesh(
            self,
            _to_byte_str(filename)
        )

    def upload(self, dynamic: bool = False)-> 'Mesh':
        """Upload mesh vertex data in GPU and provide VAO/VBO ids"""
        _rl.UploadMesh(byref(self), dynamic)
        return self

MeshPtr = POINTER(Mesh)
wrap_function('UploadMesh', [MeshPtr, Bool])
wrap_function('GenMeshPoly', [Int, Float], Mesh)
wrap_function('GenMeshPlane', [Float, Float, Int, Int], Mesh)
wrap_function('GenMeshCube', [Float, Float, Float], Mesh)
wrap_function('GenMeshSphere', [Float, Int, Int], Mesh)
wrap_function('GenMeshHemiSphere', [Float, Int, Int], Mesh)
wrap_function('GenMeshCylinder', [Float, Float, Int], Mesh)
wrap_function('GenMeshTorus', [Float, Float, Int, Int], Mesh)
wrap_function('GenMeshKnot', [Float, Float, Int, Int], Mesh)
wrap_function('GenMeshHeightmap', [Image, Vector3], Mesh)
wrap_function('GenMeshCubicmap', [Image, Vector3], Mesh)
wrap_function('MeshTangents', [MeshPtr])
wrap_function('MeshBinormals', [MeshPtr])



class Shader(Struct):
    """
        Shader type (generic)
    """
    _fields_ = [
        ('id', UInt),       #Shader program id
        ('locs', IntPtr),   #Shader locations array (MAX_SHADER_LOCATIONS)
    ]

    def get_location(self, uniformName: str)-> int:
        """Get shader uniform location"""
        return _rl.GetShaderLocation(self, _to_byte_str(uniformName))
    
    def set_value(self, locIndex: int, value: List[Any], uniformType: Union[ShaderUniformDataType, int], count: int = 1)-> 'Shader':
        """Set shader uniform value vector"""
        v_len = len(value)
        v_array = v_len * VoidPtr
        _rl.SetShaderValueV(
            self,
            _to_int(locIndex),
            v_array(*value),
            _to_int(uniformType),
            _to_int(count),
        )
        return self

    def begin_mode(self)-> 'Shader':
        """Begin custom shader drawing"""
        _rl.BeginShaderMode(self)
        return self

    def end_mode(self)-> 'Shader':
        """End custom shader drawing (use default shader)"""
        _rl.EndShaderMode()
        return self
    @staticmethod
    def load_shader(vsFilename: Union[str, Path], fsFilename: Union[str, Path])-> 'Shader':
        """Load shader from files and bind default locations"""
        return _rl.LoadShader(
            _to_byte_str(vsFilename),
            _to_byte_str(fsFilename),
        )
    
    @staticmethod
    def load_from_memory(vsCode: str, fsCode: str)-> 'Shader':
        """Load shader from code strings and bind default locations"""
        return _rl.LoadShaderFromMemory(
            _to_byte_str(vsCode),
            _to_byte_str(fsCode),
        )

    def unload(self)-> None:
        """Unload shader from GPU memory (VRAM)"""
        _rl.UnloadShader(self)

    def __enter__(self)-> 'Shader':
        self.begin_mode()
        return self
    def __exit__(self, *args)-> None:
        self.end_mode()
wrap_function('EndShaderMode')
wrap_function('BeginShaderMode', [Shader])
wrap_function('SetShaderValueV', [Shader, Int, VoidPtr, Int, Int])
wrap_function('GetShaderLocation', [Shader, CharPtr], Int)
wrap_function('LoadShader', [CharPtr, CharPtr], Shader)
wrap_function('LoadShaderFromMemory', [CharPtr, CharPtr], Shader)
wrap_function('UnloadShader', [Shader])
class _MaterialMap(Struct):
    """
        Material texture map
    """
    _fields_ = [
        ('texture', Texture2D),     #Material map texture
        ('color', Color),           #Material map color
        ('value', Float)            #Material map value
    ]

class MaterialMap(_MaterialMap):
    pass

MaterialMapPtr = POINTER(MaterialMap)

array_float_4 = Float * 4

class Material(Struct):
    """
        Material type (generic)
    """
    _fields_ = [
        ('shader', Shader),         #Material shader
        ('maps', MaterialMapPtr),   #Material maps array (MAX_MATERIAL_MAPS)
        ('params', array_float_4)   #Material generic parameters (if required)
    ]

    def __init__(self, material: 'Material'):
        if isinstance(material, Material):
            self.__set(material)
        else:
            raise ValueError('You must provide a material object.')
    
    @classmethod
    def load_default(cls):
        """Load default material (Supports: DIFFUSE, SPECULAR, NORMAL maps)"""
        return cls(_rl.LoadMaterialDefault())
    
    @staticmethod
    def load_materials(filename: Union[str, Path])-> List['Material']:
        """Load materials from model file"""
        cnt = Int(0)
        c_marray = _rl.LoadMaterials(_to_byte_str(str(filename)), byref(cnt))
        materials = []
        for i in range(cnt.value):
            materials.append(c_marray[i])
        return materials

    def unload(self)-> None:
        """Unload material from GPU memory (VRAM)"""
        _rl.UnloadMaterial(self)

    def set_texture(self, mapType: Union[MaterialMapIndex, int], texture: Texture2D)-> 'Material':
        """Set texture for a material map type (MATERIAL_MAP_DIFFUSE, MATERIAL_MAP_SPECULAR...)"""
        _rl.SetMaterialTexture(
            byref(self),
            _to_int(mapType),
            texture,
        )
        return self

    def __set(self, material: 'Material'):
        super(Material, self).__init__(
            material.shader,
            material.maps,
            material.params,
        )

MaterialPtr = POINTER(Material)

wrap_function('SetMaterialTexture', [MaterialPtr, Int, Texture])
wrap_function('LoadMaterialDefault', restype=Material)
wrap_function('UnloadMaterial', [Material])
wrap_function('LoadMaterials', [CharPtr, IntPtr], MaterialPtr)


class Transform(Struct):
    """
        Transformation properties
    """
    _fields_ = [
        ('_translation', Vector3),   #Translation
        ('_rotation', Quaternion),   #Rotation
        ('_scale', Vector3),         #Scale
    ]

    @property
    def translation(self)-> Vector3:
        return self._translation
    
    @translation.setter
    def translation(self, value: Union[Sequence[Number], Vector3])-> None:
        self._translation = _vec3(value)
    
    @property
    def rotation(self)-> Vector4:
        return self._rotation
    
    @rotation.setter
    def rotation(self, value: Union[Sequence[Number], Vector4])-> None:
        self._rotation = _vec4(value)

    @property
    def scale(self)-> Vector3:
        return self._scale
    
    @scale.setter
    def scale(self, value: Union[Sequence[Number], Vector3])-> None:
        self._scale = _vec3(value)

TransformPtr = POINTER(Transform)
TransformPtrPtr = POINTER(TransformPtr)

array_char_32 = Char * 32

class _BoneInfo(Struct):
    """
    Bone information
    """
    _fields_ = [
        ('name', array_char_32),    #Bone name
        ('parent', Int)             #Bone parent
    ]

class BoneInfo(_BoneInfo):
    pass

BoneInfoPtr = POINTER(BoneInfo)

class Model(Struct):
    """
        Model type
    """
    _fields_ = [
        ('_transform', Matrix),      #Local transform matrix
        ('_meshCount', Int),         #Number of meshes
        ('_materialCount', Int),     #Number of materials
        ('_meshes', MeshPtr),        #Meshes array
        ('_materials', MaterialPtr), #Materials array
        ('meshMaterial', IntPtr),   #Mesh material number

        #Animation data
        ('_boneCount', Int),         #Number of bones
        ('_bones', BoneInfoPtr),     #Bones information (skeleton)
        ('_bindPose', TransformPtr)  #Bones base transformation (pose)
    ]

    def __init__(self, model: 'Model')-> None:
        if isinstance(model, Model):
            self.__set(model)
        else:
            raise ValueError('Try using other constructor')

    @property
    def meshCount(self)-> int:
        return self._meshCount

    @property
    def transform(self)-> Matrix:
        return self._transform

    @property
    def materialCount(self)-> int:
        return self._materialCount

    @property
    def meshes(self)-> MeshPtr:
        return self._meshes

    @property
    def materials(self)-> MaterialPtr:
        return self._materials
    
    @property
    def boneCount(self)-> int:
        return self._boneCount

    @property
    def bones(self)-> BoneInfoPtr:
        return self._bones

    @property
    def bindPose(self)-> TransformPtr:
        return self._bindPose

    def set_material_texture(
        self, 
        material: MaterialPtr, 
        mapType: Union[MaterialMapIndex, int], 
        texture: Texture
        )-> 'Model':
        """Set texture for a material map type (MATERIAL_MAP_DIFFUSE, MATERIAL_MAP_SPECULAR...)"""
        _rl.SetMaterialTexture(byref(material), _to_int(mapType), texture)
        return self

    @staticmethod
    def load_from_file(filename: Union[str, Path])-> 'Model':
        """Load model from files (meshes and materials)"""
        model = _rl.LoadModel(_to_byte_str(str(filename)))
        return model

    @staticmethod
    def load_from_mesh(mesh: Mesh)-> Mesh:
        """Load model from generated mesh (default material)"""
        if not isinstance(mesh, Mesh):
            raise ValueError('You must provide a mesh object')
        return _rl.LoadModelFromMesh(mesh)

    def unload(self)-> None:
        """Unload model (including meshes) from memory (RAM and/or VRAM)"""
        try:
            _rl.UnloadModel(self)
            self._meshes = None
            self._materials = None
        except OSError:
            self.unload_keep_meshes()

    def unload_keep_meshes(self)-> None:
        """Unload model (but not meshes) from memory (RAM and/or VRAM)"""
        _rl.UnloadModelKeepMeshes(self)

    def set_mesh_material(self, meshId: int, materialId: int)-> 'Model':
        """Set material for a mesh"""
        _rl.SetModelMeshMaterial(byref(self), _to_int(meshId), _to_int(materialId))
        return self

    def get_collision(self, ray: 'Ray')-> 'RayHitInfo':
        """Get collision info between ray and model"""
        if not isinstance(ray, Ray):
            raise ValueError('You must provide a ray object.')

        return _rl.GetCollisionRayModel(ray, self)

    def update_animation(self, anim: 'ModelAnimation', frame: int)-> 'Model':
        """Update model animation pose"""
        _rl.UpdateModelAnimation(self, anim, _to_int(frame))
        return self
    
    def is_valid_model_animation(self, anim: 'ModelAnimation')-> bool:
        """Check model animation skeleton match"""
        return _rl.IsModelAnimationValid(self, anim)

    def draw(
        self, 
        position: Union[Sequence[float], Vector3],
        scale: float = 1.0,
        color: Union[Sequence[int], Color] = (0, 0, 0, 255),
        wires: bool=False,
        )-> 'Model':
        """
        Draw a model (with texture if set)
        or Draw a model wires (with texture if set) if wires is set to True
        """
        if wires:
            _rl.DrawModelWires(
                self,
                _vec3(position),
                _to_float(scale),
                _to_color(color),
            )
        else:
            _rl.DrawModel(
                self,
                _vec3(position),
                _to_float(scale),
                _to_color(color),
            )
        
        return self
    
    def draw_ex(
        self, 
        position: Union[Sequence[float], Vector3],
        rotation_axis: Union[Sequence[float], Vector3],
        rotation_angle: float, 
        scale: Union[Sequence[float], Vector3], 
        color: Union[Sequence[int], Color],
        wires=False) -> 'Model':

        """
        Draw a model with extended parameters
        or Draw a model wires (with texture if set) with extended parameters 
        if wires is set to True
        """
        if wires:
            _rl.DrawModelWiresEx(
                self,
                _vec3(position),
                _vec3(rotation_axis),
                _to_float(rotation_angle),
                _vec3(scale),
                _to_color(color)
            )
        else:
            _rl.DrawModelEx(
                self,
                _vec3(position),
                _vec3(rotation_axis),
                _to_float(rotation_angle),
                _vec3(scale),
                _to_color(color)
            )
        
        return self
    
    def __set(self, model: 'Model')-> None:
        super(Model, self).__init__(
            model.transform,
            model.meshCount,
            model.materialCount,
            model.meshes,
            model.materials,
            model.meshMaterial,
            model.boneCount,
            model.bones,
            model.bindPose,
        )

ModelPtr = POINTER(Model)
wrap_function('DrawModelEx', [Model, Vector3, Vector3, Float, Vector3, Color])
wrap_function('DrawModelWiresEx', [Model, Vector3, Vector3, Float, Vector3, Color])

wrap_function('DrawModel', [Model, Vector3, Float, Color])
wrap_function('DrawModelWires', [Model, Vector3, Float, Color])
wrap_function('LoadModel', [CharPtr], Model)
wrap_function('LoadModelFromMesh', [Mesh], Model)
wrap_function('UnloadModel', [Model])
wrap_function('UnloadModelKeepMeshes', [Model])
wrap_function('SetModelMeshMaterial', [ModelPtr, Int, Int])

class ModelAnimation(Struct):
    """
        Model animation
    """
    _fields_ = [
        ('_boneCount', Int),                 #Number of bones
        ('_frameCount', Int),                #Number of animation frames
        ('_bones', BoneInfoPtr),             #Bones information (skeleton)
        ('_framePoses', TransformPtrPtr)     #Poses array by frame
    ]

    @property
    def boneCount(self)-> int:
        return self._boneCount
    @property
    def frameCount(self)-> int:
        return self._frameCount
    @property
    def bones(self)-> BoneInfoPtr:
        return self._bones
    @property
    def framePoses(self)-> TransformPtrPtr:
        return self._framePoses

    def is_valid(self, model: Model)-> bool:
        """Check model animation skeleton match"""
        return _rl.IsModelAnimationValid(model, self)

    @staticmethod
    def load(filename: Union[str, Path])-> Tuple[List['ModelAnimation'], int]:
        """Load model animations from file"""
        count = Int(0)
        data = _rl.LoadModelAnimations(
            _to_byte_str(str(filename)),
            byref(count)
        )
        return (data, count.value)
    
    @staticmethod
    def unload_model_anmations(animations: 'ModelAnimationPtr', count: int)-> None:
        """Unload animation array data"""
        for i in range(count):
            animations[i].unload()

    
    def unload(self)-> None:
        """Unload animation data"""
        _rl.UnloadModelAnimation(self)


ModelAnimationPtr = POINTER(ModelAnimation)
wrap_function('UnloadModelAnimations', [ModelAnimationPtr, UInt])
wrap_function('UnloadModelAnimation', [ModelAnimation])
wrap_function('LoadModelAnimations', [CharPtr, IntPtr], ModelAnimationPtr)
wrap_function('UpdateModelAnimation', [Model, ModelAnimation, Int])
wrap_function('IsModelAnimationValid', [Model, ModelAnimation], Bool)

class _Ray(Struct):
    
    _fields_ = [
        ('_position', Vector3),      #Ray position (origin)
        ('_direction', Vector3),     #Ray direction
    ]

    @property
    def position(self)-> Vector3:
        return self._position
    
    @position.setter
    def position(self, pos: Union[Sequence[float], Vector3])-> None:
        self._position = _vec3(pos)
    
    @property
    def direction(self)-> Vector3:
        return self._direction
    
    @direction.setter
    def direction(self, pos: Union[Sequence[float], Vector3])-> None:
        self._direction = _vec3(pos)

class Ray(_Ray):
    """
        Ray type (useful for raycast)
    """

    def __init__(
        self, 
        position: Union[Sequence[float], Vector3] = (0 , 0 , 0), 
        direction: Union[Sequence[float], Vector3]= (0 , 0 , 0),
        ray: Optional['Ray'] = None
        ):
        if ray:
            self.__set(
                ray.position,
                ray.direction
            )
        else:
            self.__set(
                _vec3(position),
                _vec3(direction),
            )
    
    @python_wrapper('DrawRay', [_Ray, Color])
    def draw(self, color: Union[Sequence[int], Color] = (0 , 0 , 0 , 255))-> 'Ray':
        """Draw a ray line"""
        _rl.DrawRay(self, _to_color(color))
        return self

    @python_wrapper('CheckCollisionRaySphereEx', [_Ray, Vector3, Float, Vector3Ptr], Bool)
    def check_collision_sphere(
        self, 
        center: Union[Sequence[float], Vector3], 
        radius: float)-> Tuple[bool , Vector3]:
        """Detect collision between ray and sphere, returns collision point"""
        collision_point = Vector3(0 , 0 , 0)
        return (_rl.CheckCollisionRaySphereEx(
            self, 
            _vec3(center), 
            _to_float(radius), 
            byref(collision_point)
            ),collision_point)

    def check_collision_box(self, box: 'BoundingBox')-> bool:
        """Detect collision between ray and box"""
        return _rl.CheckCollisionRayBox(self, box)

    def get_collision_mesh(self, mesh: Mesh, transform: Matrix)-> 'RayHitInfo':
        """Get collision info between ray and mesh"""
        return _rl.GetCollisionRayMesh(self, mesh, transform)
    
    def get_collision_model(self, model: Model)-> 'RayHitInfo':
        """Get collision info between ray and model"""
        return _rl.GetCollisionRayModel(self, model)

    def get_collision_triangle(self, p: Sequence[Vector3])-> 'RayHitInfo':
        """Get collision info between ray and triangle"""
        p1, p2, p3 = _flatten(Vector3,p)
        return _rl.GetCollisionRayTriangle(self, p1, p2, p3)

    def get_collision_ground(self, ground_height: float)-> 'RayHitInfo':
        """Get collision info between ray and ground plane (Y-normal plane)"""
        return _rl.GetCollisionRayGround(self, _to_float(ground_height))
   
    def __set(
        self, 
        position: Union[Sequence[float], Vector3], 
        direction: Union[Sequence[float], Vector3],
    )-> None:
        super(Ray, self).__init__(
            position,
            direction
        )

wrap_function('GetMouseRay', restype=Ray ,argtypes=[Vector2,_Camera3D])

class RayHitInfo(Struct):
    """
        Raycast hit information
    """
    _fields_ = [
        ('_hit', Bool),          #Did the ray hit something?
        ('_distance', Float),    #Distance to nearest hit
        ('_position', Vector3),  #Position of nearest hit
        ('_normal', Vector3)     #Surface normal of hit
    ]

    @property
    def hit(self)-> bool:
        return self._hit

    @hit.setter
    def hit(self, h: bool)-> None:
        self._hit = bool(h)
    
    @property
    def distance(self)-> bool:
        return self._distance
        
    @distance.setter
    def distance(self, d: float)-> None:
        self._distance = _to_float(d)

    @property
    def position(self)-> Vector3:
        return self._position
    
    @position.setter
    def position(self, pos: Union[Sequence[float], Vector3])-> None:
        self._position = _vec3(pos)
    
    @property
    def normal(self)-> Vector3:
        return self._normal

    @normal.setter
    def normal(self, n: Union[Sequence[float], Vector3])-> None:
        self._normal = _vec3(n)

    def __str__(self) -> str:
        return f"""hit: {self.hit}\ndistance: {self.distance}\nposition: {self.position}\nnormal: {self.normal}\n"""


wrap_function('GetCollisionRayMesh', restype=RayHitInfo, argtypes=[Ray, Mesh, Matrix])
wrap_function('GetCollisionRayModel', restype=RayHitInfo, argtypes=[Ray, Model])
wrap_function('GetCollisionRayTriangle', restype=RayHitInfo, argtypes=[Ray, Vector3, Vector3, Vector3])
wrap_function('GetCollisionRayGround', restype=RayHitInfo, argtypes=[Ray, Float])
wrap_function('GetCollisionRayModel', [Ray, Model] , RayHitInfo)

class BoundingBox(Struct):
    """
        Bounding box type
    """
    _fields_ = [
        ('min' , Vector3),  #Minimum vertex box-corner
        ('max', Vector3),   #Maximum vertex box-corner
    ]

    def __init__(
        self, 
        min: Union[Sequence[float], Vector3] = (0 , 0, 0), 
        max: Union[Sequence[float], Vector3] = (0, 0, 0),
        box: Optional['BoundingBox']=None
        )-> None:
        if box and isinstance(box, BoundingBox):
            self.__set(
                box.min,
                box.max,
            )

        else:
            self.__set(
                _vec3(min),
                _vec3(max),
            )
    
    def draw(self, color: Union[Sequence[float], Color]= (255, 255, 255, 255))-> 'BoundingBox':
        """Draw bounding box (wires)"""
        _rl.DrawBoundingBox(self, _to_color(color))
        return self
    
    def check_collision(self, box: 'BoundingBox')-> bool:
        """Detect collision between two bounding boxes"""
        if not isinstance(box, BoundingBox):
            raise ValueError(f"box is {type(box)} and not of type BoundingBox")

        return _rl.CheckCollisionBoxes(self, box)

    def check_collision_sphere(self, center: Union[Sequence[float], Vector3], radius: float)-> bool:
        """Detect collision between box and sphere"""
        return _rl.CheckCollisionBoxSphere(self, _vec3(center), _to_float(radius))

    def check_collision_ray(self, ray: Ray)-> bool:
        """Detect collision between ray and box"""
        return _rl.CheckCollisionRayBox(ray, self)

    def __set(
        self, 
        min: Union[Sequence[float], Vector3], 
        max: Union[Sequence[float], Vector3]
        )-> None:
        super(BoundingBox, self).__init__(
            _vec3(min),
            _vec3(max),
        )

wrap_function('DrawBoundingBox', [BoundingBox, Color])
wrap_function('CheckCollisionBoxes', [BoundingBox, BoundingBox], Bool)
wrap_function('CheckCollisionBoxSphere', [BoundingBox, Vector3, Float], Bool)
wrap_function('MeshBoundingBox', [Mesh], BoundingBox)
wrap_function('CheckCollisionRayBox', argtypes=[Ray, BoundingBox],restype=Bool)


class Wave(Struct):
    """
        Wave type, defines audio wave data
    """

    _fields_ = [
        ('sampleCount', UInt),      #Total number of samples
        ('sampleRate', UInt),       #Frequency (samples per second)
        ('sampleSize', UInt),       #Bit depth (bits per sample): 8, 16, 32 (24 not supported)
        ('channels', UInt),         #Number of channels (1-mono, 2-stereo)
        ('data', VoidPtr)           #Buffer data pointer
    ]

    def __init__(self, wave):
        if isinstance(wave, Wave):
            self.__set(wave)
        else:
            raise ValueError('Invalid argument')

    def copy(self)-> 'Wave':
        """Copy a wave to a new wave"""
        return _rl.WaveCopy(self)

    def export(self, filename: Union[str, Path])-> bool:
        return _rl.ExportWave(self, _to_byte_str(str(filename)))

    def export_as_code(self, filename: Union[str, Path])-> bool:
        """
        Export wave sample data to code (.h) as an array of bytes, returns true on success
        """
        return _rl.ExportWaveAsCode(self, _to_byte_str(str(filename)))

    @staticmethod
    def from_file(filename)-> 'Wave':
        """Load wave data from file"""
        return _rl.LoadWave(_to_byte_str(str(filename)))

    def unload(self)-> None:
        _rl.UnloadWave(self)

    def format(self, sampleRate: int, sampleSize: int, channels: int)-> 'Wave':
        """Convert wave data to desired format"""
        _rl.WaveFormat(
            byref(self),
            _to_int(sampleRate),
            _to_int(sampleSize),
            _to_int(channels) 
            )
        return self

    def __set(self, data):
        super(Wave, self).__init__(
            data.sampleCount,
            data.sampleRate,
            data.sampleSize,
            data.channels,
            data.data,
        )

wrap_function('ExportWaveAsCode', [Wave, CharPtr], Bool)
wrap_function('WaveFormat', [POINTER(Wave), Int, Int, Int])
wrap_function('UnloadWave', [Wave])
wrap_function('ExportWave', [Wave, CharPtr], Bool)
wrap_function('WaveCopy', [Wave], Wave)
wrap_function('LoadWave', [CharPtr], Wave)
    


class AudioStream(Struct):
    """
     Audio stream type
     NOTE: Useful to create custom audio streams not bound to a specific file
    """
    _fields_ = [
        ('buffer', VoidPtr), #Pointer to internal data used by the audio system
        ('sampleRate', UInt), #Frequency (samples per second)
        ('sampleSize', UInt), #Bit depth (bits per sample): 8, 16, 32 (24 not supported)
        ('channels', UInt) #Number of channels (1-mono, 2-stereo)
    ]

    def __init__(self, audio_stream: 'AudioStream')-> None:
        self.__set(audio_stream)

    
    def play(self)-> 'AudioStream':
        """Play audio stream"""
        _rl.PlayAudioStream(self)
        return self

    def pause(self)-> 'AudioStream':
        """Pause audio stream"""
        _rl.PauseAudioStream(self)
        return self

    def resume(self)-> 'AudioStream':
        """Resume audio stream"""
        _rl.ResumeAudioStream(self)
        return self
    
    def stop(self)-> 'AudioStream':
        """Stop audio stream"""
        _rl.StopAudioStream(self)
        return self

    def is_playing(self)-> bool:
        """Check if audio stream is playing"""
        return _rl.IsAudioStreamPlaying(self)

    def is_processed(self)-> bool:
        """Check if any audio stream buffers requires refill"""
        return _rl.IsAudioStreamProcessed(self)

    def set_volume(self, volume: float)-> 'AudioStream':
        """Set volume for audio stream (1.0 is max level)"""
        _rl.SetAudioStreamVolume(self, _to_float(volume))
        return self

    def set_pitch(self, pitch: float)-> 'AudioStream':
        """Set pitch for audio stream (1.0 is base level)"""
        _rl.SetAudioStreamPitch(self, _to_float(pitch))
        return self

    @staticmethod
    @python_wrapper('SetAudioStreamBufferSizeDefault',[Int])
    def set_buffer_size(size: int)-> None:
        """Default size for new audio streams"""
        _rl.SetAudioStreamBufferSizeDefault(_to_int(size))

    @staticmethod
    def init_audio_stream(sampleRate: int, sampleSize: int, channels: int)-> 'AudioStream':
        """Init audio stream (to stream raw audio pcm data)"""
        return _rl.InitAudioStream(sampleRate, sampleSize, channels)

    def close(self)-> None:
        """Close audio stream and free memory"""
        _rl.CloseAudioStream(self)

    def __set(self, data):
        super(AudioStream, self).__init__(
            data.buffer,
            data.sampleRate,
            data.sampleSize,
            data.channels,
        )

wrap_function('IsAudioStreamPlaying',[AudioStream], Bool)
wrap_function('IsAudioStreamProcessed',[AudioStream], Bool)
wrap_function('StopAudioStream',[AudioStream, Float])
wrap_function('SetAudioStreamPitch',[AudioStream, Float])
wrap_function('InitAudioStream', [UInt, UInt, UInt], AudioStream)
wrap_function('CloseAudioStream', [AudioStream])    
wrap_function('PlayAudioStream',[AudioStream])

class Sound(Struct):
    _fields_ = [
        ('stream', AudioStream), #Audio stream
        ('sampleCount', UInt) #Total number of samples
    ]

    def __init__(self, sound: 'Sound'):
        if isinstance(sound, Sound):
            self.__set(sound)
        else:
            raise TypeError('Invalid argument')

    def play(self)-> 'Sound':
        """Play a sound"""
        _rl.PlaySound(self)
        return self

    def stop(self)-> 'Sound':
        """Stop playing a sound"""
        _rl.StopSound(self)
        return self

    def pause(self)-> 'Sound':
        """Pause a sound"""
        _rl.PauseSound(self)
        return self

    def resume(self)-> 'Sound':
        """Resume a paused sound"""
        _rl.ResumeSound(self)
        return self

    def play_multi(self)-> 'Sound':
        """Play a sound (using multichannel buffer pool)"""
        _rl.PlaySoundMulti(self)
        return self

    def stop_multi(self)-> 'Sound':
        """Stop any sound playing (using multichannel buffer pool)"""
        _rl.StopSoundMulti()
        return self

    def is_playing(self)-> bool:
        """Check if a sound is currently playing"""
        return _rl.IsSoundPlaying(self)

    def set_volume(self, volume: float)-> 'Sound':
        """Set volume for a sound (1.0 is max level)"""
        _rl.SetSoundVolume(self, _to_float(volume))
        return self

    def set_pitch(self, pitch: float)-> 'Sound':
        """Set pitch for a sound (1.0 is base level)"""
        _rl.SetSoundPitch(self, _to_float(pitch))
        return self

    @staticmethod
    def from_file(self, filename: Union[str, Path])-> 'Sound':
        """Load sound from file"""
        return _rl.LoadSound(_to_byte_str(str(filename)))

    @staticmethod
    def from_wave(wave: Wave)-> 'Sound':
        if isinstance(wave, Wave):
            return _rl.LoadSoundFromWave(wave)
        else:
            raise TypeError('Invalide argument')

    def unload(self)-> None:
        """Unload sound"""
        _rl.UnloadSound(self)

    def __set(self, data):
        super(Sound, self).__init__(
            data.stream,
            data.sampleCount
        )

wrap_function('PauseAudioStream',[AudioStream])
wrap_function('ResumeAudioStream',[AudioStream])
wrap_function('StopAudioStream',[AudioStream])
wrap_function('SetSoundVolume', [Sound, Float])
wrap_function('StopSound', [Sound])
wrap_function('PlaySoundMulti', [Sound])
wrap_function('IsSoundPlaying', [Sound], Bool)
wrap_function('PauseSound', [Sound])
wrap_function('StopSoundMulti')
wrap_function('SetSoundPitch', [Sound, Float])
wrap_function('UnloadSound', [Sound])
wrap_function('ResumeSound', [Sound])
wrap_function('PlaySound', [Sound])
wrap_function('LoadSoundFromWave', [Wave], Sound)
wrap_function('LoadSound', [CharPtr], Sound)

class Music(Struct):
    """
        Music stream type (audio file streaming from memory)
        NOTE: Anything longer than ~10 seconds should be streamed
    """
    _fields_ = [
        ('stream' , AudioStream),#Audio stream
        ('sampleCount', UInt),#Total number of samples
        ('looping', Bool), #Music looping enable
        ('ctxType', Int), #Type of music context (audio filetype)
        ('ctxData', VoidPtr), #Audio context data, depends on type
    ]
    def __init__(self, music: 'Music')-> None:
        if isinstance(music, Music):
            self.__set(music)
        else:
            raise TypeError('{!r} is not a music object'.format(music))

    def play(self)-> 'Music':
        _rl.PlayMusicStream(self)
        return self

    def update(self)-> 'Music':
        """Update music buffer with new stream data"""
        _rl.UpdateMusicStream(self)
        return self

    def stop(self)-> 'Music':
        """Stop music playing"""
        _rl.StopMusicStream(self)
        return self

    def pause(self)-> 'Music':
        """Pause music playing"""
        _rl.PauseMusicStream(self)
        return self

    def resume(self)-> 'Music':
        """Resume playing paused music"""
        _rl.ResumeMusicStream(self)
        return self

    def set_pitch(self, pitch: float)-> 'Music':
        """Set pitch for music"""
        _rl.SetMusicPitch(self, _to_float(pitch))
        return self

    def is_playing(self)-> bool:
        """Check if music is playing"""
        return _rl.IsMusicPlaying(self)

    def set_volume(self, volume: float)-> 'Music':
        """Set volume for music (1.0 is max level)"""
        _rl.SetMusicVolume(self, _to_float(volume))
        return self

    @property
    def time_played(self)-> float:
        """Get current music time played (in seconds)"""
        return _rl.GetMusicTimePlayed(self)

    def reset(self):
        self.stop()
        self.play()

    @property
    def length(self)-> float:
        """Get music time length (in seconds)"""
        return _rl.GetMusicTimeLength(self)

    def close(self)-> None:
        self.unload()

    def unload(self)-> None:
        """Unload music stream"""
        _rl.UnloadMusicStream(self)

    @staticmethod
    def load_music(filename: Union[str, Path])-> 'Music':
        return _rl.LoadMusicStream(_to_byte_str(str(filename)))

    def __set(self, data):
        super(Music, self).__init__(
            data.stream,
            data.sampleCount,
            data.looping,
            data.ctxType,
            data.ctxData
        )

wrap_function('PlayMusicStream', [Music])
wrap_function('StopMusicStream', [Music])
wrap_function('LoadMusicStream', [CharPtr], Music)
wrap_function('UpdateMusicStream', [Music])
wrap_function('PauseMusicStream', [Music])
wrap_function('ResumeMusicStream', [Music])
wrap_function('SetMusicPitch', [Music, Float])
wrap_function('IsMusicPlaying', [Music], Bool)
wrap_function('SetMusicVolume', [Music, Float])
wrap_function('GetMusicTimePlayed', [Music], Float)
wrap_function('GetMusicTimeLength', [Music], Float)
wrap_function('UpdateMusicStream', [Music])

class AudioDevice:
    """
    Audio device management functions.
    """
    def __init__(self) -> None:
        raise Exception('You cannot instantiate an object of this class')

    @staticmethod
    @python_wrapper('InitAudioDevice')
    def init()-> None:
        """Initialize audio device and context."""
        _rl.InitAudioDevice()

    @staticmethod
    @python_wrapper('CloseAudioDevice')
    def close()-> None:
        """Close the audio device and context."""
        _rl.CloseAudioDevice()

    @staticmethod
    @python_wrapper('IsAudioDeviceReady', None, Bool)
    def is_ready()-> bool:
        """Check if audio device has been initialized successfully."""
        return _rl.IsAudioDeviceReady()

    @staticmethod
    @python_wrapper('SetMasterVolume', [Float])
    def set_volume(volume: float)-> 'AudioDevice':
        """Set master volume (listener)."""
        _rl.SetMasterVolume(_to_float(volume))

class Window(object, metaclass=Singleton):

    def __init__(
        self, 
        size: Union[Sequence[int], Vector2] = (400, 400),
        title: str = 'pyraylib',
        fps = 60,
        late_init: bool = False,
        ) -> None:
        """Initialize window and OpenGL context."""
        super().__init__()
        if not late_init:  
            self.init_window(size, title)
            self.set_fps(fps)

    @python_wrapper('InitWindow', [Int, Int, CharPtr])
    def init_window(self, size: Union[Sequence[int], Vector2], title: str)-> None:
        """Initialize window and OpenGL context."""
        width, height = _flatten((int, float), size, map_to=int)
        _rl.InitWindow(
            _to_int(width),
            _to_int(height),
            _to_byte_str(title)
        )

    @python_wrapper('WindowShouldClose', None, Bool)
    def is_open(self)-> bool:
        """Check if KEY_ESCAPE pressed or Close icon pressed"""
        return not self.should_close()

    def should_close(self)-> bool:
        """Check if KEY_ESCAPE pressed or Close icon pressed"""
        return _rl.WindowShouldClose()
        
    @python_wrapper('CloseWindow')
    def close(self)->None:
        """Close window and unload OpenGL context"""
        _rl.CloseWindow()

    @python_wrapper('IsWindowReady', None, Bool)
    def is_ready(self)-> bool:
        """Check if window has been initialized successfully"""
        return _rl.IsWindowReady()

    @python_wrapper('IsWindowFullscreen', None, Bool)
    def is_fullscreen(self)-> bool:
        """Check if window is currently fullscreen"""
        return _rl.IsWindowFullscreen()

    @python_wrapper('IsWindowHidden', None, Bool)
    def is_hidden(self)-> bool:
        """Check if window is currently hidden"""
        return _rl.IsWindowHidden()

    @python_wrapper('IsWindowMinimized', None, Bool)
    def is_minimized(self)-> bool:
        """Check if window is currently minimized"""
        return _rl.IsWindowMinimized()

    @python_wrapper('IsWindowMaximized', None, Bool)
    def is_maximized(self)-> bool:
        """Check if window is currently minimized"""
        return _rl.IsWindowMaximized()

    @python_wrapper('IsWindowFocused', None, Bool)
    def is_focused(self)-> bool:
        """Check if window is currently focused"""
        return _rl.IsWindowFocused()

    @python_wrapper('IsWindowResized', None, Bool)
    def is_resized(self)-> bool:
        """Check if window has been resized last frame"""
        return _rl.IsWindowResized()

    @python_wrapper('IsWindowState', [Int], Bool)
    def is_state(self ,flag: int)-> bool:
        """Check if one specific window flag is enabled"""
        if isinstance(flag , WindowState):
            return _rl.IsWindowState(flag.value)
        else:
            raise ValueError(f'{flag} must be of type WindowState')

    @python_wrapper('SetWindowState', [Int])
    def set_state(self , flag: int)-> 'Window':
        """Set window configuration state using flags"""
        if isinstance(flag , WindowState):
            _rl.SetWindowState(flag.value)
        else:
            raise ValueError(f'{flag} must be of type WindowState')
        return self

    @python_wrapper('ClearWindowState', [Int])
    def clear_state(self , flag: int)-> 'Window':
        """Clear window configuration state flags"""
        _rl.ClearWindowState(_to_int(flag))
        return self

    @python_wrapper('ClearBackground', [Color])
    def clear_background(self , color:Union[Seq, Color]= (0 , 0 , 0 , 255))-> 'Window':
        """Clear window with given color."""
        _rl.ClearBackground(_to_color(color))
        return self

    @python_wrapper('ToggleFullscreen')
    def toggle_fullscreen(self)-> 'Window':
        """Toggle window state: fullscreen/windowed"""
        _rl.ToggleFullscreen()
        return self

    def set_fullscreen(self , flag: bool = False):
        """Set whether or not the application should be fullscreen."""
        if flag and not self.is_fullscreen():
            self.toggle_fullscreen()
        else:
            if self.is_fullscreen():
                self.toggle_fullscreen()

    @python_wrapper('MaximizeWindow')
    def maximize(self)-> 'Window':
        """Set window state: maximized, if resizable (only PLATFORM_DESKTOP)"""
        _rl.MaximizeWindow()
        return self

    @python_wrapper('MinimizeWindow')
    def minimize(self)-> 'Window':
        """Set window state: minimized, if resizable (only PLATFORM_DESKTOP)"""
        _rl.MinimizeWindow()
        return self

    @python_wrapper('RestoreWindow')
    def restore(self)-> 'Window':
        """Set window state: not minimized/maximized (only PLATFORM_DESKTOP)"""
        _rl.RestoreWindow()
        return self

    @python_wrapper('SetWindowIcon', [Image])
    def set_icon(self , icon: Image)-> 'Window':
        """
        Set icon for window
        Window icon image must be in R8G8B8A8 pixel format
        Check out PixelFormat enum
        """
        if icon.format != PixelFormat.UNCOMPRESSED_R8G8B8A8:
            icon = icon.copy().convert_to(PixelFormat.UNCOMPRESSED_R8G8B8A8)
        _rl.SetWindowIcon(icon)
        return self

    @python_wrapper('SetWindowTitle', [CharPtr])
    def set_title(self, title: str)-> 'Window':
        """Set title for window"""
        _rl.SetWindowTitle(_to_byte_str(title))
        return self

    @python_wrapper('SetWindowPosition', [Int, Int])
    def set_position(self , pos: Union[IntSequence , Vector2])-> 'Window':
        """Set window position on screen"""
        if isinstance(pos , Sequence) and len(pos) == 2:
            x , y = pos
        elif isinstance(pos , Vector2):
            x , y = pos.x , pos.y
        else:
            raise ValueError(f'Position must be of type Sequence or Vector2')

        _rl.SetWindowPosition(_to_int(x) , _to_int(y))
        return self

    @python_wrapper('SetWindowMonitor', [Int])
    def set_monitor(self , monitor: int)-> 'Window':
        """Set monitor for the current window"""
        _rl.SetWindowMonitor(_to_int(monitor))
        return self

    @python_wrapper('SetWindowMinSize', [Int, Int])
    def set_min_size(self , size: IntSequence)-> 'Window':
        """Set window minimum dimensions"""
        if isinstance(size , Sequence):
            width , height = size
        else:
            raise ValueError('Size must be of type List[int] , Tuple[int , int]')
        _rl.SetWindowMinSize(_to_int(width) , _to_int(height))
        return self


    @property
    @python_wrapper('GetScreenWidth', None, Int)
    def width(self)-> int:
        """Get current screen width"""
        return _rl.GetScreenWidth()

    @property
    @python_wrapper('GetScreenHeight', None, Int)
    def height(self)-> int:
        """Get current screen height"""
        return _rl.GetScreenHeight()

    @property
    def size(self)-> Tuple[int, int]:
        """Get the screen's width and height."""
        return (self.width , self.height)

    @size.setter
    @python_wrapper('SetWindowSize', [Int, Int])
    def size(self , size: Union[Seq , Vector2])-> None:
        """Set window dimensions"""
        width, height = _flatten((int, float), size, map_to=int)
        _rl.SetWindowSize(width , height)

    @python_wrapper('GetWindowPosition', None, Vector2)
    def get_position(self)-> 'Vector2':
        """Get window position XY on monitor"""
        return _rl.GetWindowPosition()

    @python_wrapper('GetWindowScaleDPI', None, Vector2)
    def get_scale_dpi(self)-> 'Vector2':
        """Get window scale DPI factor"""
        return _rl.GetWindowScaleDPI()

    @python_wrapper('SetTargetFPS', [Int])
    def set_fps(self , fps: int)-> None:
        """Set target FPS (maximum)"""
        _rl.SetTargetFPS(_to_int(fps))

    @python_wrapper('GetFPS', None, Int)
    def get_fps(self)-> int:
        """Returns current FPS"""
        return _rl.GetFPS()

    @python_wrapper('GetFrameTime', None, Float)
    def get_frame_time(self)-> float:
        """Returns time in seconds for last frame drawn"""
        return _rl.GetFrameTime()

    @python_wrapper('GetTime', None, Float)
    def get_time(self)-> float:
        """Returns elapsed time in seconds since init_window()"""
        return _rl.GetTime()

    @python_wrapper('BeginDrawing')
    def begin_drawing(self)-> None:
        """Setup canvas (framebuffer) to start drawing"""
        return _rl.BeginDrawing()

    @python_wrapper('EndDrawing')
    def end_drawing(self)-> None:
        """End canvas drawing and swap buffers (double buffering)"""
        return _rl.EndDrawing()

    @python_wrapper('ShowCursor')
    def show_cursor(self)-> 'Window':
        """Shows cursor"""
        _rl.ShowCursor()
        return self

    @python_wrapper('HideCursor')
    def hide_cursor(self)-> 'Window':
        """Hide Cursor"""
        _rl.HideCursor()
        return self

    @python_wrapper('EnableCursor')
    def enable_cursor(self)-> None:
        """Enables cursor (unlock cursor)"""
        _rl.EnableCursor()

    @python_wrapper('DisableCursor')
    def disable_cursor(self)-> None:
        """Disables cursor (lock cursor)"""
        _rl.DisableCursor()

    @python_wrapper('IsCursorHidden', None, Bool)
    def is_cursor_hidden(self)-> bool:
        """Check if cursor is not visible"""
        return _rl.IsCursorHidden()

    @python_wrapper('IsCursorOnScreen', None, Bool)
    def is_cursor_on_screen(self)-> bool:
        """Check if cursor is on the current screen"""
        return _rl.IsCursorOnScreen()

    @python_wrapper('DrawFPS', [Int, Int])
    def draw_fps(self, pos: Union[Seq, Vector2])-> None:
        x, y = _flatten((int, float), pos, map_to=int)
        _rl.DrawFPS(x, y)
    
    @python_wrapper('GetCurrentMonitor', restype=Int)
    def get_current_monitor(self)-> int:
        """Get current connected monitor"""
        return _rl.GetCurrentMonitor()

    @python_wrapper('GetMonitorName', [Int], CharPtr)
    def get_monitor_name(self, monitor: int = None)-> str:
        """Get the human-readable, UTF-8 encoded name of the primary monitor"""
        if not monitor:
            monitor = self.get_current_monitor()
        return _to_str(_rl.GetMonitorName(monitor))
    
    @property
    @python_wrapper('GetClipboardText', restype=CharPtr)
    def clipboard_text(self)-> str:
        """Get or set clipboard text content"""
        return _to_str(_rl.GetClipboardText())
    
    @clipboard_text.setter
    @python_wrapper('SetClipboardText', [CharPtr])
    def clipboard_text(self, text: str)-> None:
        """Set clipboard text content"""
        _rl.SetClipboardText(_to_byte_str(text))

    def get_dropped_files(self)-> List[str]:
        """Get dropped files (names)"""
        return get_dropped_files()
    
    def is_file_dropped(self)-> bool:
        """Check if a file has been dropped into window"""
        return is_file_dropped()

# Text drawing functions

_rl.DrawText.argtypes = [CharPtr, Int, Int, Int, Color]
_rl.DrawText.restype = None
def draw_text(
    title: str ,
    posX: int ,
    posY: int ,
    font_size: int ,
    color: Union[Sequence[int], Color]
    ):
    """Draw text (using default font)"""
    _rl.DrawText(
        _to_byte_str(title) ,
        _to_int(posX) ,
        _to_int(posY) ,
        _to_int(font_size) ,
        _to_color(color)
        )

wrap_function('DrawTextEx', [Font, CharPtr, Vector2, Float, Float, Color])
def draw_text_ex(
    font: Font,
    text: str,
    pos: Union[Sequence[Number], Vector2] = (0.,0.),
    size: float = 20,
    spacing: float = 1,
    color: Union[Sequence[Number], Color] = RED,
)-> None:
    _rl.DrawTextEx(
            font,
            _to_byte_str(text),
            _vec2(pos),
            _to_float(size),
            _to_float(spacing),
            _to_color(color)
        )
#Text misc. functions
_rl.MeasureText.argtypes = [CharPtr , Int]
_rl.MeasureText.restype =Int
def measure_text(text: str , font_size: int)-> int:
    """Measure string width for default font"""
    return _rl.MeasureText(_to_byte_str(text) , _to_int(font_size))

def measure_text_ex(font: Font , text: str , font_size: float , spacing: float)-> 'Vector2':
    pass
_rl.TextIsEqual.argtypes = [CharPtr , CharPtr]
_rl.TextIsEqual.restype = Bool

def text_is_equal(text_1: str , text_2: str)-> bool:
    """Check if two text string are equal"""
    return _rl.TextIsEqual(_to_byte_str(text_1) , _to_byte_str(text_2))

# Input-related functions: keyboard
@python_wrapper('IsKeyPressed', [Int], Bool)
def is_key_pressed(key: int)-> bool:
    """Detect if a key has been pressed once"""
    return _rl.IsKeyPressed(_to_int(key))

@python_wrapper('IsKeyDown', [Int], Bool)
def is_key_down(key: Union[Keyboard, int])-> bool:
    """Detect if a key is being pressed"""
    return _rl.IsKeyDown(_to_int(key))

@python_wrapper('IsKeyReleased', [Int], Bool)
def is_key_released(key: Union[Keyboard, int])-> bool:
    """Detect if a key has been released once"""
    return _rl.IsKeyReleased(_to_int(key))

@python_wrapper('IsKeyUp', [Int], Bool)
def is_key_up(key: Union[Keyboard, int])-> bool:
    """Detect if a key is NOT being pressed"""
    return _rl.IsKeyUp(_to_int(key))

@python_wrapper('SetExitKey', [Int])
def set_exit_key(key: Union[Keyboard, int])-> None:
    """Set a custom key to exit program (default is ESC)"""
    _rl.SetExitKey(_to_int(key))

@python_wrapper('GetKeyPressed', None, Int)
def get_key_pressed()-> int:
    """Get key pressed (keycode), call it multiple times for keys queued"""
    return _rl.GetKeyPressed()

@python_wrapper('GetCharPressed', None, Int)
def get_char_pressed()-> int:
    """Get char pressed (unicode), call it multiple times for chars queued"""
    return _rl.GetCharPressed()

# Input-related functions: gamepads

wrap_function('IsGamepadAvailable', [Int], Bool)
def is_gamepad_available(gamepad: int)-> bool:
    """Detect if a gamepad is available"""
    return _rl.IsGamepadAvailable(_to_int(gamepad))

wrap_function('IsGamepadName', [Int, CharPtr], Bool)
def is_gamepad_name(gamepad: int, name: str)-> bool:
    """Check gamepad name (if available)"""
    return _rl.IsGamepadName(_to_int(gamepad), _to_byte_str(name))

wrap_function('GetGamepadName', [Int], CharPtr)
def get_gamepad_name(gamepad: int)-> str:
    """Return gamepad internal name id"""
    return _to_str(
        _rl.GetGamepadName(_to_int(gamepad))
    )

wrap_function('IsGamepadButtonPressed', [Int, Int], Bool)
def is_gamepad_button_pressed(gamepad: int, button: Union[GamepadButton, int])-> bool:
    """Detect if a gamepad button has been pressed once"""
    return _rl.IsGamepadButtonPressed(_to_int(gamepad), _to_int(button))

wrap_function('IsGamepadButtonDown', [Int, Int], Bool)
def is_gamepad_button_down(gamepad: int, button: Union[GamepadButton, int])-> bool:
    """Detect if a gamepad button is being pressed"""
    return _rl.IsGamepadButtonDown(_to_int(gamepad), _to_int(button))

wrap_function('IsGamepadButtonReleased', [Int, Int], Bool)
def is_gamepad_button_released(gamepad: int, button: Union[GamepadButton, int])-> bool:
    """Detect if a gamepad button has been released once"""
    return _rl.IsGamepadButtonReleased(_to_int(gamepad), _to_int(button))

wrap_function('IsGamepadButtonUp', [Int, Int], Bool)
def is_gamepad_button_up(gamepad: int, button: Union[GamepadButton, int])-> bool:
    """Detect if a gamepad button is NOT being pressed"""
    return _rl.IsGamepadButtonUp(_to_int(gamepad), _to_int(button))

wrap_function('GetGamepadButtonPressed', restype=Int)
def get_gamepad_button_pressed()-> int:
    """Get the last gamepad button pressed"""
    return _rl.GetGamepadButtonPressed()

wrap_function('GetGamepadAxisCount', [Int], Int)
def get_gamepad_axis_count(gamepad: int)-> int:
    """Return gamepad axis count for a gamepad"""
    return _rl.GetGamepadAxisCount(_to_int(gamepad))

wrap_function('GetGamepadAxisMovement', [Int, Int], Float)
def get_gamepad_axis_movement(gamepad: int, axis: int)-> float:
    """Return axis movement value for a gamepad axis"""
    return _rl.GetGamepadAxisMovement(_to_int(gamepad), _to_int(axis))

wrap_function('SetGamepadMappings', [CharPtr], Int)
def set_gamepad_mappings(mappings: str)-> int:
    """Set internal gamepad mappings (SDL_GameControllerDB)"""
    return _rl.SetGamepadMappings(_to_byte_str(mappings))

# Input-related functions: mouse

wrap_function('IsMouseButtonPressed', [Int], Bool)
def is_mouse_button_pressed(button: Union[MouseButton, int])-> bool:
    """Detect if a mouse button has been pressed once"""
    return _rl.IsMouseButtonPressed(_to_int(button))

wrap_function('IsMouseButtonDown', [Int], Bool)
def is_mouse_button_down(button: Union[MouseButton, int])-> bool:
    """Detect if a mouse button is being pressed"""
    return _rl.IsMouseButtonDown(_to_int(button))

wrap_function('IsMouseButtonReleased', [Int], Bool)
def is_mouse_button_released(button: Union[MouseButton, int])-> bool:
    """Detect if a mouse button has been released once"""
    return _rl.IsMouseButtonReleased(_to_int(button))

wrap_function('IsMouseButtonUp', [Int], Bool)
def is_mouse_button_up(button: Union[MouseButton, int])-> bool:
    """Detect if a mouse button is NOT being pressed"""
    return _rl.IsMouseButtonUp(_to_int(button))

wrap_function('GetMouseX', None, Int)
def get_mouse_x()-> int:
    """Returns mouse position X"""
    return _rl.GetMouseX()

wrap_function('GetMouseY', None, Int)
def get_mouse_y()-> int:
    """Returns mouse position Y"""
    return _rl.GetMouseY()

wrap_function('GetMousePosition', None, Vector2)
def get_mouse_position()-> Vector2:
    """Returns mouse position XY"""
    return _rl.GetMousePosition()

wrap_function('SetMousePosition', [Int, Int])
def set_mouse_position(position: Union[Sequence[int], Vector2])-> None:
    """Set mouse position XY"""
    x , y = _flatten((int, float), position, map_to=int)
    _rl.SetMousePosition(x, y)

wrap_function('SetMouseOffset', [Int, Int])
def set_mouse_offset(offset: Union[Sequence[int], Vector2])-> None:
    """Set mouse offset"""
    x , y = _flatten((int, float), offset, map_to=int)
    _rl.SetMouseOffset(x, y)

wrap_function('SetMouseScale', [Float, Float])
def set_mouse_scale(scale: Union[Sequence[int], Vector2])-> None:
    """Set mouse scaling"""
    x , y = _flatten((int, float), scale, map_to=float)
    _rl.SetMouseOffset(x, y)

wrap_function('GetMouseWheelMove', None, Float)
def get_mouse_wheel_move()-> float:
    """Returns mouse wheel movement Y"""
    return _rl.GetMouseWheelMove()

wrap_function('SetMouseCursor', [Int])
def set_mouse_cursor(cursor: int)-> None:
    """Set mouse cursor"""
    _rl.SetMouseCursor(_to_int(cursor))

# Input-related functions: touch
wrap_function('GetTouchX', restype=Int)
def get_touch_x()-> int:
    """Returns touch position X for touch point 0 (relative to screen size)"""
    return _rl.GetTouchX()

wrap_function('GetTouchY', restype=Int)
def get_touch_y()-> int:
    """Returns touch position Y for touch point 0 (relative to screen size)"""
    return _rl.GetTouchY()

wrap_function('GetTouchPosition', [Int], Vector2)
def get_touch_position(index: int)-> Vector2:
    """Returns touch position XY for a touch point index (relative to screen size)"""
    return _rl.GetTouchPosition(_to_int(index))

# Gestures and Touch Handling Functions (Module: gestures)

wrap_function('SetGesturesEnabled', [UInt])
def set_gestures_enabled(flags: Union[Gestures, int])-> None:
    """Enable a set of gestures using flags"""
    _rl.SetGesturesEnabled(_to_int(flags))

wrap_function('IsGestureDetected', [Int], Bool)
def is_gesture_detected(gesture: int)-> bool:
    """Check if a gesture have been detected"""
    return _rl.IsGestureDetected(_to_int(gesture))

wrap_function('GetGestureDetected', restype=Int)
def get_gesture_detected()-> int:
    """Get latest detected gesture"""
    return _rl.GetGestureDetected()

wrap_function('GetTouchPointsCount', restype=Int)
def get_touch_points_count()-> int:
    """Get touch points count"""
    return _rl.GetTouchPointsCount()

wrap_function('GetGestureHoldDuration', restype=Float)
def get_gesture_hold_duration()-> float:
    """Get gesture hold time in milliseconds"""
    return _rl.GetGestureHoldDuration()

wrap_function('GetGestureDragVector', restype=Vector2)
def get_gesture_drag_vector()-> Vector2:
    """Get gesture drag vector"""
    return _rl.GetGestureDragVector()

wrap_function('GetGestureDragAngle', restype=Float)
def get_gesture_drag_angle()-> float:
    """Get gesture drag angle"""
    return _rl.GetGestureDragAngle()

wrap_function('GetGesturePinchVector', restype=Vector2)
def get_gesture_pinch_vector()-> Vector2:
    """Get gesture pinch delta"""
    return _rl.GetGesturePinchVector()

wrap_function('GetGesturePinchAngle', restype=Float)
def get_gesture_pinch_angle()-> float:
    """Get gesture pinch angle"""
    return _rl.GetGesturePinchAngle()

# Misc. functions
wrap_function('GetRandomValue', [Int, Int], Int)
def get_random_value(min: int, max: int)-> int:
    """Returns a random value between min and max (both included)"""
    return _rl.GetRandomValue(
        _to_int(min),
        _to_int(max)
    )

wrap_function('TakeScreenshot', [CharPtr])
def take_screenshot(filename: str = 'screenshot.png')-> None:
    _rl.TakeScreenshot(
        _to_byte_str(filename)
    )

wrap_function('MemAlloc', [Int], VoidPtr)
def allocate_memory(size: int)-> VoidPtr:
    """
    Internal memory allocator
    size: number of bytes to allocate
    """
    return _rl.MemAlloc(_to_int(size))

wrap_function('MemRealloc', [VoidPtr, Int])
def re_allocate_memory(ptr: VoidPtr, size: int)-> VoidPtr:
    """
    Internal memory free
    ptr: The pointer which is pointing the previously allocated memory block by allocate_memory.
    size: The new size of memory block.
    """
    return _rl.MemRealloc(
        ptr,
        _to_int(size)
    )

wrap_function('MemFree', [VoidPtr])
def free_memory(ptr: VoidPtr)-> None:
    """Internal memory free"""
    _rl.MemFree(ptr)

# Files management functions
wrap_function('LoadFileData', restype=UCharPtr, argtypes=[CharPtr, UIntPtr])
def load_file_data(filename: Union[str, Path])-> bytes:
    bytes_read = UInt()
    data = _rl.LoadFileData(_to_byte_str(str(filename)), byref(bytes_read) )
    return data

wrap_function('UnloadFileData', [UCharPtr])
def unload_file_data(data: UCharPtr)-> None:
    _rl.UnloadFileData(data)

wrap_function('FileExists', [CharPtr], Bool)
def file_exists(filename: Union[str, Path])-> bool:
    """Check if file exists"""
    return _rl.FileExists(_to_byte_str(str(filename)))

wrap_function('DirectoryExists', [CharPtr], Bool)
def directory_exists(dirPath: Union[str, Path])-> bool:
    """Check if a directory path exists"""
    return _rl.DirectoryExists(_to_byte_str(str(dirPath)))

wrap_function('IsFileExtension', [CharPtr, CharPtr], Bool)
def is_file_extension(file: Union[str, Path], ext: str):
    """Check file extension (including point: .png, .wav)"""
    return _rl.IsFileExtension(
        _to_byte_str(str(file)),
        _to_byte_str(ext)
    )

wrap_function('GetDroppedFiles', [IntPtr], CharPtrPtr)
def get_dropped_files()-> List[str]:
    """
    Get dropped files (names)
    """
    count = Int(0)
    c_carray = _rl.GetDroppedFiles(byref(count))
    files = []
    for i in range(count.value):
        files.append( _to_str(c_carray[i]) )
    clear_dropped_files()
    return files

wrap_function('ClearDroppedFiles')
def clear_dropped_files()-> None:
    """Clear dropped files paths buffer (free memory)"""
    _rl.ClearDroppedFiles()

wrap_function('GetDirectoryFiles', [CharPtr, IntPtr], CharPtrPtr)
def get_directory_files(dir_path: Union[str, Path])-> List[str]:
    """Get filenames in a directory path"""
    cnt = Int(0)
    c_array = _rl.GetDirectoryFiles(_to_byte_str(dir_path), byref(cnt))
    files: List[str] = []
    for i in range(cnt.value):
        files.append(
            _to_str(c_array[i])
        )
    clear_directory_files()
    return files

wrap_function('ClearDirectoryFiles')
def clear_directory_files()-> None:
    """Clear directory files paths buffers (free memory)"""
    _rl.ClearDirectoryFiles()
    
wrap_function('IsFileDropped', restype=Bool)
def is_file_dropped()-> bool:
    """Check if a file has been dropped into window"""
    return _rl.IsFileDropped()

wrap_function('GetFileExtension', [CharPtr], CharPtr)
def get_extension(filename: Union[str, Path])-> str:
    """Get a str extension for a filename string (includes dot: ".png")"""
    return _to_str(_rl.GetFileExtension(_to_byte_str(str(filename))))

wrap_function('GetFileName', [CharPtr], CharPtr)
def get_file_name(filepath: Union[str, Path])-> str:
    """Get str to filename for a path string"""
    return _to_str(
        _rl.GetFileName(_to_byte_str(str(filepath)))
    )

wrap_function('LoadFileText', [CharPtr, IntPtr], CharPtr)
def load_file_text(filename: Union[str, Path])->str:
    """Load text data from file (read), returns a '\0' terminated string"""
    length = Int(0)
    text = _to_str(
        _rl.LoadFileText(_to_byte_str(filename), byref(length))
    )
    return text

wrap_function('SaveFileText', [CharPtr, CharPtr], Bool)
def save_file_text(filename: Union[str, Path], text: str)-> bool:
    """Save text data to file (write), string must be '\0' terminated"""
    success = _rl.SaveFileText(
        _to_byte_str(filename),
        _to_byte_str(text),
    )
    return success

# Persistent storage management
wrap_function('SaveStorageValue', [UInt, Int], Bool)
def save_storage_value(position: int, value: int)-> bool:
    """Save integer value to storage file (to defined position), returns true on success"""
    return _rl.SaveStorageValue(_to_int(position), _to_int(value))

wrap_function('LoadStorageValue', [UInt], Int)
def load_storage_value(position: int)-> int:
    """Load integer value from storage file (from defined position)"""
    return _rl.LoadStorageValue(_to_int(position))

wrap_function('OpenURL', [CharPtr], Bool)
def open_url(url: str)-> None:
    _rl.OpenURL(_to_byte_str(url))

# Basic geometric 3D shapes drawing functions
wrap_function('DrawGrid', [Int, Float])
def draw_grid(slices: int, spacing: float)-> None:
    """Draw a grid (centered at (0, 0, 0))"""
    _rl.DrawGrid(_to_int(slices), _to_float(spacing))

def get_random_color()-> Tuple[int,int,int,int]:
    """Return a 32 bit color"""
    return Color(
        get_random_value(0, 255), 
        get_random_value(0,255),
        get_random_value(0,255),
        get_random_value(0,255))

# Basic shapes drawing functions

wrap_function('DrawPixelV', [Vector2, Color])
def draw_pixel(pos: Union[Sequence[float], Vector2], color: Union[Sequence[int], Color] = BLACK)-> None:
    """Draw a pixel"""
    _rl.DrawPixelV(_vec2(pos) ,_to_color(color))

wrap_function('DrawLineEx', [Vector2, Vector2, Float, Color])
def draw_line(
    start_position: Union[Sequence[float], Vector2], 
    end_position: Union[Sequence[float], Vector2],
    color: Union[Sequence[int], Color] = BLACK,
    thickness: float = 1,
    bezier: Optional[bool] = False
    )-> None:
    """Draw a line defining thickness"""
    if bezier:
        draw_line_bezier(
            _vec2(start_position),
            _vec2(end_position),
            _to_color(color),
            _to_float(thickness),
        )
    else:
        _rl.DrawLineEx(
            _vec2(start_position),
            _vec2(end_position),
            _to_float(thickness),
            _to_color(color),
        )

wrap_function('DrawLineBezier', [Vector2, Vector2, Float, Color])
def draw_line_bezier(
    start_position: Union[Sequence[float], Vector2], 
    end_position: Union[Sequence[float], Vector2],
    color: Union[Sequence[int], Color] = BLACK,
    thickness: float = 1
    )-> None:
    """Draw a line using cubic-bezier curves in-out"""
    _rl.DrawLineBezier(
        _vec2(start_position),
        _vec2(end_position),
        _to_float(thickness),
        _to_color(color),
    )

wrap_function('DrawLineBezierQuad', [Vector2, Vector2, Vector2, Float, Color])
def draw_line_bezier_quad(
        start_position: Union[Sequence[float], Vector2], 
        end_position: Union[Sequence[float], Vector2],
        control_pos: Union[Sequence[float], Vector2],
        color: Union[Sequence[int], Color] = BLACK,
        thickness: float = 1
    )-> None:
    """Draw line using quadratic bezier curves with a control point"""
    _rl.DrawLineBezierQuad(
        _vec2(start_position),
        _vec2(end_position),
        _vec2(control_pos),
        _to_float(thickness),
        _to_color(color),
    )

wrap_function('DrawEllipse', [Int, Int, Float, Float, Color])
def draw_ellipse(
    center_xy: Union[Sequence[int], Vector2], 
    radius_hv: Union[Sequence[int], Vector2], 
    color: Union[Sequence[int], Color],
    outline: Optional[bool] = False,
    )-> None:
    """
    Draw ellipse
    or Draw ellipse outline if outline set to True
    """
    if outline:
        draw_ellipse_lines(
            center_xy,
            radius_hv,
            color
        )
    else:
        return _rl.DrawEllipse(
            _to_int(center_xy[0]),
            _to_int(center_xy[1]),
            _to_float(radius_hv[0]),
            _to_float(radius_hv[1]),
            _to_color(color),
        )

wrap_function('DrawEllipseLines', [Int, Int, Float, Float, Color])
def draw_ellipse_lines(
    center_xy: Union[Sequence[int], Vector2], 
    radius_hv: Union[Sequence[int], Vector2], 
    color: Union[Sequence[int], Color],
    )-> None:
    """Draw ellipse outline"""
    _rl.DrawEllipseLines(
        _to_int(center_xy[0]),
        _to_int(center_xy[1]),
        _to_float(radius_hv[0]),
        _to_float(radius_hv[1]),
        _to_color(color),
    )

wrap_function('DrawRing', [Vector2, Float, Float, Float, Float, Int, Color])
def draw_ring(
    center: Union[Sequence[float], Vector2],
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
    color: Union[Sequence[int], Color]= (255, 255, 255, 255),
    outline: Optional[bool] = False,
    )-> None:
    """Draw ring"""
    if outline:
        draw_ring_lines(
            center,
            inner_radius,
            outer_radius,
            start_angle,
            end_angle,
            segments,
            color,
        )
    else:
        _rl.DrawRing(
            _vec2(center),
            _to_float(inner_radius),
            _to_float(outer_radius),
            _to_float(start_angle),
            _to_float(end_angle),
            _to_int(segments),
            _to_color(color)
        )

wrap_function('DrawRingLines', [Vector2, Float, Float, Float, Float, Int, Color])
def draw_ring_lines(
    center: Union[Sequence[float], Vector2],
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
    color: Union[Sequence[int], Color]= (255, 255, 255, 255)
    )-> None:
    """Draw ring outline"""
    _rl.DrawRingLines(
        _vec2(center),
        _to_float(inner_radius),
        _to_float(outer_radius),
        _to_float(start_angle),
        _to_float(end_angle),
        _to_int(segments),
        _to_color(color)
    )

wrap_function('DrawTriangle', [Vector2, Vector2, Vector2, Color])
def draw_triangle(
    points: Union[Sequence[Vector2], Sequence[Sequence[float]]],
    color: Union[Sequence[int], Vector4, Color] = (255, 255, 255, 255),
    outline: Optional[bool] = False,
    )-> None:
    """
    Draw a color-filled triangle (vertex in counter-clockwise order!)
    or Draw triangle outline (vertex in counter-clockwise order!)
    if outline set to True
    """
    if outline:
        draw_triangle_lines(
            points,
            color,
        )
    else:
        v1, v2, v3 = (_vec2(point) for point in points)
        _rl.DrawTriangle(v1, v2, v3, _to_color(color))

wrap_function('DrawTriangleLines', [Vector2, Vector2, Vector2, Color])
def draw_triangle_lines(
    points: Union[Sequence[Vector2], Sequence[Sequence[float]]],
    color: Union[Sequence[int], Vector4, Color] = (255, 255, 255, 255),
    )-> None:
    """Draw triangle outline (vertex in counter-clockwise order!)"""
    v1, v2, v3 = (_vec2(point) for point in points)
    color = _to_color(color)
    return _rl.DrawTriangleLines(
        v1,
        v2,
        v3,
        color,
    )

wrap_function('DrawRectangle', [Int,Int,Int,Int, Color])
def draw_rectangle(pos_x: int, pos_y: int, width: int, height: int, color: Union[Sequence[int], Color])-> None:
    """Draw a color-filled rectangle"""
    _rl.DrawRectangle(
        _to_int(pos_x),
        _to_int(pos_y),
        _to_int(width),
        _to_int(height),
        _to_color(color),
    )

wrap_function('DrawRectangleV', [Vector2, Vector2, Color])
def draw_rectangle_v(position: Union[Sequence[Number], Vector2], size: Union[Sequence[Number], Vector2], color: Union[Sequence[int], Color])-> None:
    """Draw a color-filled rectangle (Vector version)"""
    _rl.DrawRectangleV(
        _vec2(position),
        _vec2(size),
        _to_color(color)
    )

wrap_function('DrawPoly', [Vector2, Int, Float, Float, Color])
def draw_poly(
    center: Union[Sequence[Number], Vector2], 
    sides: int,
    radius: float,
    rotation: float,
    color: Union[Sequence[int], Color])-> None:
    """Draw a regular polygon (Vector version)"""
    _rl.DrawPoly(
        _vec2(center),
        _to_int(sides),
        _to_float(radius),
        _to_float(rotation),
        _to_color(color),
    )

wrap_function('DrawPolyLines', [Vector2, Int, Float, Float, Color])
def draw_poly_lines(
    center: Union[Sequence[Number], Vector2], 
    sides: int,
    radius: float,
    rotation: float,
    color: Union[Sequence[int], Color])-> None:
    """Draw a polygon outline of n sides"""
    _rl.DrawPolyLines(
        _vec2(center),
        _to_int(sides),
        _to_float(radius),
        _to_float(rotation),
        _to_color(color),
    )

# Basic geometric 3D shapes drawing functions
wrap_function('DrawLine3D', [Vector3, Vector3, Color])
def draw_line_3d(
    start_pos: Union[Sequence[Number], Vector3], 
    end_pos: Union[Sequence[Number], Vector3], 
    color: Union[Sequence[int], Color]
    )-> None:
    """Draw a line in 3D world space"""
    _rl.DrawLine3D(
        _vec3(start_pos),
        _vec3(end_pos),
        _to_color(color)
    )

wrap_function('DrawPoint3D', [Vector3, Color])
def draw_point_3d(
    position: Union[Sequence[Number], Vector3], 
    color: Union[Sequence[int], Color]
    )-> None:
    """Draw a point in 3D space, actually a small line"""
    _rl.DrawPoint3D(_vec3(position), _to_color(color))

wrap_function('DrawCircle3D', [Vector3, Float, Vector3, Float, Color])
def draw_circle_3d(
    center: Union[Sequence[Number], Vector3],
    radius: float,
    rotationAxis: Union[Sequence[Number], Vector3],
    rotationAngle: float,
    color: Union[Sequence[int], Color],
    ):
    """Draw a circle in 3D world space"""
    _rl.DrawCircle3D(
        _vec3(center),
        _to_float(radius),
        _vec3(rotationAxis),
        _to_float(rotationAngle),
        _to_color(color),
    )

wrap_function('DrawTriangle3D', [Vector3, Vector3, Vector3, Color])
def draw_triangle_3d(
    v1: Union[Sequence[Number], Vector3], 
    v2: Union[Sequence[Number], Vector3],
    v3: Union[Sequence[Number], Vector3],
    color: Union[Sequence[Number], Color])-> None:
    _rl.DrawTriangle3D(
        _vec3(v1),
        _vec3(v2),
        _vec3(v3),
        _to_color(color),
    )

wrap_function('DrawPlane', [Vector3, Vector2, Color])
def draw_plane(
    center_pos: Union[Sequence[Number], Vector3],
    size: Union[Sequence[Number], Vector2],
    color: Union[Sequence[int], Color]
    )-> None:
    """Draw a plane XZ"""
    _rl.DrawPlane(
        _vec3(center_pos),
        _vec2(size),
        _to_color(color)
    )

wrap_function('DrawCube', [Vector3, Float, Float, Float, Color])
def draw_cube(
    position: Union[Sequence[Number], Vector3],
    width: float, 
    height: float, 
    length: float, 
    color: Union[Sequence[int], Color])-> None:
    _rl.DrawCube(
        _vec3(position),
        _to_float(width),
        _to_float(height),
        _to_float(length),
        _to_color(color),
    )

wrap_function('DrawCubeWires', [Vector3, Float, Float, Float, Color])
def draw_cube_wires(
    position: Union[Sequence[Number], Vector3],
    width: float, 
    height: float, 
    length: float, 
    color: Union[Sequence[int], Color])-> None:
    """Draw cube wires"""
    _rl.DrawCubeWires(
        _vec3(position),
        _to_float(width),
        _to_float(height),
        _to_float(length),
        _to_color(color),
    )

wrap_function('DrawCubeV', [Vector3, Vector3, Color])
def draw_cube_v(
    position: Union[Vector3, Sequence[Number]],
    size: Union[Vector3, Sequence[Number]],
    color: Union[Sequence[int], Color])-> None:
    """Draw cube"""
    _rl.DrawCubeV(
        _vec3(position),
        _vec3(size),
        _to_color(color),
    )

wrap_function('DrawCubeWiresV', [Vector3, Vector3, Color])
def draw_cube_wires_v(
    position: Union[Vector3, Sequence[Number]],
    size: Union[Vector3, Sequence[Number]],
    color: Union[Sequence[int], Color])-> None:
    """Draw cube wires (Vector version)"""
    _rl.DrawCubeWiresV(
        _vec3(position),
        _vec3(size),
        _to_color(color),
    )

wrap_function('DrawCubeTexture',[Texture, Vector3, Float, Float, Float, Color])
def draw_cube_texture(
    texture: Texture,
    position: Union[Sequence[Number], Vector3],
    width: float, 
    height: float, 
    length: float, 
    color: Union[Sequence[int], Color]
    )-> None:
    """Draw cube textured"""
    _rl.DrawCubeTexture(
        texture,
        _vec3(position),
        _to_float(width),
        _to_float(height),
        _to_float(length),
        _to_color(color),
    )

wrap_function('DrawSphere', [Vector3, Float, Color])
def draw_sphere(
    center_pos: Union[Sequence[Number], Vector3], 
    radius: float, 
    color: Union[Sequence[int], Color]
    )-> None:
    """Draw sphere"""
    _rl.DrawSphere(
        _vec3(center_pos),
        _to_float(radius),
        _to_color(color),
    )

wrap_function('DrawSphereEx', [Vector3, Float, Color])
def draw_sphere_ex(
    center_pos: Union[Sequence[Number], Vector3], 
    radius: float,
    rings: int,
    slices: int,
    color: Union[Sequence[int], Color]
    )-> None:
    """Draw sphere with extended parameters"""
    _rl.DrawSphereEx(
        _vec3(center_pos),
        _to_float(radius),
        _to_int(rings),
        _to_int(slices),
        _to_color(color),
    )

wrap_function('DrawSphereWires', [Vector3, Float, Color])
def draw_sphere_ex(
    center_pos: Union[Sequence[Number], Vector3], 
    radius: float,
    rings: int,
    slices: int,
    color: Union[Sequence[int], Color],
    )-> None:
    """Draw sphere wires"""
    _rl.DrawSphereWires(
        _vec3(center_pos),
        _to_float(radius),
        _to_int(rings),
        _to_int(slices),
        _to_color(color),
    )
