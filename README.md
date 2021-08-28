<img align="left" src="https://github.com/Ho011/pyraylib/blob/main/logo/pyraylib_256x256.png" width=256>

# pyraylib

[![Downloads](https://pepy.tech/badge/pyraylib)](https://pepy.tech/project/pyraylib)
[![Downloads](https://pepy.tech/badge/pyraylib/month)](https://pepy.tech/project/pyraylib)

A python binding for the great _C_ library **[raylib](https://github.com/raysan5/raylib)**.
The library provides object-oriented wrappers around raylib's struct interfaces.

## Getting Started

<!--
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
-->

### Prerequisites

_pyraylib_ uses type [annotations](https://www.python.org/dev/peps/pep-3107/#id30) in its source, so a Python version that supports it is required.

Some Python versions may not have [enum](https://pypi.org/project/enum/) and/or [typings](https://pypi.org/project/typing/) modules as part of the standard library, wich are required. These are installed automatically by pip.

### Installing

The easiest way to install _pyraylib_ is by the pip install command:

Depending on you system and python version(s) installed, the command might be:

```python
pip install pyraylib
```

or

```python
python -m pip install pyraylib
```

or (with Python3.7 launcher with multiple versions installed)

```python
py-3.x-32 -m pip install pyraylib
```

> Note that the minimum Python version tested is 3.4. Please, let me know if you're able to run it in Python33.

_pyraylib_ comes with 32bit binaries for Windows, Mac and Linux, but you're not required to use these. If you have a custom _raylib_ _**dll**_, _**dylib**_ or _**so**_ binary, make sure to set a PATH indicating the directory it is located:

```python
import os

# set the path before raylib is imported.
os.environ["RAYLIB_PATH"] = "path/to/the/binary"

import pyraylib

# let the fun begin.
```

You can set `"__file__"` as value to `"RAYLIB_PATH"` and _pyraylib_ will search for the binary in the package dir:

```python
# bynary file is wherever the package is located.
os.environ["RAYLIB_PATH"] = "__file__"
```

`"__main__"` can also be set to look for the binary in the project's directory where the starting script is located:

```python
# binary file is in the same dir as this py file.
os.environ["RAYLIB_BIN_PATH"] = "__main__"

# ...

if __name__ == "__main__":
    # run the game
    # ...
```

> Make sure the bin file name for the respective platform is `raylib.dll`, `libraylib.3.7.0.dylib` or `libraylib.so`.

## Using pyraylib

Using pyraylib is as simple as this:

```python
import pyraylib
from pyraylib.colors import (
    LIGHTGRAY,
    RAYWHITE
)
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
```

The `examples/` directory contains more examples.

## Tests

_pyraylib_ does not have test code, but you can run the examples in the [examples directory](https://github.com/Ho011/pyraylib/tree/main/examples).
>
<!--
### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

-->

## _raylib_ vs _pyraylib_

Below are the differences in usage between _raylib_ and _pyraylib_. Note, though that these differences are being worked to make _pyraylib_ as pythonic as possible, so changes may occur without notification.

### Constant values

All C `#define`s got translated to Python 'constants'. Enums got translated to
Python [enums](https://docs.python.org/3/library/enum.html).

### Structures

In general, all structures inherit from `ctypes.Structure` class. At the moment, constructors
(except for vectors) require the exact argument types, so `int`s can't be passed
where `float`s are expected (although the argument can be omitted).

All structures have `__str__()` implemented, so they have a very basic textual representation:

## Contributing

Contributions of any kind welcome!

## Authors

* **Ramon Santamaria** - *raylib's author* - [raysan5](https://github.com/raysan5)
* **Hussein Sarea** - *python binding code* - [pyraylib](https://github.com/Ho011/pyraylib)

## License

_pyraylib_ (and _raylib_) is licensed under an unmodified zlib/libpng license, which is an OSI-certified, BSD-like license that allows static linking with closed source software.

<!--
## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
-->
