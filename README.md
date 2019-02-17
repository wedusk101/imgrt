# C++

A simple ray tracer written in C++ for educational purposes only. I plan on implementing Peter Shirley's Ray tracing in a Weekend later on. This piece of
software is strictly work in progress and is a side project for getting the very basic understanding clear about ray tracing and rendering.
Please note that as of now, this code will NOT work for other scene configurations and has broken depth test and/or shadows. This is just a rough piece of code
and the design decisions made here are of a very low quality.

For details on the included image viewer please refer to the project page at

http://openseeit.sourceforge.net/

I used Photoshop's automation features for converting the .ppm image sequences to TIFF for rendering the video of the animation. I will be using some other library for rendering
to different image formats as Photoshop CC 2019 doesn't seem to recognize image sequences in PPM. The library "stb_image.h" seems to be a good choice.

https://github.com/nothings/stb

Last Updated - February 17, 2019