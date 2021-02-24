#include <fstream>
#include <cmath>
#include <iostream> 
#include <string>
#include <chrono>
#include <climits>
#include <cstdlib>

struct Color3f
{
	float r;
	float g;
	float b;
};

void renderCPU(Color3f *fb, int width, int height)
{
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int index = y * width + x;
			fb[index].r = x / (float)width;
			fb[index].g = y / (float)height;
			fb[index].b = (fb[index].r + fb[index].g) / 2;
		}
	}
}

int main()
{
	int width = 7680;
	int height = 4320;

	int numPixels = width * height;
	size_t fbSize = numPixels * sizeof(Color3f);

	Color3f *fb = new Color3f[fbSize];

	auto start = std::chrono::high_resolution_clock::now();
	renderCPU(fb, width, height);
	auto stop = std::chrono::high_resolution_clock::now();	
	auto diff = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

	std::ofstream out("result.ppm"); // creates a PPM image file for saving the rendered output
	out << "P3\n" << width << " " << height << "\n255\n";

	for (int i = 0; i < numPixels; ++i)
		out << (int)(255.99 * fb[i].r) << " " << (int)(255.99 * fb[i].g) << " " << (int)(255.99 * fb[i].b) << "\n"; // write out the pixel values

	std::cout << "\nTime taken was " << diff.count() << " seconds." << std::endl;
	delete[] fb;
}