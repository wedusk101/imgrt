#include <fstream>
#include <cmath>
#include <iostream> 
#include <string>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <vector>
#include <cstring>
#include "omp.h"


struct Vec3
{
	double x;
	double y;
	double z;
	
	Vec3() : x(0), y(0), z(0) {}
	Vec3(const double &x_, const double &y_, const double &z_) : x(x_), y(y_), z(z_) {}
	
	double getMagnitude() const
	{
		return sqrt(x * x + y * y + z * z);
	}
	
	Vec3 getNormalized() const
	{
		double mag = getMagnitude();
		return Vec3(x/mag, y/mag, z/mag);
	}
	
	Vec3 operator+(const Vec3 &v) const // addition
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	
	Vec3 operator-(const Vec3 &v) const // subtraction
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	
	Vec3 operator*(const double &c) const // scalar multiplication
	{
		return Vec3(c * x, c * y, c * z);
	}
	
	Vec3 operator/(const double &c) const // scalar division
	{
		return Vec3(x/c, y/c, z/c);
	}
	
	double operator%(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}
	
	Vec3 operator&(const Vec3 &v) const // cross product
	{
		return Vec3(y * v.z - v.y * z, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	
	double dot(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}
};

struct Ray
{
	Vec3 o; // origin
	Vec3 d; // direction
    mutable float t;
	float tMin;
	mutable float tMax;
	
	Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_), t(INT_MAX), tMin(0.1), tMax(INT_MAX) {}
};

struct Geometry
{
	Vec3 color;

	virtual ~Geometry() {}

	virtual bool intersects(const Ray &ray) const = 0;
	virtual Vec3 getNormal(const Vec3 &point) const = 0;
};

struct Sphere : public Geometry
{
	Vec3 center;
	double radius;

	Sphere(const Vec3 &c, const double &rad, const Vec3 &col) : center(c), radius(rad)
    {
        color = col;
    }

	~Sphere() {}
	
	virtual Vec3 getNormal(const Vec3 &point) const // returns the surface normal at a point
	{
		return (point - center)/radius;
	}
	
	virtual bool intersects(const Ray &ray) const
	{
		const double eps = 1e-4;
		const Vec3 oc = ray.o - center;
		const double b = 2 * (ray.d % oc);
		const double a = ray.d % ray.d;
		const double c = (oc % oc) - (radius * radius);
		double delta = b * b - 4 * a * c;
		if(delta < eps) // discriminant is less than zero
			return false;
		delta = sqrt(delta);
		const double t0 = (-b + delta) / (2 * a);
		const double t1 = (-b - delta) / (2 * a);
		ray.t = (t0 < t1) ? t0 : t1;
		if (ray.t >= ray.tMin && ray.t <= ray.tMax)
		{
			ray.tMax = ray.t;
			return true;
		}
		else
			return false;
	}		
};

struct Plane : public Geometry
{
	Vec3 normal; // normal of the plane
	Vec3 point; // a point on the plane
	
	Plane(const Vec3 &n, const Vec3 &p, const Vec3 &col) : normal(n), point(p)
    {
        color = col;
    }

	~Plane() {}
	
	virtual Vec3 getNormal(const Vec3 &point) const
	{
		return normal;
	}
	
	virtual bool intersects(const Ray &ray) const
	{
		const double eps = 1e-4;;
		double parameter = ray.d % normal;
		if(fabs(parameter) < eps) // ray is parallel to the plane
			return false;
		ray.t = ((point - ray.o) % normal) / parameter;
		if (ray.t >= ray.tMin && ray.t <= ray.tMax)
		{
			ray.tMax = ray.t;
			return true;
		}
		else
			return false;
	}
};

struct Light
{
	Vec3 position;
	double radius;
	Vec3 color;
	double intensity;
	
	Light(const Vec3 &position_, const double &radius_, const Vec3 &color_, const double &intensity_) : position(position_), radius(radius_), color(color_), intensity(intensity_) {}
};

struct Camera
{
	Vec3 position;
	Vec3 direction;
	
	// add a lower left corner for orientation
	
	Camera(const Vec3 &pos, const Vec3 &dir) : position(pos), direction(dir) {}
};

inline double dot(const Vec3 &a, const Vec3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double getEuclideanDistance(const Vec3 &a, const Vec3 &b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
	return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

Vec3 colorModulate(const Vec3 &lightColor, const Vec3 &objectColor) // performs component wise multiplication for colors  
{
	return Vec3(lightColor.x * objectColor.x, lightColor.y * objectColor.y, lightColor.z * objectColor.z);
}

void clamp(Vec3 &col)
{
	col.x = (col.x > 1) ? 1 : (col.x < 0) ? 0 : col.x;
	col.y = (col.y > 1) ? 1 : (col.y < 0) ? 0 : col.y;
	col.z = (col.z > 1) ? 1 : (col.z < 0) ? 0 : col.z;
}

Vec3 getPixelColor(Ray &cameraRay, std::vector<Geometry*> scene, const Light *light)
{
    Vec3 ambient(0.25, 0, 0);	// light red ambient light
	double ambientIntensity = 0.25;
	Vec3 pixelColor = ambient * ambientIntensity;
	bool hitStatus = false;
	int hitIndex = 0, i = 0;
	
	for (auto &geo : scene)
	{
		if (geo->intersects(cameraRay))
		{
			hitStatus = true;		
			hitIndex = i;
		}
        i++;
	}

	if (hitStatus)
	{
		Vec3 surf = cameraRay.o + cameraRay.d * cameraRay.tMax; // point of intersection
		Vec3 L = (light->position - surf).getNormalized();

		// check for shadows
		Ray shadowRay(surf, L);
		for (auto &geo : scene)
			if (geo->intersects(shadowRay))
				return pixelColor;

		Vec3 N = scene[hitIndex]->getNormal(surf).getNormalized();
		float diffuse = L.dot(N);
		pixelColor = (colorModulate(light->color, scene[hitIndex]->color) * diffuse) * light->intensity;
		clamp(pixelColor);
	}

	return pixelColor;
}

void cleanup(Vec3 **frameBuffer, Light **light, std::vector<Geometry*> scene)
{
	delete [] *frameBuffer;
	delete *light;
	for (auto &geo : scene)
		delete geo;
}

int main(int argc, char* argv[])
{
    // colors (R, G, B)
	const Vec3 white(1, 1, 1);
	const Vec3 black(0, 0, 0);
	const Vec3 red(1, 0, 0);
	const Vec3 green(0, 1, 0);
	const Vec3 blue(0, 0, 1);
	const Vec3 cyan(0, 1, 1);
	const Vec3 magenta(1, 0, 1);
	const Vec3 yellow(1, 1, 0);

    // setup multithreading and benchmark parameters

    int nBenchLoops = 1; 
    int nThreads = 1;
    bool isBenchmark = false;

    for (int i = 0; i < argc; i++) // process command line args
    {
        if (!strcmp(argv[i], "-mt")) // enables multithreading using OpenMP
        {
            std::cout << "Multithreaded rendering enabled.\n";
            nThreads = std::thread::hardware_concurrency();
        }

        if (!strcmp(argv[i], "-bench")) // usage -bench numberLoops
        {
            isBenchmark = true;
            if (i + 1 < argc)
                nBenchLoops = atoi(argv[i+1]); // number of times to loop in benchmark mode
            else
            {
                std::cout << "Benchmark loop count not provided.\n";
                nBenchLoops = 5;
            }
            
        }
    } 

	// setup resolution, camera, colors, objects and lights	
	const int height = 4320;
	const int width = 7680;
    const int fbSize = height * width;
    Vec3 *fb = new Vec3[fbSize]; 

    // scene objects and lights
    std::vector<Geometry*> scene;
    
    scene.push_back(new Sphere(Vec3(0.5 * width, 0.45 * height, 1000), 100, Vec3(1, 0, 0)));
    scene.push_back(new Sphere(Vec3(0.65 * width, 0.2 * height, 600), 50, Vec3(0, 0, 1)));
    scene.push_back(new Plane(Vec3(0, 0, -1), Vec3(0.5 * width, 0.5 * height, 1500), Vec3(1, 1, 0)));
    scene.push_back(new Sphere(Vec3(0.5 * width, 0.52 * height, 700), 35, Vec3(0, 1, 1)));
	
	const Camera camera(Vec3(0.5 * width, 0.5 * height, 0), Vec3(0, 0, 1)); // scene camera	
	Light *light = new Light(Vec3(0.8 * width, 0.25 * height, 100), 1, white, 0.75); // white scene light		    		
	
    if (isBenchmark)
        std::cout << "Running in benchmark mode. Looping " << nBenchLoops << " times.\n";

    std::cout << "Rendering...\n";
	auto start = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < nBenchLoops; run++)
    {
#pragma omp parallel for num_threads(nThreads) shared(fb) schedule(dynamic, 1) 
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                size_t index = y * width + x;
                Ray cameraRay(Vec3(x, y, 0), camera.direction); // camera ray from each pixel 
            
                fb[index] = getPixelColor(cameraRay, scene, light);			
            }
        }
    }

	auto stop = std::chrono::high_resolution_clock::now(); 
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Rendering finished successfully.\n";
    std::cout << "\nTime taken to render was " << diff.count() << " milliseconds." << std::endl; 

    std::cout << "Saving render output...\n"; 
    std::ofstream out("result.ppm"); // creates a PPM image file for saving the rendered output
	out << "P3\n" << width << " " << height << "\n255\n";

    for (size_t i = 0; i < fbSize; i++)
        out << (int)(255.99 * fb[i].x) << " " << (int)(255.99 * fb[i].y) << " " << (int)(255.99 * fb[i].z) << "\n"; // write out the pixel values
    
    std::cout << "Render output saved successfully.\n";
	cleanup(&fb, &light, scene);
}