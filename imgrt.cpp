// author - wedusk101 (c) 2018
#include <fstream>
#include <cmath>

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
	
	Vec3 operator&(const Vec3 &v) // cross product
	{
		return Vec3(y * v.z - v.y * z, z * v.x - x * v.z, x * v.y - y * v.x);
	}
};

struct Ray
{
	Vec3 o;
	Vec3 d;
	
	Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_) {}
};

struct Sphere
{
	Vec3 center;
	double radius;
	Vec3 color;
	
	Sphere(const Vec3 &c, const double &rad, const Vec3 &col) : center(c), radius(rad), color(col) {}
	
	Vec3 getNormal(const Vec3 &point) // returns the surface normal at a point
	{
		return (point - center)/radius;
	}
	
	bool intersects(const Ray &ray, double &t) const
	{
		const double eps = 1e-4;
		const Vec3 oc = ray.o - center;
		const double b = 2 * (ray.o % oc);
		const double a = ray.d % ray.d;
		const double c = (oc % oc) - (radius * radius);
		double delta = b * b - 4 * a * c;
		if(delta < eps)
			return false;
		delta = sqrt(delta);
		const double t0 = (-b + delta) / (2 * a);
		const double t1 = (-b - delta) / (2 * a);
		t = (t0 < t1) ? t0 : t1;
		return true;
	}
};

struct Plane
{
	Vec3 normal;
	Vec3 point;
	Vec3 color;
	
	Plane(const Vec3 &n, const Vec3 &p, const Vec3 &c) : normal(n), point(p), color(c) {}
	
	Vec3 getNormal() const
	{
		return normal;
	}
	
	bool intersects(const Ray &ray, double &t) const
	{
		const double eps = 1e-4;;
		double parameter = ray.d % normal;
		if(fabs(parameter) < eps)
			return false;
		t = ((point - ray.o) % normal) / parameter;
		return true;
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
	
	Camera(const Vec3 &pos, const Vec3 &dir) : position(pos), direction(dir) {}
};

double dot(const Vec3 &a, const Vec3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 &a, const Vec3 &b)
{
	return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

Vec3 colorModulate(const Vec3 &light, const Vec3 &object) // performs component wise multiplication for colors - please note that the parameter list is order sensitive 
{
	return Vec3((light.x / 255) * object.x, (light.y / 255) * object.y, (light.z / 255) * object.z);
}

void clamp(Vec3 &col)
{
	col.x = (col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x;
	col.y = (col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y;
	col.z = (col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z;
}

int main()
{
	
	// setup resolution, camera, colors, objects and lights
	
	const int height = 480;
	const int width = 640;
	
	// colors (R, G, B)
	const Vec3 white(255, 255, 255);
	const Vec3 black(0, 0, 0);
	const Vec3 red(255, 0, 0);
	const Vec3 green(0, 255, 0);
	const Vec3 blue(0, 0, 255);
	const Vec3 cyan(0, 255, 255);
	const Vec3 magenta(255, 0, 255);
	const Vec3 yellow(255, 255, 0);
	
	const Camera camera(Vec3(0.5 * height, 0.5 * width, 0), Vec3(0, 0, 1)); // scene camera
		
	// scene objects and lights
	Sphere sphere(Vec3(0.5 * height, 0.5 * width, 100), 5, blue); // blue sphere
	// Plane plane(Vec3(0, 0, -1), Vec3(0.5 * height, 0.5 * width, 500), red); // red plane
	Light light(Vec3(0.25 * height, 0.25 * width, 25), 1, white, 0.5); // white scene light
	
	std::ofstream out("output.ppm");
	out << "P3\n" << width << " " << height << "\n255\n";
	
	double t = 0, bak = 0;
	bool depthTest = false;
	
	const double ambientIntensity = 0.5;
	const Vec3 ambient(128, 0, 0);	// light red ambient light
	Vec3 pixelColor(0, 0, 0);	// set background color to black 
	
	
	for(int y = 0; y < height; y++)
	{
		pixelColor = black; // default color of each pixel
		for(int x = 0; x < width; x++)
		{
			const Ray cameraRay(Vec3(x, y, 0), camera.direction);
			depthTest = false; // used for determining whether a ray has already intersected the sphere before intersecting the plane
			if(sphere.intersects(cameraRay, t))
			{
				Vec3 surf = cameraRay.o + cameraRay.d * t;
				Vec3 L = (light.position - surf).getNormalized();
				Vec3 N = sphere.getNormal(surf).getNormalized();
				
				double diffuse = dot(L, N);
				pixelColor = (colorModulate(light.color, sphere.color) + white * diffuse) * light.intensity + ambient * ambientIntensity;
				bak = t;
				depthTest = true;
				clamp(pixelColor);
			}
			
			/*if(plane.intersects(cameraRay, t) && !depthTest || plane.intersects(cameraRay, t) && depthTest && t < bak)
			{
				Vec3 surf = cameraRay.o + cameraRay.d * t;
				Vec3 L = (light.position - surf).getNormalized();
				Vec3 N = plane.getNormal().getNormalized();
				
				double diffuse = dot(L, N);
				pixelColor = (colorModulate(light.color, plane.color) + white * diffuse) * light.intensity + ambient * ambientIntensity; // white * diffuse = specular (non-Phong)
				clamp(pixelColor);
			}*/
			
			out << (int)pixelColor.x << " " << (int)pixelColor.y << " " << (int)pixelColor.z << "\n"; // write out the pixel values
		}
	}
}