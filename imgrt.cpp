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
	
	Sphere(const Vec3 &c, const double &rad) : center(c), radius(rad) {}
	
	Vec3 getNormal(const Vec3 &point) // returns the surface normal at a point
	{
		return (point - center)/radius;
	}
	
	bool intersects(const Ray &ray, double &t)
	{
		const double eps = 0.0004;
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

struct Light
{
	Vec3 position;
	double radius;
	Vec3 color;
	
	Light(const Vec3 &position_, const double &radius_, const Vec3 &color_) : position(position_), radius(radius_), color(color_) {}
};

double dot(const Vec3 &a, const Vec3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

void clamp(Vec3 &col)
{
	col.x = (col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x;
	col.y = (col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y;
	col.z = (col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z;
}

int main()
{
	// setup camera, colors and lights
	
	const Vec3 white(255, 255, 255);
	const Vec3 black(0, 0, 0);
	const Vec3 red(255, 0, 0);
	const Vec3 blue(0, 255, 0);
	const Vec3 green(0, 0, 255);
	
	const int height = 480;
	const int width = 640;
	
	Sphere obj(Vec3(0.5 * height, 0.5 * width, 200), 5);
	Light light(Vec3(0.25 * height, 0.25 * width, 25), 1, red);
	
	std::ofstream out("output.ppm");
	out << "P3\n" << width << " " << height << "\n255\n";
	
	double t = 0;
	Vec3 pixelColor(0, 0, 0); // set background color to black 
	
	for(int y = 0; y < height; y++)
	{
		pixelColor = black; // default color of each pixel
		for(int x = 0; x < width; x++)
		{
			const Ray cameraRay(Vec3(x, y, 0), Vec3(0, 0, 1));
			if(obj.intersects(cameraRay, t))
			{
				Vec3 surf = cameraRay.o + cameraRay.d * t;
				Vec3 L = light.position - surf;
				Vec3 N = obj.getNormal(surf);
				
				double diff = dot(L.getNormalized(), N.getNormalized());
				pixelColor = (light.color + white * diff) * 0.5;
				clamp(pixelColor);
			}
			
			out << (int)pixelColor.x << " " << (int)pixelColor.y << " " << (int)pixelColor.z << "\n"; // write out the pixel values
		}
	}
}