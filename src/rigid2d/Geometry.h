#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <cassert>
#include "Constants.h"
#define hypot _hypot

inline int sgn(double x) {
	if (x < -eps) return -1;
	if (x > eps) return 1;
	return 0;
}

class Angle {
public:
	double c, s;
	Angle(double ang = 0) {
		c = cos(ang);
		s = sin(ang);
	}
};

class Vector2D {
public:
	double x, y, z;
	Vector2D() {}
	Vector2D(double x, double y, double z) : x(x), y(y), z(z) {}
	Vector2D operator += (const Vector2D &vec) {
		return *this = Vector2D(x + vec.x, y + vec.y, z + vec.z);
	}
	Vector2D operator -= (const Vector2D &vec) {
		return *this = Vector2D(x - vec.x, y - vec.y, z - vec.z);
	}
	Vector2D operator *= (const double &a) {
		return *this = Vector2D(x * a, y * a, z);
	}
	Vector2D operator /= (const double &a) {
		return *this = Vector2D(x / a, y / a, z);
	}
	Vector2D operator / (const double &a) const {
		return Vector2D(x / a, y / a, z);
	}
	Vector2D operator +(const Vector2D &vec) const {
		return Vector2D(x + vec.x, y + vec.y, z + vec.z);
	}
	Vector2D operator -(const Vector2D &vec) const {
		return Vector2D(x - vec.x, y - vec.y, z - vec.z);
	}
	Vector2D operator -() const{
		return Vector2D(-x, -y, z);
	}
	double operator *(const Vector2D &vec) const {
		return x * vec.x + y * vec.y;
	}
	double operator %(const Vector2D &vec) const {
		return x * vec.y - y * vec.x;
	}
	double GetAngle() const{
		return atan2(y, x); 
	}
	void Normalize() {
		double r = 1.0 / GetLength();
		x *= r, y *= r;
	}
	Vector2D GetDirection() const {
		double r = 1.0 / GetLength();
		return Vector2D(x * r, y * r, 0);
	}
	double GetLength() const {
		return sqrt(GetLength2());
	}
	double GetLength2() const {
		return x * x + y * y;
	}
	Vector2D GetRotate(double angle) const {
		return Vector2D(x * cos(angle) - y * sin(angle), y * cos(angle) + x * sin(angle), z);
	}
	Vector2D GetRotate() const {
		return Vector2D(-y, x, z);
	}
	Vector2D GetInverseRotate() const {
		return Vector2D(y, -x, z);
	}
	Vector2D GetVector() const {
		return Vector2D(x, y, 0);
	}
	Vector2D GetPosition() const {
		return Vector2D(x, y, 1);
	}
	Vector2D GetRotated(Angle ang) const {
		return Vector2D(x * ang.c - y * ang.s, x * ang.s + y * ang.c, z);
	}
	void Rotate(Angle ang) {
		*this = GetRotated(ang);
	}
	static Vector2D RotatedUnitVector(double angle) {
		return Vector2D(cos(angle), sin(angle), 0);
	}
	static Vector2D Origin;
};


inline bool operator <(Vector2D a, Vector2D b) {return false;}

inline bool operator == (Vector2D a, Vector2D b) {
	return sgn(a.x - b.x) == 0 && sgn(a.y - b.y) == 0 && sgn(a.z - b.z) == 0;
}

inline Vector2D operator *(double a, const Vector2D &vec) {
	return Vector2D(a * vec.x, a * vec.y, vec.z);	
}

inline Vector2D operator *(const Vector2D &vec, double a) {
	return Vector2D(a * vec.x, a * vec.y, vec.z);	
}

class Line {
public:
	Vector2D a, b, v;
	Line() {}
	Line(Vector2D a, Vector2D b) : a(a), b(b), v(b - a) {}
	bool IsParallel(const Line &l) const {
		return abs(l.v % v) < eps;
	}
	bool IsInside(const Vector2D &p) const {
		return sgn((p - a) * v) * sgn((p - b) * v) < eps;
	}
};

inline Vector2D operator *(const Line &a, const Line &b) {
	return b.a + (((a.a - b.a) % a.v) / (b.v % a.v)) * b.v;
}

struct Matrix3x3 {
	double a[3][3];
	double *operator[] (int i) {return a[i];}
	const double *operator[] (int i) const {return a[i];}
	Matrix3x3() {memset(a, 0, sizeof(a));}
	Matrix3x3 Transpose() {
		Matrix3x3 res = *this;
		swap(res[0][1], res[1][0]);
		swap(res[0][2], res[2][0]);
		swap(res[1][2], res[2][1]);
		return res;
	}
	Vector2D operator() (Vector2D vec) const {
		return Vector2D(vec.x * a[0][0] + vec.y * a[0][1] + vec.z * a[0][2],
						vec.x * a[1][0] + vec.y * a[1][1] + vec.z * a[1][2],
						vec.x * a[2][0] + vec.y * a[2][1] + vec.z * a[2][2]);
	}
	void SetAsAxisTransform(Vector2D movement, double rotationAngle) {
		a[0][0] = cos(rotationAngle); a[0][1] = -sin(rotationAngle);
		a[1][0] = -a[0][1], a[1][1] = a[0][0];
		a[0][2] = movement.x; a[1][2] = movement.y;
		a[2][0] = a[2][1] = 0; a[2][2] = 1;
	}
	void SetAsAxisTransformInverse(Vector2D movement, double rotationAngle) {
		double cosV = cos(rotationAngle), sinV = sin(rotationAngle);
		a[0][0] = a[1][1] = cosV;
		a[0][1] = sinV; a[1][0] = -sinV;
		a[2][0] = a[2][1] = 0; a[2][2] = 1;
		a[0][2] = -movement.x * cosV - movement.y * sinV;
		a[1][2] = movement.x * sinV - movement.y * cosV;
	}
};

inline Matrix3x3 operator *(const Matrix3x3 &a, const Matrix3x3 &b) {
	Matrix3x3 c;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++) 
				c[i][j] += a[i][k] * b[k][j];
	return c;
} 

#define CONVEX_CCW 1
#define CONVEX_CW 2

vector<vector<Vector2D> > CutIntoConvex(vector<Vector2D> points);

#endif
