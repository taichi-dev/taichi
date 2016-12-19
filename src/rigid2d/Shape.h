#ifndef SHAPE_H
#define SHAPE_H

#include "Graphics.h"
#include "Geometry.h"

class Object;

class AABB {
public:
	double x0, y0, x1, y1;
	AABB(double x0, double y0, double x1, double y1) : x0(x0), y0(y0), x1(x1), y1(y1) {}
	AABB() {}	
	void Enlarge(double w = 1.0) {
		x0 -= w; y0 -= w;
		x1 += w; y1 += w;
	}
	AABB &operator += (const AABB &other) {
		x0 = min(x0, other.x0);
		y0 = min(y0, other.y0);
		x1 = max(x1, other.x1);
		y1 = max(y1, other.y1);

	}
	AABB &operator += (const Vector2D &vec) {
		x0 = min(x0, vec.x);
		y0 = min(y0, vec.y);
		x1 = max(x1, vec.x);
		y1 = max(y1, vec.y);
	}
	bool Overlap(const AABB &other) {
		return (!(x1 < other.x0 || other.x1 < x0)) && (!(y1 < other.y0 || other.y1 < y0));
	}
};

class Shape {
	friend class ShapeFactory;
	friend class Constraint;
	friend class Contact;
	friend class DistanceConstraint;
	friend class Force;
	friend class Spring;
	friend class Object;
	friend class ShapeFactory;
	friend class Physics;
public:
	Vector2D centroidPosition;
	double mass, inertia;
	double density;
	double restitution, friction;
	double boundaryWidth;
	Object *object;
	int layerMask;
	RGB3f color;
	Shape() {
		color = RGB3f::RandomBrightColor();
		density = 1.;
		centroidPosition = Vector2D(0, 0, 1);
		restitution = 0.6;
		friction = 2.8;
		layerMask = 1;
		boundaryWidth = 1.0;
//		priority = 1;
	}
	virtual void Move(Vector2D vec) = 0;
	//Update Shape Information
	virtual void Update() = 0;
	virtual int GetType() = 0;
	virtual bool IsPointInside(Vector2D p) const = 0;
	virtual void Redraw() = 0;
	virtual AABB GetAABB() = 0;
	Vector2D GetCentroidPosition();
	void ApplyTorque(double torque);
	void ApplyImpulse(Vector2D r, Vector2D p);
	void ApplyCorrectiveImpulse(Vector2D r, Vector2D p);
	void SetBoundaryWidth(double width);
	const Matrix3x3 &GetTransformToWorld() const;
	const Matrix3x3 &GetTransformToWorldInverse() const;
};

#endif