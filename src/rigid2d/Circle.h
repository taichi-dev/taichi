#pragma once

#include "Shape.h"

class Circle : public Shape {
	friend class Physics;
public:
	static const int ShapeType = 1;
private:
	double radius;
public:
	Circle() : Shape() {
		radius = 1.;
		density = 1.;
		Update();
	}
	void Update() {
		mass = density * sqr(radius) * pi;
		inertia = density * sqr(sqr(radius)) * pi / 2.;
	}
	void GetProjection(Vector2D normal, double &minI, double &maxI) {
		minI = maxI = GetCentroidPosition() * normal;
		minI -= radius; maxI += radius;
	}
	int GetType() {
		return ShapeType;
	}
	void Redraw() {
		//graphics.DrawCircle(GetTransformToWorld()(centroidPosition), radius, color);
		//graphics.DrawLine(GetTransformToWorld()(centroidPosition), GetTransformToWorld()(centroidPosition + Vector2D(radius, 0, 0)));
	}
	bool IsPointInside(Vector2D p) const {
		return sgn((GetTransformToWorldInverse()(p) - centroidPosition).GetLength2() - sqr(radius)) <= 0;
	}
	void Move(Vector2D vec) {
		centroidPosition += vec;
	}
	static Circle *GenerateCircle(double radius) {
		Circle *cir = new Circle();
		cir->radius = radius;
		cir->Update();
		return cir;
	}
	AABB GetAABB() {
		Vector2D worldCentroid = GetCentroidPosition();
		return AABB(worldCentroid.x - radius, worldCentroid.y - radius, worldCentroid.x + radius, worldCentroid.y + radius);
	}
};
