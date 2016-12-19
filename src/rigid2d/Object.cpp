#include "Object.h"

//static const int Object::TYPE_WORLD_BOX = 1;
//static const int Object::TYPE_GEAR = 2;


void Polygon::UpdateCurrentInformation() {
	Angle angle = object->angle;
	Vector2D x = object->position.GetVector();
	for (int i = 0; i < nPoints; i++) {
		curPoints[i] = points[i].GetRotated(angle) + x;
	}
	for (int i = 0; i < nPoints; i++) {
//		curNormals[i] = (curPoints[loopNext(i, nPoints)] - curPoints[i]).GetInverseRotate().GetDirection();
		curNormals[i] = normals[i].GetRotated(angle);
	}

}