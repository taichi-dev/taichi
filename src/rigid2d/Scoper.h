#ifndef SCOPER_H
#define SCOPER_H

#include "Settings.h"
#include "Input.h"
#include "Geometry.h"

class Game;

class Scoper {
private:
	double leftBound, rightBound, topBound, bottomBound;
	double totWidth, totHeight;
	double bottom, top, left, right;
	Game *game;
	Mouse *mouse;
	void Apply() {
		double xDelta = 0.0, yDelta = 0.0;
		if (left < leftBound) xDelta = leftBound - left;
		if (right > rightBound) xDelta = rightBound - right;
		if (bottom < bottomBound) yDelta = bottomBound - bottom;
		if (top > topBound) yDelta = topBound - top;
		left += xDelta, right += xDelta;
		top += yDelta, bottom += yDelta;
		glLoadIdentity();
		glOrtho(left, right, bottom, top, -1.0f, 1.0f);
		Matrix3x3 mat;
		mat[2][2] = 1.0; mat[0][2] = left; mat[1][2] = bottom;
		mat[0][0] = (right - left) / screenWidth;
		mat[1][1] = (top - bottom) / screenHeight;
		mouse->transform = mat;
	}
public:
	double GetLeft() {
		return left;
	}
	double GetRight() {
		return right;
	}
	double GetBottom() {
		return bottom;
	}
	double GetTop() {
		return top;
	}
	double GetScale() {
		return (right - left) / screenWidth;
	}
	double GetDistanceOnScr(Vector2D a, Vector2D b) {
		return (a - b).GetLength() / GetScale();
	}
	void Move(Vector2D vec) {
		left += vec.x, right += vec.x;
		bottom += vec.y; top += vec.y;
		Apply();
	}
	Scoper() {
		left = 0; right = screenWidth;
		bottom = 0; top = screenHeight;
		leftBound = -screenWidth * 3, rightBound = screenWidth * 3;
		bottomBound = -screenHeight * 3, topBound = screenHeight * 3;
		totWidth = rightBound - leftBound;
		totHeight = topBound - bottomBound;
	}
	void SetScope(double x, double y, double w, double h) {
		if (w > totWidth) w = totWidth;
		if (w < 1.0) w = 1.0;
		h = w / screenWidth * screenHeight;
		left = x - w / 2; right = x + w / 2;
		bottom = y - h / 2; top = y + h / 2;
		Apply();
	}
	void SetGame(Game *game);
	bool MouseWheelEvent(int delta);
	
};


#endif