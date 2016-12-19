#ifndef DESIGNER_H
#define DESIGNER_H

#include "Polygon.h"
#include "Geometry.h"
#include "Input.h"

class Game;

class Designer {
private:
	int type;
	bool active, buttonPressed;
	bool showGrid;
	Input *input;
	Game *game;
	Vector2D center;
	
public:
	vector<Vector2D> points;
	bool IsActive() {
		return active;
	}
	void SetGame(Game *game);
	Designer() {
		type = 0;
		active = false;
	}
	bool KeyEvent(int key, int action);
	bool MouseButtonEvent(int button, int action);
	void Redraw();
	
};


#endif