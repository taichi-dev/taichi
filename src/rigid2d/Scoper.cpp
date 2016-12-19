#include "Scoper.h"
#include "Game.h"

void Scoper::SetGame(Game *game) {
	this->game = game;
	this->mouse = &game->input.mouse;
}

bool Scoper::MouseWheelEvent(int delta) {
	if (!game) return false;
	double w = right - left, h = top - bottom;
	double x = mouse->position.x, y = mouse->position.y;
	double rate = 1.0;
	rate = pow(0.85, delta);
	SetScope(x - (x - left - w / 2) * rate, y - (y - bottom - h / 2) * rate, w * rate, h * rate);
	return true;
}
