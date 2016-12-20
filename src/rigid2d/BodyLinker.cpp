#include "BodyLinker.h"
#include "Game.h"

void BodyLinker::SetGame(Game *game) {
	this->game = game;
	input = &game->input;
}

bool BodyLinker::KeyEvent(int key, int action) {
	if (key == 'Q') {
		active = action == GLFW_PRESS;
		return true;
	}
	if (!active) return false;
	/*
	if (key == GLFW_KEY_LSHIFT) {
		showGrid = action == GLFW_PRESS;
		return true;
	}
	*/
	return false;
}

bool BodyLinker::MouseButtonEvent(int button, int action) {
	if (showGrid) {
		double step = 1;
		//while (game->scoper.GetDistanceOnScr(Vector2D(0, 0, 1), Vector2D(0, step * 2, 1)) < 30) step *= 2;
		Vector2D &p = game->input.mouse.position;
		p.x = round(p.x / step) * step;
		p.y = round(p.y / step) * step;
	}
	if (!active) return false;
	Vector2D p = input->mouse.position;
	if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS) {
		game->physics.SelectObject(input->mouse.position, body[state], r[state]);
		if (body[state]) {
			state++;
			if (state == 2) {
				game->physics.AddConstantConstraint(new DistanceConstraint(body[0], r[0], body[1], r[1]));
				state = 0;
			}
		} 
	}
	return true;
}	

void BodyLinker::Redraw() {
		/*
	if (!active) return;
	RGB3f color = Colors::White;
	if (showGrid) {
		double step = 1;
		while (game->scoper.GetDistanceOnScr(Vector2D(0, 0, 1), Vector2D(0, step * 2, 1)) < 30) step *= 2;
		int x0, x1, y0, y1;
		x0 = (int)(game->scoper.GetLeft() / step) - 5;
		x1 = (int)(game->scoper.GetRight() / step) + 5;
		y0 = (int)(game->scoper.GetBottom() / step) - 5;
		y1 = (int)(game->scoper.GetTop() / step) + 5;
		for (int i = x0; i < x1; i++)
			for (int j = y0; j < y1; j++)
				graphics.DrawPoint(i * step, j * step, HSB3f(130.0f, 0.8f, 0.8f));
		Vector2D &p = game->input.mouse.position;
		p.x = round(p.x / step) * step;
		p.y = round(p.y / step) * step;
		//graphics.DrawPoint(p, HSB3f(0.0f, 0.7f, 0.8f), 3);
	}
	Vector2D p = input->mouse.position;
	if (state == 1) {
		Vector2D p0 = body[0]->GetTransformToWorld(r[0].GetPosition());
		//graphics.DrawLine(p0, p, RGB3f(0.0f, 1.0f, 0.0f), 5);
	}
		*/
}