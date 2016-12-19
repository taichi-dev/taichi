#include "Designer.h"
#include "Game.h"
#include "Scoper.h"

void Designer::SetGame(Game *game) {
	this->game = game;
	input = &game->input;
}
bool Designer::KeyEvent(int key, int action) {
	if (key == GLFW_KEY_LCTRL) {
		active = action == GLFW_PRESS;
		return true;
	}
	if (key == GLFW_KEY_LSHIFT) {
		showGrid = action == GLFW_PRESS;
		return true;
	}
	if (!active) return false;
	if (active == GLFW_RELEASE) return false;
	if ('0' <= key && key <= '9') {
		type = key - '0';
		points.clear();
		return true;
	}
	return false;
}
bool Designer::MouseButtonEvent(int button, int action) {
	if (showGrid) {
		double step = 1;
		while (game->scoper.GetDistanceOnScr(Vector2D(0, 0, 1), Vector2D(0, step * 2, 1)) < 30) step *= 2;
		Vector2D &p = game->input.mouse.position;
		p.x = round(p.x / step) * step;
		p.y = round(p.y / step) * step;
	}
	if (!active) return false;
	if (button == GLFW_MOUSE_BUTTON_2)
		return false;
	Vector2D p = input->mouse.position;
	if (button == GLFW_MOUSE_BUTTON_1) {
		if (action == GLFW_PRESS) {
			switch (type) {
				case 0:
				case 2:
				case 3:
				case 4:
				case 5:
				case 6:
				case 7:
				case 8:
				case 9:
					center = p;
					break;
				case 1:
					if (points.size() && game->scoper.GetDistanceOnScr(p, points[0]) <= 5) {
						Object *object = ShapeFactory::GeneratePolygonObject(points);
						game->physics.AddObject(object);
						points.clear();
						return true;
					}
					points.push_back(p);
				break;
			
			}
			return true;
		} else {
			Object *object = NULL;
			Vector2D p = input->mouse.position;
			Vector2D r = p - center;
			int n;
			switch (type) {
				case 0:
					if (r.GetLength2() == 0.0) return true;
					object = ShapeFactory::GenerateCircleObject(r.GetLength());
					object->SetPosition(center);
					break;
				case 2:
					if (p.x == center.x) return true;
					if (p.y == center.y) return true;
					object = ShapeFactory::GeneratePolygonObject(ShapeFactory::GetBoxPoints(center, p));
					object->SetPosition((center + p - Vector2D::Origin) / 2);
					break;
				case 1:
					break;
				case 4:
				case 5:
				case 6:
				case 7:
				case 8:
				case 9:
					if (r.GetLength() == 0) return true;
					points.clear();
					n = type;
					for (int i = 0; i < n; i++) {
						points.push_back(center + r);
						r = r.GetRotate(2 * pi / n);
					}
					object = ShapeFactory::GeneratePolygonObject(points);
					object->SetPosition(center);
					break;
				case 3:
					if (r.GetLength() == 0) return true;
					object = ShapeFactory::GenerateGearObject(center, r.GetLength(), Polygon::GearD / r.GetLength());
					object->SetPosition(center);
					game->physics.AddConstantConstraint(new DistanceConstraint(game->physics.GetWorldBox(), center.GetVector(), object, Vector2D(0, 0, 0)));
					break;
			}
			if (object != NULL)
				game->physics.AddObject(object);

		}
	}
	return true;
}	

void Designer::Redraw() {
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
		graphics.DrawPoint(p, HSB3f(0.0f, 0.7f, 0.8f), 3);
	}
	Vector2D p = input->mouse.position;
	switch (type) {
		case 0:
			if (input->mouse.IsPressed(GLFW_MOUSE_BUTTON_1)) {
				graphics.DrawCircle(center.x, center.y, (p - center).GetLength(), RGB3f(1.0f, 1.0f, 1.0f), false);
			}
			break;
		case 1:
			if ((int)points.size()) {
				if (game->scoper.GetDistanceOnScr(p, points[0]) <= 8)
					color = RGB3f(0.0f, 0.8f, 0.0f);
				graphics.DrawLine(points.back().x, points.back().y, p.x, p.y, color, 2.0);
			}
			for (int i = 0; i < (int)points.size(); i++)
				graphics.DrawPoint(points[i].x, points[i].y, color);
			graphics.DrawLines(points, color, 2.0);
			break;
		case 2:
			if (input->mouse.IsPressed(GLFW_MOUSE_BUTTON_1)) {
				graphics.DrawLinesLoop(ShapeFactory::GetBoxPoints(center, p), Colors::White, 2.0);
			}
			break;
		case 4:
		case 5:
		case 6:
		case 7:
		case 8:
		case 9:
			if (input->mouse.IsPressed(GLFW_MOUSE_BUTTON_1)) {
				points.clear();
				Vector2D r = p - center;
				int n = type;
				for (int i = 0; i < n; i++) {
					points.push_back(center + r);
					r = r.GetRotate(2 * pi / n);
				}
				graphics.DrawLinesLoop(points, Colors::White, 2.0);
			}
			break;
		case 3:
			if (input->mouse.IsPressed(GLFW_MOUSE_BUTTON_1)) {
				Vector2D r = p - center;
				vector<Vector2D> points = Polygon::GenerateGearPoints(center, r.GetLength(), Polygon::GearD / r.GetLength());
				graphics.DrawLinesLoop(points, Colors::White, 2.0);
			}
			break;
	}
}