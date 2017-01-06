#ifndef BODY_LINKTER_H
#define BODY_LINKTER_H

#include "Polygon.h"
#include "Geometry.h"
#include "Input.h"

class Game;

class BodyLinker {
private:
    int state;
    Object *body[2];
    Vector2D r[2];
    bool active;
    Input *input;
    Game *game;
    bool showGrid;
public:
    vector<Vector2D> points;
    bool IsActive() {
        return active;
    }
    void SetGame(Game *game);
    BodyLinker() {
        active = false;
        state = 0;
        showGrid = false;
        memset(body, 0, sizeof(body));
    }
    bool KeyEvent(int key, int action);
    bool MouseButtonEvent(int button, int action);
    void Redraw();
    
};


#endif