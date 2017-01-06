#ifndef SHAPE_FACTORY_H
#define SHAPE_FACTORY_H

#include "Object.h"

class ShapeFactory {
private:
    static int type;
public:
    static Object *GenerateWorldBox() {
        Object *object = ShapeFactory::GenerateBoxObject(100000, 100000);
        object->shapes[0]->color = HSB3f(231.3f, 0.702f, 0.5f);
        object->shapes[0]->layerMask = 0;
        object->priority = -1;

        object->SetFixed(true);
        return object;
    }
    static Polygon *GeneratePolygon(int nPoints, double r) {
        Polygon *pol = Polygon::GeneratePolygon(nPoints, r);
        return pol;
    }
    static Object *GeneratePolygonObject(int nPoints, double r) {
        return new Object(Polygon::GeneratePolygon(nPoints, r));
    }

    static Polygon *GeneratePolygon(vector<Vector2D> points) {
        return new Polygon(points);
    }
    
    static Object *GeneratePolygonObject(vector<Vector2D> points) {
        vector<vector<Vector2D> > polygons = CutIntoConvex(points);
        vector<Shape *> shapes;
        for (int i = 0; i < (int)polygons.size(); i++) {
            shapes.push_back(GeneratePolygon(polygons[i]));
        }
        Object * object = new Object(shapes);
        object->SetColor(Colors::RandomBrightColor());
        return object;
    }

    static Circle *GenerateCircle(double r) {
        return Circle::GenerateCircle(r);
    }

    static Object *GenerateCircleObject(double r) {
        return new Object(Circle::GenerateCircle(r));
    }

    static Polygon *GenerateBox(double w, double h) {
        return Polygon::GenerateBox(w, h);
    }

    static Object *GenerateBoxObject(double w, double h) {
        return new Object(Polygon::GenerateBox(w, h));
    }

    static Object *GenerateDualBoxObject(double w, double h) {
        vector<Shape *> shapes;
        shapes.push_back(Polygon::GenerateBox(w, h));
        shapes.push_back(Polygon::GenerateBox(w, h));
        shapes[0]->Move(Vector2D(-w * 2, 0, 0));
        Object *object = new Object(shapes);
        return object;
    }
    static Object *GenerateGearObject(Vector2D center, double r, double rad) {
        return GeneratePolygonObject(Polygon::GenerateGearPoints(center, r, rad));
    }
    static Object *Generate() {
        switch(type) {
        case 1:
            return GenerateDualBoxObject(30, 30);
        case 2:
            return GenerateBoxObject(200, 30);
        case 3:    
        case 4:    
        case 5:    
        case 6:    
        case 7:    
        case 8:    
        case 9:    
            return GeneratePolygonObject(type, 50);
        default:
            return GenerateCircleObject(50);
        }
    }
    static bool KeyEvent(int key, int action) {
        if (action == GLFW_PRESS) {
            if ('0' <= key && key <= '9') {
                type = key - '0';
                return true;
            }
        }
        return false;
    }
    static void Init() {
        type = 4;
    }
    static vector<Vector2D> GetBoxPoints(Vector2D a, Vector2D b) {
        if (a.x > b.x) swap(a.x, b.x);
        if (a.y > b.y) swap(a.y, b.y);
        double w = b.x - a.x, h = b.y - a.y;
        vector<Vector2D> points;
        points.push_back(Vector2D(a.x, a.y, 1));
        points.push_back(Vector2D(a.x + w, a.y, 1));
        points.push_back(Vector2D(a.x + w, a.y + h, 1));
        points.push_back(Vector2D(a.x, a.y + h, 1));
        return points;
    }
};


#endif