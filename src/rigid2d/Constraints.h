#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include "Object.h"
#include "MemoryAllocator.h"

class Force {
public:
    virtual bool Linking(Object *object) = 0;
    virtual void Apply(double T) = 0;
    virtual void Redraw() {}
};

class Spring : public Force {
public :
    double strength, L;
    RGB3f color;
    double D; 
    double width;
    double damping;
    Object *objectA, *objectB;
    Vector2D r0, r1;
    Spring(Object *objectA, Vector2D r0, Object *objectB, Vector2D r1, double strength, double L = -1.0) :
        objectA(objectA), r0(r0), objectB(objectB), r1(r1), strength(strength), L(L) {
        color = RGB3f::RandomBrightColor();
        if (L < 0) this->L = (objectA->transformToWorld(r0) - objectB->transformToWorld(r1)).GetLength();
        damping = 150;
    }
    void Apply(double T) {
        Vector2D p = objectA->transformToWorld(r0);
        Vector2D q = objectB->transformToWorld(r1);
        Vector2D n = p - q;
        double l = n.GetLength();
        n.Normalize();
        Vector2D v = objectA->GetPointVelocity(p) - objectB->GetPointVelocity(q);
        double v0 = n * v;
        Vector2D impulse = (-damping * v0 * T + T * strength * (L - l)) * n.GetDirection();
        objectA->ApplyImpulse(p, impulse);
        objectB->ApplyImpulse(q, -impulse);
    }
    void Redraw() {
        Vector2D p = objectA->GetTransformToWorld()(r0);
        Vector2D q = objectB->GetTransformToWorld()(r1);
        
        width = max(1.0, L / 100.0);
        int n = 16;
        Vector2D t = (p - q).GetDirection().GetRotate() * L / 12;
        vector<Vector2D> points;
        points.push_back(p);
        for (int i = 0; i < n; i++) {
            t = -t;
            points.push_back(p + (q - p) * (i + 0.5)/ n + t);
        }
        points.push_back(q);
        //graphics.DrawLines(points, color, width);
        //graphics.DrawPoint(p.x, p.y, color, width * 4);
        //graphics.DrawPoint(q.x, q.y, color, width * 4);
    }
    bool Linking(Object *object) {
        return objectA == object || objectB == object;
    }
};

class Constraint {
public:
    int type;
    Constraint() {type = 0;}
    virtual void ProcessVelocity() = 0;
    virtual void ProcessPosition() = 0;
    virtual Constraint *Copy() = 0;
    virtual bool Linking(Object *object) = 0;
    virtual void Redraw() {}
    virtual ~Constraint() {};
};


class Contact : public Constraint{
public:
#define TYPE_CONTACT 1
    Shape *shapeA, *shapeB;
    Object *objectA, *objectB;
    Vector2D p, n;
    double restitution, friction;
    double depth;
    Contact(Shape *a, Shape *b, Vector2D p, Vector2D n, double depth) : shapeA(a), objectA(a->object), shapeB(b), objectB(b->object), p(p), n(n), depth(depth) {
        restitution = sqrt(a->restitution * b->restitution);
        friction = sqrt(a->friction * b->friction);
        type = 1;
    }
    
    void PreprocessVelocity() {
        Vector2D v10 = objectB->GetPointVelocity(p) - objectA->GetPointVelocity(p);
        Vector2D r0 = p - objectA->position, r1 = p - objectB->position;
        double v0 = -n * v10;

        Vector2D tao = n.GetRotate();
        double cRestitution;
        if (v0 > 10) cRestitution = restitution;
        else cRestitution = 0.0;
        double J = ((1 + cRestitution) * v0) /
            (objectA->invMass + objectB->invMass + sqr(r0 % n) * objectA->invInertia + sqr(r1 % n) * objectB->invInertia);
        if (J < 0) return;
        Vector2D impulse = J * n;
        objectA->ApplyImpulse(p, -impulse);
        objectB->ApplyImpulse(p, impulse);
        if (settings.frictionSwitch) {
            v10 = objectB->GetPointVelocity(p) - objectA->GetPointVelocity(p);
            double j = -(v10 * tao) / (objectA->invMass + objectB->invMass + sqr(r0 % tao) * objectA->invInertia + sqr(r1 % tao) * objectB->invInertia);
            j = max(min(j, friction * J), -friction * J);
            Vector2D fImpulse = j * tao;
            objectA->ApplyImpulse(p, -fImpulse);
            objectB->ApplyImpulse(p, fImpulse);
        }
    
    }

    void ProcessVelocity() {
        Vector2D v10 = objectB->GetPointVelocity(p) - objectA->GetPointVelocity(p);
        Vector2D r0 = p - objectA->position, r1 = p - objectB->position;
        double v0 = -n * v10;

        Vector2D tao = n.GetRotate();
        double cRestitution;
        if (v0 > 10) cRestitution = restitution;
        else cRestitution = 0.0;
        double J = ((1 + cRestitution) * v0) /
            (objectA->invMass + objectB->invMass + sqr(r0 % n) * objectA->invInertia + sqr(r1 % n) * objectB->invInertia);
        if (J < 0) return;
        Vector2D impulse = J * n;
        objectA->ApplyImpulse(p, -impulse);
        objectB->ApplyImpulse(p, impulse);
        if (settings.frictionSwitch) {
            v10 = objectB->GetPointVelocity(p) - objectA->GetPointVelocity(p);
            double j = -(v10 * tao) / (objectA->invMass + objectB->invMass + sqr(r0 % tao) * objectA->invInertia + sqr(r1 % tao) * objectB->invInertia);
            j = max(min(j, friction * J), -friction * J);
            Vector2D fImpulse = j * tao;
            objectA->ApplyImpulse(p, -fImpulse);
            objectB->ApplyImpulse(p, fImpulse);
        }
    }

    void ProcessPosition() {
        Vector2D r0 = p - objectA->position, r1 = p - objectB->position;
        double correctiveJ = (0.5 * max(0.0, depth - 0.1) / timeInterval) / 
            (objectA->invMass + objectB->invMass + sqr(r0 % n) * objectA->invInertia + sqr(r1 % n) * objectB->invInertia);
        if (correctiveJ > 0) {
            objectA->ApplyCorrectiveImpulse(p, -correctiveJ * n, true);
            objectB->ApplyCorrectiveImpulse(p, correctiveJ * n, true);
        }
    }
    Constraint *Copy() {
        return new Contact(*this);
    }
    bool Linking(Object *object) {
        if (object == objectA || object == objectB) return true;
        return false;
    }

    void *operator new(size_t _);

    void operator delete(void *p, size_t _);


};


class DistanceConstraint : public Constraint {
private:
    double L;
    Object *objectA, *objectB;
    Vector2D r0, r1;
public:
    RGB3f color;
    DistanceConstraint(Object *objectA, Vector2D r0, Object *objectB, Vector2D r1) :
    L((objectA->transformToWorld(r0.GetPosition()) - objectB->transformToWorld(r1.GetPosition())).GetLength()),
        objectA(objectA), r0(r0), objectB(objectB), r1(r1)
    {
        color = RGB3f::RandomBrightColor();
    }
    void ProcessVelocity() {
        Vector2D p = objectA->transformToWorld(r0.GetPosition());
        Vector2D q = objectB->transformToWorld(r1.GetPosition());
        Vector2D n = q - p;
        double l = n.GetLength();
        if (abs(l - L) < 0.1) return;
        n.Normalize();
        double J;
        double v = n * (objectB->GetPointVelocity(q) - objectA->GetPointVelocity(p));
        J = v / (objectA->invMass + sqr(objectA->transformToWorld(r0) % n) * objectA->invInertia 
            + objectB->invMass + sqr(objectB->transformToWorld(r1) % n) * objectB->invInertia);
        objectA->ApplyImpulse(p, n * J);
        objectB->ApplyImpulse(q, n * -J);
    }
    void ProcessPosition() {
        Vector2D p = objectA->position + objectA->transformToWorld(r0);
        Vector2D q = objectB->position + objectB->transformToWorld(r1);
        Vector2D n = q - p;
        double l = n.GetLength();
        if (abs(l - L) < 0.1) return;
        n.Normalize();
        double J;
        J = (0.6 / timeInterval * (l - L)) / (objectA->invMass + sqr(objectA->transformToWorld(r0) % n) * objectA->invInertia + objectB->invMass + sqr(objectB->transformToWorld(r1) % n) * objectB->invInertia);
        objectA->ApplyCorrectiveImpulse(p, n * J);
        objectB->ApplyCorrectiveImpulse(q, n * -J);
    }
    Constraint *Copy() {
        return new DistanceConstraint(*this);
    }
    void Redraw() {
        Vector2D p = objectA->position + objectA->transformToWorld(r0);
        Vector2D q = objectB->position + objectB->transformToWorld(r1);
        double width = max(1.0, L / 100.0);
        //graphics.DrawLine(p.x, p.y, q.x, q.y, color, width);
        //graphics.DrawPoint(p.x, p.y, color, width * 4);
        //graphics.DrawPoint(q.x, q.y, color, width * 4);
    }
    bool Linking(Object *object) {
        if (object == objectA) return true;
        if (object == objectB) return true;
        return false;
    }
};

    

#endif