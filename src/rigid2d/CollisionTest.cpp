#include "Physics.h"

void Physics::TestCollision(Polygon *a, Polygon *b) {
    Vector2D n;
    Line edge; int edgeId;
    double minDepth = DBL_INF;
    int flg = 0;
    for (int i = 0; i < a->nPoints; i++) {
        double min0, max0, min1, max1;
        Vector2D normal = a->GetNormal(i);
        a->GetProjection(normal, min0, max0);
        b->GetProjection(normal, min1, max1);
        if (max0 < min1 || max1 < min0) return;
        double d = max0 - min1;
        if (min1 < max0 && (sgn(d - minDepth) < 0) || sgn(d - minDepth) == 0 && a->mass > b->mass) {
            flg = 0;
            minDepth = d;
            n = normal;
            edgeId = i;
            edge = a->GetEdge(i);
        }
    }
    for (int i = 0; i < b->nPoints; i++) {
        double min0, max0, min1, max1;
        Vector2D normal = b->GetNormal(i);
        b->GetProjection(normal, min0, max0);
        a->GetProjection(normal, min1, max1);
        if (max0 < min1 || max1 < min0) return;
        double d = max0 - min1;
        if (min1 < max0 && (sgn(d - minDepth) < 0) || sgn(d - minDepth) == 0 && b->mass > a->mass) {
            flg = 1;
            minDepth = d;
            n = normal;
            edgeId = i;
            edge = b->GetEdge(i);
        }
    }
    if (flg) swap(a, b);
    flg = -1;
    for (int i = 0; i < b->nPoints; i++) {
        if (n * (b->GetPoint(i) - edge.a) < 0 && n * (b->GetPoint((i + 1) % b->nPoints) - edge.a) < 0) {
            flg = i;
            break;
        }
    }
    if (flg != -1) {
        int i = edgeId, j = flg;
        pair<double, Vector2D> cons[4];
        Vector2D tan = n.GetRotate();
        Vector2D A = a->GetPoint(i),
            B = a->GetPoint((i + 1) % a->nPoints),
            C = b->GetPoint(j),
            D = b->GetPoint((j + 1) % b->nPoints);
        cons[0] = make_pair(tan * A, A);
        cons[1] = make_pair(tan * B, B);
        cons[2] = make_pair(tan * C, C);
        cons[3] = make_pair(tan * D, D);
        sort(cons, cons + 4);
        if (sgn(a->GetNormal(i) * b->GetNormal(j) + 1.0) != 0) {
            constraints.push_back(new Contact(a, b, cons[1].second, n, minDepth));
            constraints.push_back(new Contact(a, b, cons[2].second, n, minDepth));
        } else {
            constraints.push_back(new Contact(a, b, (cons[1].second + cons[2].second - Vector2D::Origin) * 0.5, n, minDepth));                
        }
    } else {
        Vector2D retP;
        double retDep = -DBL_INF;
        for (int i = 0; i < b->nPoints; i++) {
            Vector2D p = b->GetPoint(i);
            double depth = -(p - edge.a) * n;
            if (depth > retDep)
                retDep = depth, retP = p;
        }
        if (retDep > -eps)
            constraints.push_back(new Contact(a, b, retP, n, retDep));
    }
}

void Physics::TestCollision(Circle *a, Circle *b) {
    Vector2D n = (b->GetCentroidPosition() - a->GetCentroidPosition());
    double depth = a->radius + b->radius - n.GetLength();
    if (depth < 0) return;
    n.Normalize();
    constraints.push_back(new Contact(a, b, a->GetCentroidPosition() + n * a->radius, n, depth));
}

void Physics::TestCollision(Circle *a, Polygon *b) {
    Vector2D n, p;
    Line edge;
    double minDepth = DBL_INF;
    for (int i = 0; i < b->nPoints; i++) {
        double min0, max0, min1, max1;
        Vector2D normal = b->GetNormal(i);
        b->GetProjection(normal, min0, max0);
        a->GetProjection(normal, min1, max1);
        if (max0 < min1 || max1 < min0) return;
        Line tempEdge(b->GetPoint(i), b->GetPoint((i + 1) % b->nPoints));
        Vector2D tempP = tempEdge.a + (a->GetCentroidPosition() - tempEdge.a) * tempEdge.v.GetDirection() * tempEdge.v.GetDirection();
        if (tempEdge.IsInside(tempP) && max0 < max1 && min1 < max0 && max0 - min1 < minDepth) {
            minDepth = max0 - min1;
            n = normal;
            edge = Line(b->GetPoint(i), b->GetPoint((i + 1) % b->nPoints));
            p = tempP;
        }
    }
    if (minDepth < DBL_INF)
        constraints.push_back(new Contact(b, a, p, n, minDepth));
    for (int i = 0; i < b->nPoints; i++) {
        Vector2D p = b->GetTransformToWorld()(b->points[i]);
        if (a->IsPointInside(p)) {
            double depth = a->radius - (a->GetCentroidPosition() - p).GetLength();
            constraints.push_back(new Contact(a, b, p, (p - a->GetCentroidPosition()).GetDirection(), depth));
            return;
        }
    }
}

void Physics::TestCollision(Shape *a, Shape *b) {
    if (!(a->layerMask & b->layerMask)) return;
    if (a->object == b->object) return;
    int typeA = a->GetType(), typeB = b->GetType();
    if (typeA == Polygon::ShapeType && typeB == Polygon::ShapeType)
        return TestCollision((Polygon *)a, (Polygon *)b);
    if (typeA == Circle::ShapeType && typeB == Circle::ShapeType)
        return TestCollision((Circle *)a, (Circle *)b);
    if (typeA == Circle::ShapeType && typeB == Polygon::ShapeType)
        return TestCollision((Circle *)a, (Polygon *)b);
    if (typeA == Polygon::ShapeType && typeB == Circle::ShapeType)
        return TestCollision((Circle *)b, (Polygon *)a);
}

/*
void Physics::TestCollision(Object *a, Object *b) {
    if (!(a->layerMask & b->layerMask)) return;
    for (int i = 0; i < (int)a->shapes.size(); i++) {
        for (int j = 0; j < (int)b->shapes.size(); j++)
            TestCollision(a->shapes[i], b->shapes[j]);
    }
}
*/