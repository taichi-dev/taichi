// My old, Olympiad for Informatics version of Kd-tree ... just for a replacement of ANN

#include <taichi/math/linalg.h>

/*
using namespace std;
#define PL printf("%d\n", __LINE__);
#define MM(a, b) memset(a, b, sizeof(a));
#define For(i, a) for (register int i = 0; i < (a); i++)
#define Foru(i, a, b) for (register int i = (a); i < (b); i++)
#define foru(i, a, b) for (register int i = (a); i <= (b); i++)
#define ford(i, a, b) for (register int i = (a); i >= (b); i--)
#define fore(i, a, b) for (register __typeof(a) i = (a); (i) != (b); (i)++)
#define inf 1000000000
#define linf 10000000000000000LL
#define pb push_back
#define mp make_pair
#define eps 1e-7
#define Walk(u) for (elist l = de[u]; l; l = l->next)
#define bgn begin
#define fi first
#define se second
#define ite iterator
#define All(x) (x).bgn(), (x).end()
#define sz(x) ((int)x.size())
#define pq priority_queue
typedef long long LL;
typedef pair<int, int> pii;
typedef vector<int> vi;

class KDTree2D {
    typedef Vector2 point;

    struct KDNode {
        real x, y;
        real x0, y0, x1, y1;
        KDNode *ch[2];
        KDNode(point p) {
            x = x0 = x1 = p.x; y = y0 = y1 = p.y;
            ch[0] = ch[1] = 0;
        }
    };
    typedef KDNode *kd;

    kd root;

#define down(a, b) {if ((a) > (b)) (a) = (b);}
#define up(a, b) {if ((a) < (b)) (a) = (b);}

    inline void Update(kd &t) {
        if (t->ch[0]) {
            down(t->x0, t->ch[0]->x0);
            down(t->y0, t->ch[0]->y0);
            up(t->x1, t->ch[0]->x1);
            up(t->y1, t->ch[0]->y1);
        }
        if (t->ch[1]) {
            down(t->x0, t->ch[1]->x0);
            down(t->y0, t->ch[1]->y0);
            up(t->x1, t->ch[1]->x1);
            up(t->y1, t->ch[1]->y1);
        }
    }

    inline void Insert(kd &t, point p, int depth) {
        if (t == 0) {
            t = new KDNode(p);
            return;
        }
        if (depth & 1) Insert(t->ch[t->x < p.x], p, depth + 1);
        else Insert(t->ch[t->y < p.y], p, depth + 1);
        Update(t);
    }

    inline void Insert(point p) {
        Insert(root, p, 0);
    }

    real dist(real x0, real y0, real x1, real y1) {
        real dist = hypot(x0 - x1, y0 - y1);
        if (dist == 0.0f) {
            return std::numeric_limits<real>::infinity();
        } else {
            return dist;
        }
    }

    real MinDist(kd t, point p) {
        real res = 0;
        if (p.x < t->x0) res += sqr(t->x0 - p.x);
        else if (t->x1 < p.x) res += sqr(p.x - t->x1);
        if (p.y < t->y0) res += sqr(t->y0 - p.y);
        else if (t->y1 < p.y) res += sqr(p.y - t->y1);
        return std::sqrt(res);
    }

    void Query(kd t, point p, real &opt, int depth) {
        opt = min(opt, dist(p.x, p.y, t->x, t->y));
        int c = 0;
        if (depth & 1) c = t->x < p.x;
        else c = t->y < p.y;
        if (t->ch[c] && MinDist(t->ch[c], p) < opt) Query(t->ch[c], p, opt, depth + 1);
        if (t->ch[!c] && MinDist(t->ch[!c], p) < opt) Query(t->ch[!c], p, opt, depth + 1);
    }

    real Query(point p) {
        real opt = std::numeric_limits<real>::infinity();
        Query(root, p, opt, 0);
        return opt;
    }

    bool cmpX(point a, point b) {return a.x < b.x;}

    bool cmpY(point a, point b) {return a.y < b.y;}

    void Build(kd &t, int bgn, int end, int depth) {
        if (bgn == end) return;
        if (bgn + 1 == end) {
            t = new KDNode(points[bgn]);
            return;
        }
        int mid = (bgn + end) / 2;
        nth_element(points + bgn, points + mid, points + end, (depth & 1) ? cmpX : cmpY);
        t = new KDNode(points[mid]);
        Build(t->ch[0], bgn, mid, depth + 1);
        Build(t->ch[1], mid + 1, end, depth + 1);
        Update(t);
    }

    KDTree2D() {

    }

    KDTree2D(const std::vector<Vector2>& data_points) {
        root = 0;
        this->data_points = data_points;
        Build(root, 0, n, 0);
            if (t == 1) Insert(p);
            else Query(p);
        }
    }
};
*/
