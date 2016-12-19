#ifndef QUICK_INTERSECTION_TEST_H
#define QUICK_INTERSECTION_TEST_H

#include "Object.h"

#include <set>
/*
class IntersectionTest {
	vector<Shape *> shapes;
	int n;
	struct STree {
		set<int> S[4096];
		void Build(int t, int l, int r) {
			S[t].clear();
			if (l + 1 == r) return;
			int m = (l + r) / 2;
			Build(t * 2, l, m);
			Build(t * 2 + 1, m, r);
		}
		void Insert(int t, int l, int r, int ll, int rr, int key) {
			if (ll <= l && r <= rr) {
				S[t].insert(key);
				return;
			}
			int m = (l + r) / 2;
			if (rr <= m) return Insert(t * 2, l, m, ll, rr, key);
			if (m <= ll) return Insert(t * 2 + 1, m, r, ll, rr, key);
			Insert(t * 2, l, m, ll, rr, key); Insert(t * 2 + 1, m, r, ll, rr, key);
		}
		void Erase(int t, int l, int r, int ll, int rr, int key) {
			if (ll <= l && r <= rr) {
				S[t].erase(S[t].lower_bound(key));
				return;
			}
			int m = (l + r) / 2;
			if (rr <= m) return Erase(t * 2, l, m, ll, rr, key);
			if (m <= ll) return Erase(t * 2 + 1, m, r, ll, rr, key);
			Erase(t * 2, l, m, ll, rr, key); Erase(t * 2 + 1, m, r, ll, rr, key);
		}
		void Query(int t, int l, int r, int p, vector<int> &ret) {
			if (!S[t].empty())
				for (set<int>::iterator it = S[t].begin(); it != S[t].end(); it++) ret.push_back(*it);
			if (l + 1 == r) return;
			int m = (l + r) / 2;
			if (p < m) Query(t * 2, l, m, p, ret);
			else Query(t * 2 + 1, m, r, p, ret);
		}
	} stree;
	AABB aabbs[1024];
	struct Event {
		int x, y0, y1;
		int flg, key;
		Event() {}
		Event(int x, int y0, int y1, int flg, int key) : x(x), y0(y0), y1(y1), flg(flg), key(key) {}
	} events[2048];
	static bool cmp(const Event &a, const Event &b) {
		return a.x < b.x;
	}
	double skin;
public:
	IntersectionTest() {}
	void Init(vector<Object *> objects, double skin) {
		shapes.clear();
		for (int i = 0; i < (int)objects.size(); i++) {
			for (vector<Shape *>::iterator it = objects[i]->shapes.begin(); it != objects[i]->shapes.end(); it++) {
				if ((*it)->layerMask != 0) {
					shapes.push_back(*it);
				}
			}
		}
		this->skin = skin;
	}
	vector<pair<Shape *, Shape *> > GetResult() {
		
		n = (int)shapes.size();
		if (n <= 1) return vector<pair<Shape *, Shape *> >();
		vector<double> xList, yList;
		for (int i = 0; i < n; i++) {
			aabbs[i] = shapes[i]->GetAABB();
			aabbs[i].Enlarge(skin);
			xList.push_back(aabbs[i].x0);
			xList.push_back(aabbs[i].x1);
			yList.push_back(aabbs[i].y0);
			yList.push_back(aabbs[i].y1);
		}
		sort(xList.begin(), xList.end()); sort(yList.begin(), yList.end());
		for (int i = 0; i < n; i++) {
			aabbs[i].x0 = lower_bound(xList.begin(), xList.end(), aabbs[i].x0) - xList.begin();
			aabbs[i].x1 = lower_bound(xList.begin(), xList.end(), aabbs[i].x1) - xList.begin();
			aabbs[i].y0 = lower_bound(yList.begin(), yList.end(), aabbs[i].y0) - yList.begin();
			aabbs[i].y1 = lower_bound(yList.begin(), yList.end(), aabbs[i].y1) - yList.begin();
			events[i * 2] = Event((int)aabbs[i].x0, (int)aabbs[i].y0, (int)aabbs[i].y1, 1, i);
			events[i * 2 + 1] = Event((int)aabbs[i].x1, (int)aabbs[i].y0, (int)aabbs[i].y1, -1, i);
		}
		int treeN = yList.size();
		sort(events, events + n * 2, cmp);
		vector<pair<Shape *, Shape *> > ret;
		for (int i = 0; i < n * 2; i++) {
			if (events[i].flg == 1) {
				if (events[i].y0 == events[i].y1) {
					return ret;
				}			
				stree.Insert(1, 0, treeN, events[i].y0, events[i].y1, events[i].key);
			}
			vector<int> tmp;
			stree.Query(1, 0, treeN, events[i].y0, tmp);
			stree.Query(1, 0, treeN, events[i].y1, tmp);
			sort(tmp.begin(), tmp.end());
			tmp.resize(unique(tmp.begin(), tmp.end()) - tmp.begin());
			for (int j = 0; j < (int)tmp.size(); j++) if (tmp[j] != events[i].key)
				ret.push_back(make_pair(shapes[events[i].key], shapes[tmp[j]]));
			if (events[i].flg == -1) stree.Erase(1, 0, treeN, events[i].y0, events[i].y1, events[i].key);
		}
		for (int i = 0; i < (int)ret.size(); i++) 
			if (ret[i].first > ret[i].second) swap(ret[i].first, ret[i].second);
		sort(ret.begin(), ret.end());
		ret.resize(unique(ret.begin(), ret.end()) - ret.begin());
		return ret;
	}
};

*/

class IntersectionTest {
	vector<Shape *> shapes;
	int n;
	AABB aabbs[1024];
	double skin;
public:
	IntersectionTest() {}
	void Init(vector<Object *> objects, double skin) {
		shapes.clear();
		for (int i = 0; i < (int)objects.size(); i++) {
			for (vector<Shape *>::iterator it = objects[i]->shapes.begin(); it != objects[i]->shapes.end(); it++) {
				if ((*it)->layerMask != 0) {
					shapes.push_back(*it);
				}
			}
		}
		this->skin = skin;
	}
	vector<pair<Shape *, Shape *> > GetResult() {
		
		n = (int)shapes.size();
		if (n <= 1) return vector<pair<Shape *, Shape *> >();
		for (int i = 0; i < n; i++) {
			aabbs[i] = shapes[i]->GetAABB();
			aabbs[i].Enlarge(skin);
		}
		vector<pair<Shape *, Shape *> > ret;
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (aabbs[i].Overlap(aabbs[j])) {
					ret.push_back(make_pair(shapes[i], shapes[j]));
				}
			}
		}	
		return ret;
	}
};


#endif