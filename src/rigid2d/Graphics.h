#ifndef GRAPHICS_H
#define GRAPHICS_H

#include "Constants.h"
#include "Geometry.h"
#include "Color.h"
//#include <GL/glfw.h>
//#include <AntTweakBar.h>
#define GLFW_DLL
#include <algorithm>
using namespace std;

void GLFWCALL WindowSizeCB(int width, int height);

class Graphics {
public:
	double a;
	int Initialize() {
		///glfwInit();
		//glfwOpenWindow(screenWidth, screenHeight, 8, 8, 8, 8, 32 , 0 , GLFW_WINDOW);
		//glfwSetWindowTitle("Physics Engine" );

		glClearColor(0.0f , 0.0f , 0.0f , 0.0f );      // ��ɫ���� 
		glColor3f(1.0f , 1.0f , 1.0f );
		glShadeModel(GL_FLAT);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, screenWidth, 0, screenHeight, -1.0f, 1.0f);
		glEnable(GL_POINT_SMOOTH);  
		glEnable(GL_LINE_SMOOTH);
		glEnable(GL_POLYGON_SMOOTH);
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST); // Make round points, not square points  
		glHint(GL_POINT_SMOOTH_HINT, GL_NICEST); // Make round points, not square points  
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);  // Antialias the lines  
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

		//glfwEnable(GLFW_MOUSE_CURSOR);
	    //glfwEnable(GLFW_KEY_REPEAT);
	

		//TwInit(TW_OPENGL, NULL);
		//TwWindowSize(screenWidth, screenHeight);

	    //glfwSetWindowSizeCallback(WindowSizeCB);

		//glfwSetCharCallback((GLFWcharfun)TwEventCharGLFW);

		return 0;	
	}
	void BeginRendering() {
		RGB3f color = HSB3f(231.3f, 0.702f, 0.5f);
		glClearColor(color.r, color.g, color.b, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);
	}
	void EndRendering() {
		//TwDraw();
		//glfwSwapBuffers();
    }

	void DrawLine(double x0, double y0, double x1, double y1, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), double width = 1.0) {
		glLineWidth((float)width);
		glColor3f(color.r, color.g, color.b);
		glBegin(GL_LINES);
			glVertex2f((float)x0, (float)y0);
			glVertex2f((float)x1, (float)y1);
		glEnd();
		glPointSize((float)width);
		glBegin(GL_POINTS);
			glVertex2f((float)x0, (float)y0);
			glVertex2f((float)x1, (float)y1);
		glEnd();
	}
	void DrawLine(Vector2D p, Vector2D q, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), double width = 1.0) {
		double x0 = p.x, y0 = p.y;
		double x1 = q.x, y1 = q.y;
		glLineWidth((float)width);
		glColor3f(color.r, color.g, color.b);
		glBegin(GL_LINES);
			glVertex2f((float)x0, (float)y0);
			glVertex2f((float)x1, (float)y1);
		glEnd();
		glPointSize((float)width);
		glBegin(GL_POINTS);
			glVertex2f((float)x0, (float)y0);
			glVertex2f((float)x1, (float)y1);
		glEnd();
	}
	void DrawLines(vector<Vector2D> points, RGB3f color, double width) {
		for (int i = 0; i < (int)points.size() - 1; i++)
			DrawLine(points[i].x, points[i].y, points[i + 1].x, points[i + 1].y, color, width);
	}
	void DrawLinesLoop(vector<Vector2D> points, RGB3f color, double width) {
		int n = (int)points.size();
		for (int i = 0; i < n; i++)
			DrawLine(points[i].x, points[i].y, points[(i + 1) % n].x, points[(i + 1) % n].y, color, width);
	}
	void DrawPoint(double x, double y, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), double width = 1.0) {
		glColor3f(color.r, color.g, color.b);
		glPointSize((float)width);
		glBegin(GL_POINTS);
			glVertex2f((float)x, (float)y);
		glEnd();
	}
	void DrawPoint(Vector2D p, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), double width = 1.0) {
		glColor3f(color.r, color.g, color.b);
		glPointSize((float)width);
		glBegin(GL_POINTS);
			glVertex2f((float)p.x, (float)p.y);
		glEnd();
	}
	void DrawCircle(double x, double y, double r, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), bool fill = true) {
		int n = 64;
		if (fill) {
			glColor3f(color.r, color.g, color.b);
			glBegin(GL_TRIANGLE_FAN);
			for (int i = 0; i < n; i++)
				glVertex2f((float)(x + cos(2 * pi / n * i) * r), float(y + sin(2 * pi / n * i) * r));
			glEnd();
		}
		glLineWidth(1.0f);
		glColor3f(1.0f, 1.0f, 1.0f);
		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < n; i++)
			glVertex2f((float)(x + cos(2 * pi / n * i) * r), float(y + sin(2 * pi / n * i) * r));
		glEnd();		
	}
	void DrawCircle(Vector2D p, double r, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), bool fill = true) {
		DrawCircle(p.x, p.y, r, color);
	}
	inline void DrawPolygon(const Vector2D *begin, const Vector2D *end, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f)) {
		DrawPolygon(vector<Vector2D>(begin, end), color);
	}
	inline void DrawPolygon(const vector<Vector2D> &points, RGB3f color = RGB3f(1.0f, 1.0f, 1.0f), double bound = 1.0) {
		glColor3f(color.r, color.g, color.b);
		glLineWidth(1.0f);
		glBegin(GL_TRIANGLE_FAN);
		for (int i = 0; i < (int)points.size(); i++)
			glVertex2f((float)points[i].x, (float)points[i].y);
		glEnd();
		if (bound > 0) {
			glLineWidth((float)bound);
			glPointSize(1.0f);
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_LINE_LOOP);
			for (int i = 0; i < (int)points.size(); i++)
				glVertex2f((float)points[i].x, (float)points[i].y);
			glEnd();
			glBegin(GL_POINTS);
			for (int i = 0; i < (int)points.size(); i++)
				glVertex2f((float)points[i].x, (float)points[i].y);
			glEnd();
		}
	}
};

extern Graphics graphics;

#endif
