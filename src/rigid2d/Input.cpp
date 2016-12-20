#include "Input.h"
#include "Constants.h"

void Input::Update() {
	int x, y;
	//glfwGetMousePos(&x, &y);
	// y = screenHeight - y;
	mouse.SetPos(x, y);
	/*
	for (int i = 0; i < 512; i++) keyboard.Set(i, glfwGetKey(i) == GLFW_PRESS);
	mouse.Set(GLFW_MOUSE_BUTTON_1, glfwGetMouseButton(GLFW_MOUSE_BUTTON_1) == GLFW_PRESS);
	mouse.Set(GLFW_MOUSE_BUTTON_2, glfwGetMouseButton(GLFW_MOUSE_BUTTON_2) == GLFW_PRESS);
	mouse.Set(GLFW_MOUSE_BUTTON_3, glfwGetMouseButton(GLFW_MOUSE_BUTTON_3) == GLFW_PRESS);
	*/
}
