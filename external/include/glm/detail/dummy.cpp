///////////////////////////////////////////////////////////////////////////////////
/// OpenGL Mathematics (glm.g-truc.net)
///
/// Copyright (c) 2005 - 2015 G-Truc Creation (www.g-truc.net)
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// Restrictions:
///		By making use of the Software for military purposes, you choose to make
///		a Bunny unhappy.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.
///
/// @ref core
/// @file glm/core/dummy.cpp
/// @date 2011-01-19 / 2011-06-15
/// @author Christophe Riccio
///
/// GLM is a header only library. There is nothing to compile. 
/// dummy.cpp exist only a wordaround for CMake file.
///////////////////////////////////////////////////////////////////////////////////

#define GLM_MESSAGES
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <limits>

struct material
{
	glm::vec4 emission; // Ecm
	glm::vec4 ambient; // Acm
	glm::vec4 diffuse; // Dcm
	glm::vec4 specular; // Scm
	float shininess; // Srm
};

struct light
{
	glm::vec4 ambient; // Acli
	glm::vec4 diffuse; // Dcli
	glm::vec4 specular; // Scli
	glm::vec4 position; // Ppli
	glm::vec4 halfVector; // Derived: Hi
	glm::vec3 spotDirection; // Sdli
	float spotExponent; // Srli
	float spotCutoff; // Crli
	// (range: [0.0,90.0], 180.0)
	float spotCosCutoff; // Derived: cos(Crli)
	// (range: [1.0,0.0],-1.0)
	float constantAttenuation; // K0
	float linearAttenuation; // K1
	float quadraticAttenuation;// K2
};


// Sample 1
#include <glm/vec3.hpp>// glm::vec3
#include <glm/geometric.hpp>// glm::cross, glm::normalize

glm::vec3 computeNormal
(
	glm::vec3 const & a,
	glm::vec3 const & b,
	glm::vec3 const & c
)
{
	return glm::normalize(glm::cross(c - a, b - a));
}

typedef unsigned int GLuint;
#define GL_FALSE 0
void glUniformMatrix4fv(GLuint, int, int, float*){}

// Sample 2
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4, glm::ivec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/gtc/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale, glm::perspective
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr
void func(GLuint LocationMVP, float Translate, glm::vec2 const & Rotate)
{
	glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.f);
	glm::mat4 ViewTranslate = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -Translate));
	glm::mat4 ViewRotateX = glm::rotate(ViewTranslate, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
	glm::mat4 View = glm::rotate(ViewRotateX, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
	glm::mat4 MVP = Projection * View * Model;
	glUniformMatrix4fv(LocationMVP, 1, GL_FALSE, glm::value_ptr(MVP));
}

// Sample 3
#include <glm/vec2.hpp>// glm::vec2
#include <glm/packing.hpp>// glm::packUnorm2x16
#include <glm/integer.hpp>// glm::uint
#include <glm/gtc/type_precision.hpp>// glm::i8vec2, glm::i32vec2
std::size_t const VertexCount = 4;
// Float quad geometry
std::size_t const PositionSizeF32 = VertexCount * sizeof(glm::vec2);
glm::vec2 const PositionDataF32[VertexCount] =
{
	glm::vec2(-1.0f,-1.0f),
	glm::vec2( 1.0f,-1.0f),
	glm::vec2( 1.0f, 1.0f),
	glm::vec2(-1.0f, 1.0f)
	};
// Half-float quad geometry
std::size_t const PositionSizeF16 = VertexCount * sizeof(glm::uint);
glm::uint const PositionDataF16[VertexCount] =
{
	glm::uint(glm::packUnorm2x16(glm::vec2(-1.0f, -1.0f))),
	glm::uint(glm::packUnorm2x16(glm::vec2( 1.0f, -1.0f))),
	glm::uint(glm::packUnorm2x16(glm::vec2( 1.0f, 1.0f))),
	glm::uint(glm::packUnorm2x16(glm::vec2(-1.0f, 1.0f)))
};
// 8 bits signed integer quad geometry
std::size_t const PositionSizeI8 = VertexCount * sizeof(glm::i8vec2);
glm::i8vec2 const PositionDataI8[VertexCount] =
{
	glm::i8vec2(-1,-1),
	glm::i8vec2( 1,-1),
	glm::i8vec2( 1, 1),
	glm::i8vec2(-1, 1)
};
// 32 bits signed integer quad geometry
std::size_t const PositionSizeI32 = VertexCount * sizeof(glm::i32vec2);
glm::i32vec2 const PositionDataI32[VertexCount] =
{
	glm::i32vec2 (-1,-1),
	glm::i32vec2 ( 1,-1),
	glm::i32vec2 ( 1, 1),
	glm::i32vec2 (-1, 1)
};

struct intersection
{
	glm::vec4 position;
	glm::vec3 normal;
};

/*
// Sample 4
#include <glm/vec3.hpp>// glm::vec3
#include <glm/geometric.hpp>// glm::normalize, glm::dot, glm::reflect
#include <glm/exponential.hpp>// glm::pow
#include <glm/gtc/random.hpp>// glm::vecRand3
glm::vec3 lighting
(
	intersection const & Intersection,
	material const & Material,
	light const & Light,
	glm::vec3 const & View
)
{
	glm::vec3 Color(0.0f);
	glm::vec3 LightVertor(glm::normalize(
		Light.position - Intersection.position +
		glm::vecRand3(0.0f, Light.inaccuracy));

	if(!shadow(Intersection.position, Light.position, LightVertor))
	{
		float Diffuse = glm::dot(Intersection.normal, LightVector);
		if(Diffuse <= 0.0f)
			return Color;
		if(Material.isDiffuse())
			Color += Light.color() * Material.diffuse * Diffuse;
		if(Material.isSpecular())
		{
			glm::vec3 Reflect(glm::reflect(
				glm::normalize(-LightVector),
				glm::normalize(Intersection.normal)));
			float Dot = glm::dot(Reflect, View);
			float Base = Dot > 0.0f ? Dot : 0.0f;
			float Specular = glm::pow(Base, Material.exponent);
			Color += Material.specular * Specular;
		}
	}
	return Color;
}
*/


template <typename T, glm::precision P, template<typename, glm::precision> class vecType>
T normalizeDotA(vecType<T, P> const & x, vecType<T, P> const & y)
{
	return glm::dot(x, y) * glm::inversesqrt(glm::dot(x, x) * glm::dot(y, y));
}

#define GLM_TEMPLATE_GENTYPE typename T, glm::precision P, template<typename, glm::precision> class

template <GLM_TEMPLATE_GENTYPE vecType>
T normalizeDotB(vecType<T, P> const & x, vecType<T, P> const & y)
{
	return glm::dot(x, y) * glm::inversesqrt(glm::dot(x, x) * glm::dot(y, y));
}

template <typename vecType>
typename vecType::value_type normalizeDotC(vecType const & a, vecType const & b)
{
	return glm::dot(a, b) * glm::inversesqrt(glm::dot(a, a) * glm::dot(b, b));
}

int main()
{
	glm::vec1 o(1);
	glm::vec2 a(1);
	glm::vec3 b(1);
	glm::vec4 c(1);

	glm::quat q;
	glm::dualquat p;

	glm::mat4 m(1);

	float a0 = normalizeDotA(a, a);
	float b0 = normalizeDotB(b, b);
	float c0 = normalizeDotC(c, c);

	return 0;
}
