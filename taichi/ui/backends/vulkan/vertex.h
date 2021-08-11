#pragma once

#include <array>
#include <volk.h>

namespace taichi {
namespace ui {

struct Vertex {
  struct vec3 {
    float x;
    float y;
    float z;
  };
  struct vec2 {
    float x;
    float y;
  };
  vec3 pos;
  vec3 normal;
  vec2 texCoord;
  vec3 color;

  static VkVertexInputBindingDescription get_binding_description() {
    VkVertexInputBindingDescription binding_description{};
    binding_description.binding = 0;
    binding_description.stride = sizeof(Vertex);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return binding_description;
  }

  static std::array<VkVertexInputAttributeDescription, 4>
  get_attribute_descriptions() {
    std::array<VkVertexInputAttributeDescription, 4> attribute_descriptions{};

    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[0].offset = offsetof(Vertex, pos);

    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = offsetof(Vertex, normal);

    attribute_descriptions[2].binding = 0;
    attribute_descriptions[2].location = 2;
    attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_descriptions[2].offset = offsetof(Vertex, texCoord);

    attribute_descriptions[3].binding = 0;
    attribute_descriptions[3].location = 3;
    attribute_descriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[3].offset = offsetof(Vertex, color);

    return attribute_descriptions;
  }
};

}  // namespace ui
}  // namespace taichi
