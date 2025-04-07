#pragma once

#include <SFML/Graphics.hpp>
#include "ray_tracer_types.h"

// Implementation of the sf::Color constructor for Material
Material::Material(sf::Color c) {
    albedo = { c.r / 255.0f, c.g / 255.0f, c.b / 255.0f };
}

sf::Color ConvertColor(const glm::vec3& hdrColor);
glm::vec3 GetPixelColor(int x, int y, sf::Image& image, uint32_t seed);
glm::vec3 TraceRay(Ray ray, uint32_t& rngState);
