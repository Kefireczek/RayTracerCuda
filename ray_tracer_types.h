#pragma once

#include <glm/glm.hpp>

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

// Forward declaration of sf::Color for non-CUDA code
#ifndef __CUDACC__
namespace sf {
    class Color;
}
#endif

struct Material {
    glm::vec3 albedo;
    float specular = 0;
    float emissionStength = 0;
    glm::vec3 emissionColor = glm::vec3();

    // CUDA-friendly constructors only
    Material() : albedo(glm::vec3()) {}

    Material(glm::vec3 c) : albedo(c) {}

    Material(glm::vec3 c, float eS, glm::vec3 eC) :
        albedo(c), emissionColor(eC), emissionStength(eS) {}

    Material(glm::vec3 c, float s) : albedo(c), specular(s) {}

#ifndef __CUDACC__
    // SFML-specific constructor, only available in host code
    Material(sf::Color c);
#endif
};

struct Sphere {
    glm::vec3 center;
    float radius;
    Material material;
};

struct HitInfo {
    bool didHit;
    float dist;
    glm::vec3 hitPoint;
    glm::vec3 normal;
    Material material;
};

// Constants
const int WIDTH = 800;
const int HEIGHT = 800;
const int MAX_BOUNCE_COUNT = 4;
const int NUM_RAYS_PER_PIXEL = 8;

// Common function declarations
float RandomValue01(uint32_t& state);
float RandomValueNormalDistribution(uint32_t& state);
glm::vec3 RandomUnitVectorCosineWeighted(uint32_t& state);
HitInfo rayIntersectsSphere(const Ray& ray, const Sphere& sphere);
HitInfo CheckRayIntersections(const Ray& ray);
glm::vec3 ACESToneMapping(const glm::vec3& x);
glm::vec3 LinearToSRGB(glm::vec3 color);