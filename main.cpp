#include "ray_tracer_types.h"
#include "sfml_helpers.h"

#include <SFML/Graphics.hpp>
#include <vector>

// Define the spheres 
std::vector<Sphere> spheres = {
    {{0, 0, 5}, 1.0f, Material(glm::vec3(1, 0.75, 0.82), 1)},
    {{1, 1, 6}, 1.0f, Material(glm::vec3(0.75, 1, 0.80))},
    {{-2, 0, 7}, 1.0f, Material(glm::vec3(0.70, 0.80, 1))},
    {{-2, 0, 7}, 1.0f, Material(glm::vec3(0.70, 0.80, 1), 0.75)},
    {{3, 0, 3}, 1.0f, Material(glm::vec3(0.70, 0.40, 0.7), 0.5)},
    {{-2, 2, 4}, 1.0f, Material(glm::vec3(1, 0.80, 0.5), 0.25)},
    {{0, -15, 10}, 15.0f, Material(glm::vec3{1, 1, 1})},
    {{ 5, 2, 5}, 3.0f, Material(glm::vec3(), 2.0f, glm::vec3{1, 1, 1})},
    { { -3, 2, 10 }, 3.0f, Material(glm::vec3(), 2.0f, glm::vec3{1, 1, 1}) }
};

// Global accumulation buffer
std::vector<glm::vec3> accumulationBuffer(WIDTH* HEIGHT, glm::vec3(0.0f));

// Implement non-CUDA specific functions
sf::Color ConvertColor(const glm::vec3& hdrColor) {
    glm::vec3 mapped = ACESToneMapping(hdrColor);
    glm::vec3 gammaCorrected = LinearToSRGB(mapped);

    int r = static_cast<int>(gammaCorrected.r * 255.0f);
    int g = static_cast<int>(gammaCorrected.g * 255.0f);
    int b = static_cast<int>(gammaCorrected.b * 255.0f);

    return sf::Color(r, g, b);
}

glm::vec3 GetPixelColor(int x, int y, sf::Image& image, uint32_t seed) {
    glm::vec2 pos(x, y);

    float aspectRatio = (float)WIDTH / HEIGHT;

    glm::vec2 jitter = glm::vec2(RandomValue01(seed), RandomValue01(seed)) - 0.5f;
    pos += jitter;

    // Normalizacja wspó³rzêdnych ekranu
    glm::vec2 uv(
        ((float)pos.x / WIDTH) * 2 - 1,
        -((float)pos.y / HEIGHT) * 2 + 1
    );

    Ray ray;
    ray.origin = { 0,0,0 };

    ray.direction = glm::normalize(glm::vec3(uv.x, uv.y / aspectRatio, 1.0f) - ray.origin);

    glm::vec3 totalIncomingLight = glm::vec3(0);

    for (int i = 0; i < NUM_RAYS_PER_PIXEL; i++) {
        totalIncomingLight += TraceRay(ray, seed);
    }

    totalIncomingLight /= NUM_RAYS_PER_PIXEL;

    return totalIncomingLight;
}

glm::vec3 TraceRay(Ray ray, uint32_t& rngState) {
    glm::vec3 rayColor = glm::vec3(1.0f);
    glm::vec3 incomingLight = glm::vec3(0.0f);

    for (int i = 0; i < MAX_BOUNCE_COUNT; i++) {
        HitInfo hit = CheckRayIntersections(ray);
        if (hit.didHit) {
            ray.origin = hit.hitPoint;

            if (RandomValue01(rngState) > hit.material.specular) {
                ray.direction = glm::normalize(hit.normal + RandomUnitVectorCosineWeighted(rngState));
            }
            else {
                ray.direction = glm::reflect(ray.direction, hit.normal);
            }

            Material material = hit.material;
            glm::vec3 emittedLight = material.emissionColor * material.emissionStength;
            incomingLight += emittedLight * rayColor;
            rayColor *= material.albedo;

            float p = std::max(rayColor.x, std::max(rayColor.y, rayColor.z));
            if (RandomValue01(rngState) > p) {
                break;
            }

            rayColor *= 1.0f / p;
        }
        else {
            break;
        }
    }

    return incomingLight;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Tracer");
    sf::Image newFrame;
    newFrame.create(WIDTH, HEIGHT, sf::Color::Black);
    sf::Texture texture;

    uint32_t iFrame = 0;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            // Close window: exit
            if (event.type == sf::Event::Closed)
                window.close();
        }

        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                uint32_t rngState = (uint32_t)(uint32_t(x) * uint32_t(1973) + uint32_t(y) * uint32_t(9277) + uint32_t(iFrame) * uint32_t(26699)) | uint32_t(1);

                glm::vec3 newColor = GetPixelColor(x, y, newFrame, rngState);

                float weight = 1.0f / (iFrame + 1);

                accumulationBuffer[y * WIDTH + x] = accumulationBuffer[y * WIDTH + x] * (1.0f - weight) + newColor * weight;

                glm::vec3 accColor = accumulationBuffer[y * WIDTH + x];
                sf::Color color = ConvertColor(accColor);

                newFrame.setPixel(x, y, color);
            }
        }

        if (!texture.loadFromImage(newFrame)) {
            return 1;
        }
        sf::Sprite sprite(texture);

        window.clear();
        window.draw(sprite);
        window.display();

        iFrame++;
    }

    return 0;
}