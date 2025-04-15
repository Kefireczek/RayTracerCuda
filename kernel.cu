#include <SFML/Graphics.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

#include <stdio.h>

const int WIDTH = 800;
const int HEIGHT = 800;
const int MAX_BOUNCE_COUNT = 10;
const int NUM_RAYS_PER_PIXEL = 10;


float host_C_PHI = 7.017f;  // Initial value
float host_N_PHI = 0.221f;  // Initial value
float host_P_PHI = 1.215f;  // Initial value


float kernel1D[5] = { 1.0f / 16, 1.0f / 4, 3.0f / 8, 1.0f / 4, 1.0f / 16 };

#define KERNEL_SIZE 25
__constant__ float kernel[25];

__constant__ int2 offsets[25];

__host__ __device__ struct Material {
	float3 albedo;
	float specular = 0;
	float emissionStength = 0;
	float3 emissionColor = make_float3(0.0f, 0.0f, 0.0f);

	__host__ __device__ Material(sf::Color c) {
		albedo = make_float3(c.r, c.g, c.b);
	}

	__host__ __device__ Material() {
		albedo = make_float3(0.0f, 0.0f, 0.0f);
	}

	__host__ __device__ Material(glm::vec3 c) {
		albedo = make_float3(c.x, c.y, c.z);
	}

	__host__ __device__ Material(glm::vec3 c, float eS, glm::vec3 eC) {
		albedo = make_float3(c.x, c.y, c.z);
		emissionColor = make_float3(eC.x, eC.y, eC.z);
		emissionStength = eS;
	}

	__host__ __device__ Material(glm::vec3 c, float s) {
		albedo = make_float3(c.x, c.y, c.z);
		specular = s;
	}
};

struct Ray {
	float3 origin;
	float3 direction;
};

struct HitInfo
{
	bool didHit;
	float dist;
	float3 hitPoint;
	float3 normal;
	Material material;
};

//Scene
struct Sphere {
	float3 center;
	float radius;
	Material material;
};

struct Plane {
	float3 point;
	float3 normal;
	Material material;
};

#define NUM_SPHERES 8
__managed__ Sphere* d_spheres;
std::vector<Sphere> hostSpheres = {
	{{0, 0, 5}, 1.0f, Material(glm::vec3(0.5, 0.5, 0.5), 1)},
	{{1, 1, 6}, 1.0f, Material(glm::vec3(0.4, 0.75, 0.95))},
	{{-2, -2, 5}, 1.0f, Material(glm::vec3(0.5, 0.9, 0.7), 0.3)},
	{{2, -3, 7}, 1.0f, Material(glm::vec3(1.0, 0.6, 0.7), 0.75)},
	{{3, -1, 3}, 1.0f, Material(glm::vec3(0.75, 0.65, 1.0), 0.5)},
	{{-2, 2, 4}, 1.0f, Material(glm::vec3(1.0, 0.95, 0.5), 0.25)},
	{{ 5, 3, 5}, 3.0f, Material(glm::vec3(1.0, 0.7, 0.5))},
	{ { -3, 4, 10 }, 3.0f, Material(glm::vec3(0.4, 0.8, 0.5))}
};

#define NUM_PLANES 5
__managed__ Plane* d_planes;
std::vector<Plane> hostPlanes = {
	{{0, 0, 20}, {0, 0, 1}, Material(glm::vec3(0.4, 0.4, 0.4))},
	{{-10, 0, 0}, {-1, 0, 0}, Material(glm::vec3(0, 0.4, 0))},
	{{10, 0, 0}, {1, 0, 0}, Material(glm::vec3(0, 0, 0.4))},
	{{0, 10, 0}, {0, 1, 0}, Material(glm::vec3(0.9, 0.9, 0.9), 1, glm::vec3(1, 1, 1))},
	{{0, -10, 0}, {0, -1, 0}, Material(glm::vec3(0.4, 0.4, 0.4))},
	//{{0, 0, -10}, {0, 0, -1}, Material(glm::vec3(0.4, 0.4, 0.4))},
};

//Intersection logic
__device__ HitInfo rayIntersectsSphere(const Ray& ray, const Sphere& sphere) {
	HitInfo hitInfo{};

	float3 oc = ray.origin - sphere.center;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0f * dot(oc, ray.direction);
	float c = dot(oc, oc) - sphere.radius * sphere.radius;

	float discriminant = b * b - 4 * a * c;

	if (discriminant >= 0) {
		float t0 = (-b - std::sqrt(discriminant)) / (2.0f * a);

		if (t0 >= 0) {
			hitInfo.didHit = true;
			hitInfo.dist = t0;
			hitInfo.hitPoint = ray.origin + ray.direction * t0;
			hitInfo.normal = normalize(hitInfo.hitPoint - sphere.center);
			hitInfo.material = sphere.material;
		}
	}

	return hitInfo;
}

__device__ HitInfo rayIntersectsPlane(const Ray& ray, const Plane& plane) {
	HitInfo hitInfo{};

	float denom = dot(plane.normal, ray.direction);
	if (denom > 1e-6) {
		float3 p0l0 = plane.point - ray.origin;
		hitInfo.dist = dot(p0l0, plane.normal) / denom;
		hitInfo.didHit = true;
		hitInfo.material = plane.material;
		hitInfo.normal = plane.normal * -1;
		hitInfo.hitPoint = ray.origin + ray.direction * hitInfo.dist;
	}

	return hitInfo;
}

//Random values
__device__ float RandomValue01(uint32_t& state) {
	state = state * 747796405 + 2891336453;
	uint32_t result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
	result = (result >> 22) ^ result;
	return result / 4294967295.0;
}

__device__ float RandomValueNormalDistribution(uint32_t& state) {
	float theta = 2 * 3.1415926 * RandomValue01(state);
	float rho = sqrt(-2 * log(RandomValue01(state)));
	return rho * cos(theta);
}

__device__ float3 RandomUnitVectorCosineWeighted(uint32_t& state) {
	float z = RandomValue01(state) * 2.0f - 1.0f;
	float a = RandomValue01(state) * 6.28318;
	float r = sqrt(1.0f - z * z);
	float x = r * cos(a);
	float y = r * sin(a);
	return make_float3(x, y, z);
}

//Ray logic
__device__ HitInfo CheckRayIntersections(const Ray& ray) {
	HitInfo closestHit{};
	closestHit.didHit = false;
	float closestDistance = 50;

	for (int i = 0; i < NUM_SPHERES; i++) {
		HitInfo hit = rayIntersectsSphere(ray, d_spheres[i]);
		if (hit.didHit && hit.dist < closestDistance) {
			closestDistance = hit.dist;
			closestHit = hit;
		}
	}

	for (int i = 0; i < NUM_PLANES; i++) {
		HitInfo hit = rayIntersectsPlane(ray, d_planes[i]);
		if (hit.didHit && hit.dist < closestDistance) {
			closestDistance = hit.dist;
			closestHit = hit;
		}
	}

	return closestHit;
}

__device__ float3 TraceRay(Ray ray, uint32_t& rngState) {
	float3 rayColor = make_float3(1.0f, 1.0f, 1.0f);
	float3 incomingLight = make_float3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < MAX_BOUNCE_COUNT; i++)
	{
		HitInfo hit = CheckRayIntersections(ray);
		if (hit.didHit) {
			ray.origin = hit.hitPoint;

			if (RandomValue01(rngState) > hit.material.specular) {
				ray.direction = normalize(hit.normal + RandomUnitVectorCosineWeighted(rngState));
			}
			else {
				ray.direction = reflect(ray.direction, hit.normal);
			}


			Material material = hit.material;
			float3 emittedLight = material.emissionColor * material.emissionStength;
			incomingLight += emittedLight * rayColor;
			rayColor *= material.albedo;

			float p = max(rayColor.x, max(rayColor.y, rayColor.z));
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

__device__ float3 GetPixelColor(int x, int y, uint32_t seed)
{
	float2 pos = make_float2(x, y);

	float aspectRatio = (float)WIDTH / HEIGHT;

	float2 jitter = (make_float2(RandomValue01(seed), RandomValue01(seed)) - 0.5f) * 2;
	pos += jitter;

	// Normalizacja współrzędnych ekranu
	float2 uv = make_float2(
		((float)pos.x / WIDTH) * 2 - 1,
		-((float)pos.y / HEIGHT) * 2 + 1
	);

	Ray ray;
	ray.origin = { 0,0,0 };

	ray.direction = normalize(make_float3(uv.x, uv.y / aspectRatio, 1.0f) - ray.origin);

	float3 totalIncomingLight = make_float3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < NUM_RAYS_PER_PIXEL; i++)
	{
		totalIncomingLight += TraceRay(ray, seed);
	}

	totalIncomingLight /= NUM_RAYS_PER_PIXEL;

	return totalIncomingLight;
}

//Rendering Kernel
__global__ void renderKernel(float3* accBuffer, float3* posBuffer, float3* normalBuffer, int width, int height, uint32_t iFrame) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= WIDTH || y >= HEIGHT) return;

	int index = y * width + x;
	uint32_t rngState = (uint32_t)(uint32_t(x) * uint32_t(1973) + uint32_t(y) * uint32_t(9277) + uint32_t(iFrame) * uint32_t(26699)) | uint32_t(1);

	Ray ray;
	ray.origin = { 0,0,0 };
	float aspectRatio = (float)width / height;
	float2 uv = make_float2(
		((float)x / width) * 2 - 1,
		-((float)y / height) * 2 + 1
	);
	ray.direction = normalize(make_float3(uv.x, uv.y / aspectRatio, 1.0f));

	HitInfo hit = CheckRayIntersections(ray);
	if (hit.didHit) {
		posBuffer[index] = hit.hitPoint;
		normalBuffer[index] = hit.normal;
	}
	else {
		posBuffer[index] = make_float3(0, 0, 0);
		normalBuffer[index] = make_float3(0, 0, 0);
	}

	float weight = 1.0f / (iFrame + 1);
	float3 color = GetPixelColor(x, y, rngState);
	accBuffer[index] = color * weight + accBuffer[index] * (1.0f - weight);
}

//Color correction
glm::vec3 ACESToneMapping(const glm::vec3& x) {
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

glm::vec3 LinearToSRGB(glm::vec3 color) {
	glm::vec3 higher = glm::pow(color, glm::vec3(1.0f / 2.4f)) * 1.055f - 0.055f;
	glm::vec3 lower = color * 12.92f;

	glm::vec3 result;
	result.x = (color.x < 0.0031308f) ? lower.x : higher.x;
	result.y = (color.y < 0.0031308f) ? lower.y : higher.y;
	result.z = (color.z < 0.0031308f) ? lower.z : higher.z;

	return result;
}

sf::Color ConvertColor(const float3& hdrColor) {

	glm::vec3 c = glm::vec3(hdrColor.x, hdrColor.y, hdrColor.z);

	glm::vec3 mapped = ACESToneMapping(c);

	glm::vec3 gammaCorrected = LinearToSRGB(mapped);
	//glm::vec3 gammaCorrected = mapped;

	int r = static_cast<int>(gammaCorrected.r * 255.0f);
	int g = static_cast<int>(gammaCorrected.g * 255.0f);
	int b = static_cast<int>(gammaCorrected.b * 255.0f);

	return sf::Color(r, g, b);
}

//Denoising
__global__ void atrousFilterKernel(float3* inputBuffer, float3* posBuffer, float3* normalBuffer, float3* outputBuffer, float c_phi, float n_phi, float p_phi, int stepWidth) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= WIDTH || y >= HEIGHT) return;

	int idx = y * WIDTH + x;

	float3 c_val = inputBuffer[idx];
	float3 n_val = normalBuffer[idx];
	float3 p_val = posBuffer[idx];

	float3 sum = make_float3(0);
	float cum_w = 0.0f;

	float filter[3] = { 3.f / 8.f, 1.f / 4.f, 1.f / 16.f };

	for (int v = -2; v <= 2; v++)
	{
		for (int u = -2; u <= 2; u++)
		{
			int sampleX = x + u * stepWidth;
			int sampleY = y + v * stepWidth;

			if (sampleX < 0 || sampleX >= WIDTH || sampleY < 0 || sampleY >= HEIGHT) continue;
			int sampleIdx = sampleY * WIDTH + sampleX;

			float3 c_sample = inputBuffer[sampleIdx];
			float3 diff = c_val - c_sample;
			float dist2 = dot(diff, diff);
			float c_w = fminf(expf(-dist2 / (c_phi * c_phi)), 1.0f);

			float3 n_sample = normalBuffer[sampleIdx];
			diff = n_val - n_sample;
			dist2 = fmaxf(dot(diff, diff), 0.0f);
			float n_w = fminf(expf(-dist2 / (n_phi * n_phi)), 1.0f);

			float3 p_sample = posBuffer[sampleIdx];
			diff = p_val - p_sample;
			dist2 = dot(diff, diff);
			float p_w = fminf(expf(-dist2 / (p_phi * p_phi)), 1.0f);

			float f_val = filter[max(abs(u), abs(v))];

			float weight = c_w * n_w * p_w;
			sum += c_sample * weight * f_val;
			cum_w += weight * f_val;
		}
	}
	outputBuffer[idx] = sum / cum_w;
}

float3* d_accumulationBuffer;
float3* d_positionBuffer;
float3* d_normalBuffer;
float3* d_outputBuffer;


void displayParameters(sf::RenderWindow& window) {
	sf::Font font;
	if (!font.loadFromFile("arial.ttf")) {  // Make sure you have this font or change to one you have
		return;
	}

	char buffer[256];
	sprintf(buffer, "C_PHI: %.3f | N_PHI: %.3f | P_PHI: %.3f", host_C_PHI, host_N_PHI, host_P_PHI);

	sf::Text text(buffer, font, 14);
	text.setPosition(10, 10);
	text.setFillColor(sf::Color::White);
	window.draw(text);
}

void DenoiseImage() {
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

	float3* d_denoiserBuffer;
	cudaMalloc(&d_denoiserBuffer, WIDTH * HEIGHT * sizeof(float3));
	cudaMemcpy(d_outputBuffer, d_accumulationBuffer, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToDevice);

	for (int k = 0; k < 5; k++)
	{
		float stepWidth = 1 << k;

		std::swap(d_outputBuffer, d_denoiserBuffer);

		atrousFilterKernel << < numBlocks, threadsPerBlock >> > (d_denoiserBuffer, d_positionBuffer, d_normalBuffer, d_outputBuffer, (host_C_PHI / stepWidth), host_N_PHI, host_P_PHI, stepWidth);
		cudaDeviceSynchronize();
	}
	cudaFree(d_denoiserBuffer);
}

int main() {
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Tracer");
	sf::Image newFrame;
	newFrame.create(WIDTH, HEIGHT, sf::Color::Black);
	sf::Texture texture;



	cudaMalloc(&d_accumulationBuffer, WIDTH * HEIGHT * sizeof(float3));
	cudaMalloc(&d_positionBuffer, WIDTH * HEIGHT * sizeof(float3));
	cudaMalloc(&d_normalBuffer, WIDTH * HEIGHT * sizeof(float3));
	cudaMalloc(&d_outputBuffer, WIDTH * HEIGHT * sizeof(float3));

	//__managed__ Sphere* d_spheres;
	cudaMallocManaged(&d_spheres, NUM_SPHERES * sizeof(Sphere));
	memcpy(d_spheres, hostSpheres.data(), NUM_SPHERES * sizeof(Sphere));

	cudaMallocManaged(&d_planes, NUM_PLANES * sizeof(Plane));
	memcpy(d_planes, hostPlanes.data(), NUM_PLANES * sizeof(Plane));

	enum class DisplayMode { Denoised, NoDenoiser };
	DisplayMode mode = DisplayMode::NoDenoiser;

	uint32_t iFrame = 0;

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
	renderKernel << < numBlocks, threadsPerBlock >> > (d_accumulationBuffer, d_positionBuffer, d_normalBuffer, WIDTH, HEIGHT, iFrame);
	cudaDeviceSynchronize();

	float hostKernel[25] = { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2], kernel1D[0] * kernel1D[3], kernel1D[0] * kernel1D[4], kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2], kernel1D[1] * kernel1D[3], kernel1D[1] * kernel1D[4], kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2], kernel1D[2] * kernel1D[3], kernel1D[2] * kernel1D[4], kernel1D[3] * kernel1D[0], kernel1D[3] * kernel1D[1], kernel1D[3] * kernel1D[2], kernel1D[3] * kernel1D[3], kernel1D[3] * kernel1D[4], kernel1D[4] * kernel1D[0], kernel1D[4] * kernel1D[1], kernel1D[4] * kernel1D[2], kernel1D[4] * kernel1D[3], kernel1D[4] * kernel1D[4] };
	cudaMemcpyToSymbol(kernel, &hostKernel, 25 * sizeof(float));

	int2 hostOffsets[25] = { make_int2(-2, -2), make_int2(-1, -2), make_int2(0, -2), make_int2(1, -2), make_int2(2, -2), make_int2(-2, -1), make_int2(-1, -1), make_int2(0, -1), make_int2(1, -1), make_int2(2, -1), make_int2(-2, 0), make_int2(-1, 0), make_int2(0, 0), make_int2(1, 0), make_int2(2, 0), make_int2(-2, 1), make_int2(-1, 1), make_int2(0, 1), make_int2(1, 1), make_int2(2, 1), make_int2(-2, 2), make_int2(-1, 2), make_int2(0, 2), make_int2(1, 2), make_int2(2, 2) };
	cudaMemcpyToSymbol(offsets, &hostOffsets, 25 * sizeof(int2));

	DenoiseImage();

	float totalWeight = 0.0f;
	for (int i = 0; i < 25; i++) {
		totalWeight += hostKernel[i];
	}
	printf("Total kernel weight: %f\n", totalWeight);

	std::vector<float3> hostDenoiserOutput(WIDTH * HEIGHT);
	cudaMemcpy(hostDenoiserOutput.data(), d_outputBuffer, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

	std::vector<float3> hostBuffer(WIDTH * HEIGHT);
	cudaMemcpy(hostBuffer.data(), d_accumulationBuffer, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

	bool needsUpdate = false;
	float adjustmentSpeed = 0.005f;

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
			if (event.type == sf::Event::KeyPressed) {
				switch (event.key.code) {
				case sf::Keyboard::Q:
					host_C_PHI += adjustmentSpeed;
					needsUpdate = true;
					break;
				case sf::Keyboard::A:
					host_C_PHI = max(0.1f, host_C_PHI - adjustmentSpeed);
					needsUpdate = true;
					break;

				case sf::Keyboard::W:
					host_N_PHI += adjustmentSpeed;
					needsUpdate = true;
					break;
				case sf::Keyboard::S:
					host_N_PHI = max(0.001f, host_N_PHI - adjustmentSpeed);
					needsUpdate = true;
					break;

				case sf::Keyboard::E:
					host_P_PHI += adjustmentSpeed;
					needsUpdate = true;
					break;
				case sf::Keyboard::D: 
					host_P_PHI = max(0.001f, host_P_PHI - adjustmentSpeed);
					needsUpdate = true;
					break;
				}
			}
			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Space) {
				if (mode == DisplayMode::Denoised) {
					mode = DisplayMode::NoDenoiser;
					needsUpdate = true;
				}
				else if (mode == DisplayMode::NoDenoiser) {
					mode = DisplayMode::Denoised;
					needsUpdate = true;
				}	
			}
		}


		if (needsUpdate) {
			DenoiseImage();

			// Update host buffer with new denoised results
			cudaMemcpy(hostDenoiserOutput.data(), d_outputBuffer,
				WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

			// Update image based on current display mode
			for (int y = 0; y < HEIGHT; ++y) {
				for (int x = 0; x < WIDTH; ++x) {
					int index = y * WIDTH + x;
					sf::Color pixelColor;
					if (mode == DisplayMode::Denoised) {
						pixelColor = ConvertColor(hostDenoiserOutput[index]);
					}
					else if (mode == DisplayMode::NoDenoiser) {
						pixelColor = ConvertColor(hostBuffer[index]);
					}
					newFrame.setPixel(x, y, pixelColor);
				}
			}

			if (!texture.loadFromImage(newFrame)) {
				return 1;
			}

			needsUpdate = false;
		}



		sf::Sprite sprite(texture);

		window.clear();
		window.draw(sprite);

		displayParameters(window);

		window.display();
	}



	cudaFree(d_accumulationBuffer);
	cudaFree(d_positionBuffer);
	cudaFree(d_normalBuffer);
	cudaFree(d_outputBuffer);
	cudaFree(d_planes);
	cudaFree(d_spheres);

	return 0;
}

