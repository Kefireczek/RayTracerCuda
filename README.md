# CUDA Ray Tracer

This project is a simple GPU-based path tracer using CUDA and SFML for real-time rendering. It simulates realistic global illumination by tracing light rays that bounce through a 3D scene made of spheres and planes.

## Example

![obraz](https://github.com/user-attachments/assets/6a629638-0701-49fa-97a0-421f1474da4e)

## Features

- Real-time GPU rendering using CUDA
- Monte Carlo path tracing with multiple rays per pixel
- Diffuse and specular reflections
- Emissive materials (area lighting)
- Cosine-weighted hemisphere sampling
- ACES tone mapping and sRGB gamma correction
- Real-time image display using SFML

## Technologies

- **CUDA** – Parallel ray tracing on the GPU
- **C++** – Host application logic
- **SFML** – Rendering output to a window
- **GLM** – Vector math library

## Based on

The ray tracer was build upon [my previous one running on CPU](https://github.com/Kefireczek/RayTracerCpp), which in turn was based on [this blogpost](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/) and this [video](https://www.youtube.com/watch?v=Qz0KTGYJtUk)
