CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Zimeng Yang
* Tested on: Windows 10, i7-4850 @ 2.3GHz 16GB, GT 750M (Personal Laptop)

## Roadmap
* [x] Part 1 - core features
  * [x] BSDF evaluation : diffuse, perfect specular and imperfect specular surface
  * [x] path termination using stream compaction
  * [x] toggleable method of sorting path/intersection continuous by material type
  * [x] toggleable method of using first bounce caching
* [] Part 2
* [] reduce noise

## Part1 - Core Features
|diffuse|perfect-specular and imperfect-specular|
|------|------|
|![](renderings/roadmap_cornell_diffuse_2008sample.png)|![](renderings/roadmap_cornell_imperfect_specular_2000sample.png)|
* Iterations: ~2000 
* sphere in right rendering is 0.5 reflectance combined with diffuse white [need better approximation]


### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

