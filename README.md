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
* [ ] Part 2
* [ ] reduce noise
* [ ] artifacts ? 
* [ ] performace anylasis for `reshuffleByMaterialIDs` and `useFirstBounceIntersectionCache`

## Part1 - Core Features
| transmission test |
|----|
|![](renderings/roadmap_cornell_0.8Rf_0.1Rl_0.1Di_perfect_transmission_3392sample.png)|
* Iterations: ~3300
* Test render for:
  * perfect transmission (right sphere): 1.0 refraction
  * weighted material (left sphere): 0.8 refraction + 0.1 reflection + 0.1 diffuse

|diffuse|perfect-specular and imperfect-specular|
|------|------|
|![](renderings/roadmap_cornell_diffuse_2008sample.png)|![](renderings/roadmap_cornell_imperfect_specular_2000sample.png)|
* Iterations: ~2000 
* sphere in right rendering is 0.5 reflectance combined with diffuse white [need better approximation]


### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

