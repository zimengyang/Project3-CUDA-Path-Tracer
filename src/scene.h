#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "image.h"

using namespace std;
// test csg primitives
extern Geom csg_box, csg_sphere;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

	void InitializeCSGTree();

	// keep a pointer vector intead of object vector
	// recycle
	std::vector<image*> textures;
	std::vector<image*> normalMaps;
};
