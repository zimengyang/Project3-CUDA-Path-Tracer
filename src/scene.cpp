#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

Geom csg_box, csg_sphere;
Scene::Scene(string filename) {

	textures.clear();

	cout << "Reading scene from " << filename << " ..." << endl;
	cout << " " << endl;
	char* fname = (char*)filename.c_str();
	fp_in.open(fname);
	if (!fp_in.is_open()) {
		cout << "Error reading from file - aborting!" << endl;
		throw;
	}
	while (fp_in.good()) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
				loadMaterial(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
				loadGeom(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
				loadCamera();
				cout << " " << endl;
			}
		}
	}

	InitializeCSGTree();
}

void Scene::InitializeCSGTree()
{
	// test set csg primitives
	csg_box.type = CUBE;
	csg_box.materialid = 2; // red
	csg_box.translation = glm::vec3(0, 0, 0);
	csg_box.rotation = glm::vec3(0, 0, 0);
	csg_box.scale = glm::vec3(1,1,1);
	csg_box.hasMotionBlur = false;
	csg_box.transform = utilityCore::buildTransformationMatrix(
		csg_box.translation, csg_box.rotation, csg_box.scale);
	csg_box.inverseTransform = glm::inverse(csg_box.transform);
	csg_box.invTranspose = glm::inverseTranspose(csg_box.transform);

	csg_sphere.type = SPHERE;
	csg_sphere.materialid = 3; // green
	csg_sphere.translation = glm::vec3(0, 0, 0);
	csg_sphere.rotation = glm::vec3(0, 0, 0);
	csg_sphere.scale = glm::vec3(1.3f, 1.3f, 1.3f);
	csg_sphere.hasMotionBlur = false;
	csg_sphere.transform = utilityCore::buildTransformationMatrix(
		csg_sphere.translation, csg_sphere.rotation, csg_sphere.scale);
	csg_sphere.inverseTransform = glm::inverse(csg_sphere.transform);
	csg_sphere.invTranspose = glm::inverseTranspose(csg_sphere.transform);
}


int Scene::loadGeom(string objectid) {
	int id = atoi(objectid.c_str());
	if (id != geoms.size()) {
		cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
		return -1;
	}
	else {
		cout << "Loading Geom " << id << "..." << endl;
		Geom newGeom;
		string line;

		//load object type
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			if (strcmp(line.c_str(), "sphere") == 0) {
				cout << "Creating new sphere..." << endl;
				newGeom.type = SPHERE;
			}
			else if (strcmp(line.c_str(), "cube") == 0) {
				cout << "Creating new cube..." << endl;
				newGeom.type = CUBE;
			}
			else if (strcmp(line.c_str(), "csg") == 0)
			{
				cout << "Creating new CSG..." << endl;
				newGeom.type = CSG;
			}

		}

		//link material
		utilityCore::safeGetline(fp_in, line);
		if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			newGeom.materialid = atoi(tokens[1].c_str());
			cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
		}

		//load transformations
		bool motionBlur_translation = false;
		bool motionBlur_rotation = false;
		bool motionBlur_scale = false;

		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "TRANS_DST") == 0)
			{
				newGeom.translation_dst = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				motionBlur_translation = true;
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT_DST") == 0)
			{
				newGeom.rotation_dst = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				motionBlur_rotation = true;
			}
			else if (strcmp(tokens[0].c_str(), "SCALE_DST") == 0)
			{
				newGeom.scale_dst = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				motionBlur_scale = true;
			}

			utilityCore::safeGetline(fp_in, line);
		}

		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);


		// assign motion blur transformations
		if (!motionBlur_translation)
		{
			newGeom.translation_dst = newGeom.translation;
		}
		if (!motionBlur_rotation)
		{
			newGeom.rotation_dst = newGeom.rotation;
		}
		if (!motionBlur_scale)
		{
			newGeom.scale_dst = newGeom.scale;
		}

		newGeom.hasMotionBlur = (motionBlur_rotation || motionBlur_scale || motionBlur_translation);

		//newGeom.transform_dst = utilityCore::buildTransformationMatrix(
		//	newGeom.translation_dst, newGeom.rotation_dst, newGeom.scale_dst);
		//newGeom.inverseTransform_dst = glm::inverse(newGeom.transform_dst);
		//newGeom.invTranspose_dst = glm::inverseTranspose(newGeom.transform_dst);

		geoms.push_back(newGeom);
		return 1;
	}
}

int Scene::loadCamera() {
	cout << "Loading Camera ..." << endl;
	RenderState &state = this->state;
	Camera &camera = state.camera;
	float fovy;

	//load static properties
	for (int i = 0; i < 5; i++) {
		string line;
		utilityCore::safeGetline(fp_in, line);
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "RES") == 0) {
			camera.resolution.x = atoi(tokens[1].c_str());
			camera.resolution.y = atoi(tokens[2].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
			fovy = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
			state.iterations = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
			state.traceDepth = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
			state.imageName = tokens[1];
		}
	}

	// initialize camera DOF to zero
	camera.DOF = glm::vec2(0, 0);
	string line;
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "EYE") == 0) {
			camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
			camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
			camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "DOF") == 0){
			camera.DOF = glm::vec2(atof(tokens[1].c_str()), atof(tokens[2].c_str()));
		}

		utilityCore::safeGetline(fp_in, line);
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy * (PI / 180));
	float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
	float fovx = (atan(xscaled) * 180) / PI;
	camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
		, 2 * yscaled / (float)camera.resolution.y);

	camera.view = glm::normalize(camera.lookAt - camera.position);

	//set up render camera stuff
	int arraylen = camera.resolution.x * camera.resolution.y;
	state.image.resize(arraylen);
	std::fill(state.image.begin(), state.image.end(), glm::vec3());

	cout << "Loaded camera!" << endl;
	return 1;
}

int Scene::loadMaterial(string materialid) {
	int id = atoi(materialid.c_str());
	if (id != materials.size()) {
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 8; i++) {
			string line;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0)
			{
				string texFilename = tokens[1];
				if (strcmp(texFilename.c_str(), "NULL") == 0)
				{
					newMaterial.texId = -1;
				}
				else
				{
					// record new texture, push
					newMaterial.texId = this->textures.size();
					this->textures.push_back(new image(texFilename));
				}
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}
