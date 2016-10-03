#pragma once

#include <glm/glm.hpp>

using namespace std;

class image {
private:
    int xSize;
    int ySize;
    

public:
    image(int x, int y);
    ~image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);

	image(string & filename);

	void loadImage(const std::string & filename);

	int pixelCount();
	glm::vec3 *pixels;

	glm::vec2 getSize();
};
