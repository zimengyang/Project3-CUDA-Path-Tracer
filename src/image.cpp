#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>

#include "image.h"

image::image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new glm::vec3[x * y]) {
}

image::~image() {
    delete[] pixels;
}

void image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) { 
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}

void image::loadImage(const std::string & filename)
{

}

image::image(string & filename)
{
	//int width, height;
	int dim = 3;
	unsigned char *bytes = stbi_load(filename.c_str(), &xSize, &ySize, &dim, 0);
	// keep dim for png alpha channel
	// stbi_load is a column-major or row-major? tbd
	pixels = new glm::vec3[xSize * ySize];
	for (int y = 0; y < ySize; ++y)
		for (int x = 0; x < xSize; ++x)	
		{
			int index = y * xSize + x;
			pixels[index].x = float(bytes[dim * index + 0]) / 255.0f;
			pixels[index].y = float(bytes[dim * index + 1]) / 255.0f;
			pixels[index].z = float(bytes[dim * index + 2]) / 255.0f;
		}

	cout << xSize << " X " << ySize << endl;
	cout << "load texture " << filename.c_str() << " done!" << endl;
	/*cout << "test pixel:" 
		<< pixels[10 * ySize + 2].x << ","
		<< pixels[10 * ySize + 2].y << ","
		<< pixels[10 * ySize + 2].z << endl;;*/
	delete[] bytes; 
}

int image::pixelCount()
{
	return xSize * ySize;
}

glm::vec2 image::getSize()
{
	return glm::vec2(xSize, ySize);
}