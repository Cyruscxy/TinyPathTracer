#include <fstream>
#include <iostream>
#include <sstream>
#include "path_tracer.h"

int main(int argc, char *argv[]) {
	std::string path;
#ifdef PATH_TO_MESH_DIR
	path += PATH_TO_MESH_DIR;
#endif
	PathTracer pt(path + "evening_road_01_puresky.jpg");
	pt.render(path + "balls.gltf");
	return 0;
}