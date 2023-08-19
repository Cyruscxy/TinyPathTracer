#include <fstream>
#include <iostream>
#include <sstream>
#include "path_tracer.h"

int main(int argc, char *argv[]) {
	std::string path;
#ifdef PATH_TO_MESH_DIR
	path += PATH_TO_MESH_DIR;
#endif
	PathTracer pt;
	pt.render(path + "ball.gltf");
	return 0;
}