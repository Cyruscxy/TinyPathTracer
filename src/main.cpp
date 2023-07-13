#include <fstream>
#include <iostream>
#include <sstream>
#include "vkEngine.h"

int main(int argc, char *argv[]) {

	VkEngine engine;
	try
	{
		engine.init();

		engine.run();

		engine.cleanup();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}