cmake_minimum_required(VERSION 3.20)

project(tinyPathTracer LANGUAGES CXX CUDA )
set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_VERSION "1.3.243.0")
set(VULKAN_PATH "${VULKAN_SDK}/${VULKAN_VERSION}")
set(VULKAN_GLSLC_EXE "${VULKAN_PATH}/Bin/glslc.exe")

find_package(CUDAToolkit 11.0 REQUIRED)

include_directories(CUDA_INCLUDE_DIRS)

set(VULKAN_LIB_LIST "vulkan-1.lib")

file(GLOB_RECURSE CXX_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR} )
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh)
file(GLOB_RECURSE THIRD_PARTY_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.h ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.hpp)
file(GLOB_RECURSE THIRD_PARTY_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(AFTER ${VULKAN_PATH}/Include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

link_directories(${VULKAN_PATH}/Bin; ${VULKAN_PATH}/Lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

foreach(src IN LISTS CXX_SRCS)
    get_filename_component(srcPath ${src} PATH)
    string(REPLACE "/" "\\" srcPathMSVC ${srcPath})
    source_group(${srcPathMSVC} FILES ${src})
endforeach()

foreach(src IN LISTS CU_SRCS)
    get_filename_component(srcPath ${src} PATH)
    string(REPLACE "/" "\\" srcPathMSVC ${srcPath})
    source_group(${srcPathMSVC} FILES ${src})
endforeach()

foreach(header IN LISTS HEADERS)
    get_filename_component(headerPath ${header} PATH)
    string(REPLACE "/" "\\" headerPathMSVC ${headerPath})
    source_group(${headerPathMSVC} FILES ${header})
endforeach()

foreach(src IN LISTS THIRD_PARTY_SRCS)
    get_filename_component(srcPath ${src} PATH)
    string(REPLACE "/" "\\" srcPathMSVC ${srcPath})
    source_group(${srcPathMSVC} FILES ${src})
endforeach()

foreach(header IN LISTS THIRD_PARTY_HEADERS)
    get_filename_component(headerPath ${header} PATH)
    string(REPLACE "/" "\\" headerPathMSVC ${headerPath})
    source_group(${headerPathMSVC} FILES ${header})
endforeach()

set(CMAKE_CXX_STANDARD 17) 
add_definitions(-DPATH_TO_MESH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/input/")
add_compile_options(-G)
add_executable(tinyPathTracer ${CXX_SRCS} ${CU_SRCS} ${HEADERS} ${THIRD_PARTY_SRCS} ${THIRD_PARTY_SRCS})

target_link_libraries(tinyPathTracer ${CMAKE_CURRENT_SOURCE_DIR}/lib/GLFW/glfw3.lib ${VULKAN_LIB_LIST} ${CMAKE_CURRENT_SOURCE_DIR}/lib/FreeImage/FreeImage.lib)

set_target_properties(tinyPathTracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_directory_properties(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT tinyPathTracer)

