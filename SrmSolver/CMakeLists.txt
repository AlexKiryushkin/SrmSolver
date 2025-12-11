cmake_minimum_required(VERSION 3.23)

# Project definition
project(MyCudaApp LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED)
find_package(gcem REQUIRED)

# Set language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Collect all source files in current directory
file(GLOB PROJECT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shape/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/shape/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Create executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})



# -------------------------------------------------------------------------------------------------
# Organize project in the folder
# -------------------------------------------------------------------------------------------------
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "src")


# -------------------------------------------------------------------------------------------------
# Group files by folders in project
# -------------------------------------------------------------------------------------------------
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PROJECT_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${gcem_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA runtime
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart Boost::filesystem Boost::system Boost::iostreams)

target_compile_options(MyCudaApp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

target_compile_definitions(MyCudaApp PRIVATE _USE_MATH_DEFINES >
)
