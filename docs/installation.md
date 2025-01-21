---
parent: VERTEX-CFD v1.0 User Guide
nav_order: 1
---

# Installation
VERTEX-CFD supports both CPU and GPU solvers. For full CPU and GPU capabilities, please refer to GPU installation. Otherwise, check CPU installation below.

## CPU installation
For the CPU installation, first of all, make sure you have `TRILINOS_HOME` variable defined in your environment.
```
TRILINOS_HOME=PATH_TO_TRILINOS/TRILINOS_VERSION
```
Once the `TRILINOS_HOME` is set, load the dependencies:
```
module load intel
module load tbb
module load compiler-rt/
module load mkl
module load python
```
Once the environment is ready, configuration file can be run as:
```
./vertexcfd-env
```
The content of the `vertexcfd-env` file is:
```
#!/bin/sh                                                                                  

SOURCE=PATH_TO/vertex-cfd
INSTALL=INSTALLATION_PATH
Trilinos_ROOT=PATH_TO_TRILINOS/TRILINOS_VERSION
BUILD="Release"
BUILD_SYSTEM=Ninja

rm -rf CMake*
rm DartConfiguration.tcl
rm CTestTestfile.cmake
rm VertexCFDConfig.cmake
rm -rf Testing

cmake \
    -G "$BUILD_SYSTEM" \
    -D CMAKE_BUILD_TYPE="$BUILD" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL" \
    -D VertexCFD_ENABLE_COVERAGE_BUILD=OFF \
    -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -fdiagnostics-color" \
    -D VertexCFD_ENABLE_TESTING=ON \
    -D CLANG_FORMAT_EXECUTABLE="PATH_TO_CLANG_FORMAT" \
    -D Trilinos_ROOT="$Trilinos_ROOT" \
    \
    ${SOURCE}
```
Please note that `BUILD_SYSTEM` environment can be both `Ninja` or `Make`. Choose the correct one based on your system. Once the configuration is completed, compilation can be initiated with following command. Please note that we used `ninja` in this example so please switch to `make` if your `BUILD_SYSTEM` is defined as `Make`.
```
ninja
```
By default, `ninja` command will use the available cores, this can be limitted as:
```
ninja -j4
```
You can replace the number 4 with the number of cores that you prefer. We do recommend to use `-j` flag as some systems tends to compile on every CPUs on the login node and fail due to the allocated/used CPUs. Once the compilation is done, VERTEX-CFD can be installed by using:
```
ninja install
```
Once installed, VERTEX-CFD is ready to run.                                                                                  

## GPU installation
For the GPU installation, CUDA needs to be loaded in the HPC environment and Trilinos needs to be built with the GPU support. Just like the CPU version, define the `TRILINOS_HOME` variable as:
```
TRILINOS_HOME=PATH_TO_TRILINOS/TRILINOS_VERSION_WITH_CUDA
```
Once the `TRILINOS_HOME` is set, load the dependencies by including cuda:
```
module load intel
module load tbb
module load compiler-rt/
module load mkl
module load python
module load cuda
```
Once the environment is ready, configuration file can be run as:
```
./vertexcfd-env-gpu
```
The content of the `vertexcfd-env-gpu` file is:
```
#!/bin/sh                                                                                  

SOURCE=PATH_TO/vertex-cfd
INSTALL=INSTALLATION_PATH
Trilinos_ROOT=PATH_TO_TRILINOS/TRILINOS_VERSION_WITH_CUDA
BUILD="Release"
BUILD_SYSTEM=Ninja

rm -rf CMake*
rm DartConfiguration.tcl
rm CTestTestfile.cmake
rm VertexCFDConfig.cmake
rm -rf Testing

cmake \
    -G "$BUILD_SYSTEM" \
    -D CMAKE_BUILD_TYPE="$BUILD" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL" \
    -D VertexCFD_ENABLE_COVERAGE_BUILD=OFF \
    -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -fdiagnostics-color" \
    -D VertexCFD_ENABLE_TESTING=ON \
    -D CLANG_FORMAT_EXECUTABLE="PATH_TO_CLANG_FORMAT" \
    -D Trilinos_ROOT="$Trilinos_ROOT" \
    \
    ${SOURCE}
```