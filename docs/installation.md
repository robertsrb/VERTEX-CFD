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

rm -rf CMake*
rm -rf .ninja*
rm DartConfiguration.tcl
rm CTestTestfile.cmake
rm build.ninja
rm VertexCFDConfig.cmake
rm -rf Testing

cmake \
    -G Ninja \
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
Once the configuration is completed, compilation can be initiated with following command.
```
ninja
```
By default, `ninja` command will use the available cores, this can be limitted as:
```
ninja -j4
```
You can replace the number 4 with the number of cores that you prefer. Once the compilation is done, VERTEX-CFD can be installed by using:
```
ninja install
```
Once installed, VERTEX-CFD is ready to run.                                                                                  

## GPU installation
