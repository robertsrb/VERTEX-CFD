SOURCE=<SOURCE_DIR>
INSTALL=<INSTALL_DIR>

BUILD="Release"

rm -rf CMake*

# Unset variable set by spack modules.
# If this is present, any directories present will be treated
# as "implicit" library paths by CMake and it will strip them
# out of the executable RPATHS. Then you have to set
# LD_LIBRARY_PATH appropriately to run jobs.
unset LIBRARY_PATH

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE="$BUILD" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL" \
    -D VertexCFD_ENABLE_COVERAGE_BUILD=ON \
    -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -fdiagnostics-color" \
    -D CLANG_FORMAT_EXECUTABLE="/usr/bin/clang-format" \
    -D Trilinos_DIR=/projects/mp_common/spack_env/v0.4/trilinos/13.2.0-mkl-complex/lib/cmake/Trilinos \
    -D VertexCFD_ENABLE_TESTING=ON \
    \
    ${SOURCE}
