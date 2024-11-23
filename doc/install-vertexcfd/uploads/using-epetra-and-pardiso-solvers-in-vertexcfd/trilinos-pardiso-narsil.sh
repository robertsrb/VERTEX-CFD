#!/bin/sh

# Update these paths
SOURCE=<TRILINOS_SRC_DIR>
INSTALL=<TRILINOS_INSTALL_DIR>

BUILD="Release"

# Load relevant environment modules
module use /projects/mp_common/spack_env/v0.4/modulefiles
module load gcc/gcc-11.2.0
module load cmake/cmake-3.21.3

module use /projects/mp_common/spack_env/v0.4/spack/share/spack/modules/linux-rhel8-zen2/
module load openmpi/4.1.1
module load ninja/1.10.1
module load boost/1.74.0 netcdf-c/4.7.4
module load parmetis/4.0.3
module load intel-oneapi-mkl

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE:STRING="$BUILD" \
    -D CMAKE_INSTALL_PREFIX:PATH=$INSTALL \
    -D CMAKE_CXX_FLAGS="-Wno-implicit-int-float-conversion -Wno-user-defined-warnings" \
    -D CMAKE_CXX_STANDARD=14 \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D TPL_ENABLE_MPI=ON \
    -D TPL_ENABLE_BLAS=ON \
    -D TPL_ENABLE_LAPACK=ON \
    -D TPL_ENABLE_HDF5=ON \
    -D TPL_ENABLE_Netcdf=ON \
    -D TPL_ENABLE_Boost=ON \
    -D TPL_ENABLE_BoostLib=ON \
    -D TPL_ENABLE_Matio=OFF \
    -D TPL_ENABLE_METIS=ON \
    -D TPL_ENABLE_ParMETIS=ON \
    -D TPL_ENABLE_MKL=ON \
    -D BLAS_LIBRARY_DIRS="${MKLROOT}/lib/intel64" \
    -D BLAS_LIBRARY_NAMES="mkl_intel_lp64;mkl_gnu_thread;mkl_core" \
    -D LAPACK_LIBRARY_DIRS="${MKLROOT}/lib/intel64" \
    -D LAPACK_LIBRARY_NAMES="mkl_intel_lp64;mkl_gnu_thread;mkl_core" \
    -D MKL_LIBRARY_NAMES="mkl_intel_lp64;mkl_gnu_thread;mkl_core" \
    -D TPL_ENABLE_PARDISO_MKL=ON \
    -D MKL_LIBRARY_DIRS="${MKLROOT}/lib/intel64" \
    -D MKL_INCLUDE_DIRS="${MKLROOT}/include" \
    -D PARDISO_MKL_INCLUDE_DIRS="${MKLROOT}/include" \
    -D PARDISO_MKL_LIBRARY_DIRS="${MKLROOT}/lib/intel64" \
    -D Amesos_ENABLE_PARDISO_MKL=ON \
    -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
    -D Trilinos_ENABLE_ALL_PACKAGES=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
    -D Trilinos_ENABLE_TESTS=OFF \
    -D Trilinos_ENABLE_EXAMPLES=OFF \
    -D Trilinos_ENABLE_OpenMP=ON \
    -D Trilinos_ENABLE_Kokkos=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA=OFF \
    -D Kokkos_ARCH_EPYC=ON \
    -D Trilinos_ENABLE_KokkosKernels=ON \
    -D Trilinos_ENABLE_Sacado=ON \
    -D Trilinos_ENABLE_Tpetra=ON \
    -D Tpetra_INST_COMPLEX_DOUBLE=OFF \
    -D Tpetra_INST_COMPLEX_FLOAT=OFF \
    -D Tpetra_INST_SERIAL=ON \
    -D Tpetra_INST_OPENMP=ON \
    -D Tpetra_INST_CUDA=OFF \
    -D Trilinos_ENABLE_Intrepid2=ON \
    -D Trilinos_ENABLE_Belos=ON \
    -D Trilinos_ENABLE_Ifpack2=ON \
    -D Trilinos_ENABLE_Amesos2=ON \
    -D Trilinos_ENABLE_MueLu=ON \
    -D Trilinos_ENABLE_NOX=ON \
    -D Trilinos_ENABLE_SEACAS=ON \
    -D Trilinos_ENABLE_STKMesh=ON \
    -D Trilinos_ENABLE_STKIO=ON \
    -D Trilinos_ENABLE_Zoltan2=ON \
    -D Trilinos_ENABLE_Tempus=ON \
    -D Trilinos_ENABLE_Piro=ON \
    -D Trilinos_ENABLE_PanzerCore=ON \
    -D Trilinos_ENABLE_PanzerDofMgr=ON \
    -D Trilinos_ENABLE_PanzerDiscFE=ON \
    -D Trilinos_ENABLE_PanzerAdaptersSTK=ON \
    -D Trilinos_ENABLE_PanzerMiniEM=ON \
    -D Trilinos_ENABLE_PanzerExprEval=ON \
    -D Panzer_ENABLE_EXAMPLES=ON \
    -D TPL_ENABLE_gtest=OFF \
    -D Trilinos_ENABLE_Gtest=OFF \
    ${SOURCE}

