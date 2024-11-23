[Home page](https://code-int.ornl.gov/vertex/vertex-cfd/-/wikis/home)

This page describes the steps necessary to build and execute VertexCFD using the Epetra solver stack within Trilinos. Epetra represents the legacy software stack in Trilinos, supporting only CPU execution (no GPU support) and only double precision arithmetic. In contrast, the Tpetra software stack (the default in VertexCFD) supports GPU execution and arbitrary data types, including support for automatic differentiation types. However, Epetra has been in use much longer and is therefore generally more robust, has more solver options available, and in some cases offers improved performance. We will be working with the Trilinos development team to improve the capabilities of the Tpetra solvers to attempt to ultimately match the Epetra features, but at the current time there are likely benefits to using Epetra instead of Tpetra.

<details>
<summary>Building Trilinos with Pardiso support</summary>

**On Narsil, this step can be skipped and you can simply use existing Trilinos installations with Pardiso support located at `/projects/mp_common/spack_env/v0.4/trilinos/13.2.0-mkl-complex/lib/cmake/Trilinos`.**

The primary reason for using Epetra is to enable the usage of the Pardiso sparse direct solver, which is both very efficient and supports CPU multithreading. First, the Intel OneAPI library (which contains the Math Kernel Library, or MKL) should be installed. This is most easily installed through the `intel-oneapi-mkl` Spack package. This is already installed on the Narsil machine and can be loaded with:
```plaintext
module load intel-oneapi-mkl
```
The same module is also available on Narsil:
```plaintext
module load intel-oneapi-mkl/2021.1.1
```
Next, Trilinos must be built with support for MKL and Pardiso. For a new build of Trilinos, it is necessary to add the following options to an existing Trilinos CMake configure script (full configure script is attached below):

```plaintext
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
```

Note that `MKLROOT` is an environment variable set by the `intel-oneapi-mkl` module.

</details>
<details>
<summary>Building VertexCFD with MKL-enabled Trilinos</summary>
Nothing special is required from VertexCFD -- simply reference an install of Trilinos that has MKL and Pardiso enabled. The following works on Narsil (full script also attached below) and is already part of the [installation workflow for VertexCFD](doc/install-vertexcfd/install-vertexcfd-on-narsil-cpu.md):

```plaintext
# Set these to the location of VertexCFD source and the desired install location
SOURCE=<SOURCE_DIR>
INSTALL=<INSTALL_DIR>
BUILD="Release"

export TRILINOS_DIR=/projects/mp_common/spack_env/v0.4/trilinos/13.2.0-mkl-complex/lib/cmake/Trilinos
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${TRILINOS_DIR}

cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE="$BUILD" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL" \
    -D CMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -fdiagnostics-color" \
    -D BUILD_SHARED_LIBS=ON \
    -D CLANG_FORMAT_EXECUTABLE="/usr/bin/clang-format" \
    -D VertexCFD_ENABLE_TESTING=ON \
    ${SOURCE}
```

</details>
<details>
<summary>Enabling the Epetra and Pardiso solvers</summary>

From an existing VertexCFD XML input file, a few changes are necessary to use Epetra/Pardiso (see `examples/inputs/tp7_epetra.xml` for a working example):

1. In the `"User Data"` parameter list, set the linear algebra type to Epetra:

   ```plaintext
   <Parameter name="Linear Algebra Type"  type="string" value="Epetra"/>
   ```
2. In the `"Linear Solver"` list, set the preconditioner type to Ifpack (rather than Ifpack2):

   ```plaintext
   <Parameter name="Preconditioner Type" type="string" value="Ifpack"  />
   ```

   Failure to do this will result in a fairly incomprehensible message similar to:

   ```plaintext
   terminate called after throwing an instance of 'std::logic_error'
     what():  /home/hu4/Codes/Trilinos/trilinos/packages/ifpack2/adapters/thyra/Thyra_Ifpack2PreconditionerFactory_def.hpp:122:
   
   Throw number = 1
   
   Throw test that evaluated to true: !(this->isCompatible(*fwdOpSrc))
   
   Error!
   ```
3. In the `"Preconditioner Types"` list, add the following block:

   ```plaintext
         <ParameterList name="Ifpack">
           <Parameter name="Prec Type" type="string" value="Amesos"  />
           <Parameter name="Overlap" type="int" value="1"  />
           <ParameterList name="Ifpack Settings">
             <Parameter name="amesos: solver type" type="string" value="Pardiso"  />
           </ParameterList> <!--Ifpack Settings-->
         </ParameterList> <!--Ifpack-->
   ```

   This will set the additive Schwarz overlap to 1 (highly recommended) as well as enable the Pardiso sparse direct solver.
</details>
<details>
<summary>Launching VertexCFD with multithreading</summary>

The ability of VertexCFD to use shared-memory parallelism (i.e., multithreading) depends strongly on the solver and preconditioner selection. At the current time, the Pardiso sparse direct solver is the only option that is able to use multithreading in a meaningful way. Using multithreading (instead of exclusively MPI-based parallelism) allows the linear solver to maintain larger subdomains when constructing a preconditioner, leading to smaller iteration counts from the linear solver and greater robustness. The impact on _runtime_ is problem-dependent. On Narsil (or other Slurm/sbatch systems), the following modifications can be made to an `sbatch` submission script to enable multithreading:

1. Request multiple CPU cores per MPI task. This is accomplished using the `--ntasks-per-node` and `--cpus-per-task` options in the sbatch resource request. For example, the following lines will request 32 MPI ranks per node with 4 threads per rank (thus using 128 CPU cores per node):

   ```
   #SBATCH --ntasks-per-node 32
   #SBATCH --cpus-per-task 4
   ```

   In general, the number of tasks per node multiplied by the number of CPUs per task should equal the number of physical CPU cores on a node.
2. Set the number of OpenMP tasks per rank. Both Trilinos and Pardiso use OpenMP for multithreading. The number of threads per task can be set by including the following lines in the submission script anywhere before the executable is launched:

   ```
   export OMP_PROC_BIND=true
   export OMP_PLACES=cores
   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
   ```

   The first two lines are optional, but will silence warnings from Kokkos. The final line tells OpenMP to use all of the cores allocated by Slurm for multithreading.
3. Set the MPI process binding for launching the executable. Depending on how Slurm is configured, the default assignment of threads to CPU cores may not work well (multiple threads may be placed on the same physical core, leading to significantly reduced performance). On Narsil, the following block will launch a job with correct binding of threads:

   ```
   # Special case for single MPI rank per node
   if [ ${SLURM_NTASKS_PER_NODE} -eq 1 ]
   then
     MAP_STRING="ppr:1:node:pe=${SLURM_CPUS_PER_TASK}"
   else
     # Compute MPI mapping -- assuming a node has 2 sockets
     NUM_SOCKETS=2
     let "TASKS_PER_SOCKET = ${SLURM_NTASKS_PER_NODE} / ${NUM_SOCKETS}"
     MAP_STRING="ppr:${TASKS_PER_SOCKET}:socket:pe=${SLURM_CPUS_PER_TASK}"
   fi

The full sbatch submission script is included below.

</details>   

[sbatch_submission_script](uploads/using-epetra-and-pardiso-solvers-in-vertexcfd/sbatch_submission_script)

[vertexcfd-pardiso-narsil.sh](uploads/using-epetra-and-pardiso-solvers-in-vertexcfd/vertexcfd-pardiso-narsil.sh)

[trilinos-pardiso-narsil.sh](uploads/using-epetra-and-pardiso-solvers-in-vertexcfd/trilinos-pardiso-narsil.sh)

[Home page](https://code-int.ornl.gov/vertex/vertex-cfd/-/wikis/home)
