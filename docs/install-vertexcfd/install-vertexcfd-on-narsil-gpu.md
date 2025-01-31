# Building VERTEX-CFD on GPU architecture
These instructions detail the process for building VERTEX-CFD in a CUDA-enabled form for use on the mod cluster GPU nodes. Similar instructions for a basic CPU build can be found [here](./install-vertexcfd-on-narsil-cpu.md).

## Initial Build
First, be sure you are logged into a GPU node to build the code:

```
ssh narsil-gpu-login
```

Enter your login credentials when prompted. Then, create project directories to clone the source repository and compile the code:
```
cd
mkdir -p projects/build
cd projects
```

Next, clone the repository from the internal ORNL GitLab repository. Enter your credentials if prompted.

```
git clone git@code-int.ornl.gov:vertex/vertex-cfd.git
```

Any edits made to the code will be done in the source directory created by the clone operation. However, the code will be compiled in the `build` directory which was created in the previous step.

The repository contains a configuration file which loads the required modules to compile and use VERTEX-CFD. This file can be sourced via `~/projects/vertex-cfd/scripts/ci/vertexcfd-env.sh`, however it is often more convenient to copy this file to your home directory:

```
cp ~/projects/vertex-cfd/scripts/ci/vertexcfd-env.sh ~/.vertexcfd-env
```

The environment for VERTEX-CFD can then be configured with:

```
source ~/.vertexcfd-env
```

For GPU compilation, you will also have to load the CUDA module:

```
module load cuda/cuda-11.4
```

Once the environment is configured, you are ready to build. Change to the build directory to begin:

```
cd ~/projects/build
```

Now, copy the build scripts from the repository. You will use the CUDA version of the build script for the GPU-enabled solver.

```
cp ../vertex-cfd/scripts/build/vertex_narsil_cuda_release.sh .
```

Ensure that `SOURCE` and `INSTALL` within the build script point to the desired source and installation directories, then execute the script.

```
./vertex_narsil_cuda_release.sh
```

This will copy the necessary files to your build directory. Now, the source code is ready to be compiled and installed. Note that you will use `make` instead of `ninja` for compilation on the GPU node. Be sure to use the `-j` flag to build in parallel, but do not use all available resources on the login node.

```
make -j 32
make install
```

If the code compiles without error, it is ready to use. Executables are located within the `bin` directory of the build location. The `build` directory should be reserved for testing, while production runs should be carried out in `install` or another location.

## Rebuilding After Changes
After making changes or pulling updates from the head repository, you will again recompile the code from the build directory.

```
cd ~/projects/build
```

If any files have been created or removed since the previous build, rerun the build script:

```
./vertexcfd_narsil_cuda_release.sh
```

Then, compile and install:

```
make
make install
```

If no errors are given, your code has successfully rebuilt with the desired changes and is ready to use. **Prior to pushing any changes to the repository, be sure to run the formatting tool:**

```
make format
```

## [RUNNING CASES WITH VERTEX-CFD](../run-vertexcfd/run-vertexcfd.md)
