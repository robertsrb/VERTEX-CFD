# Building VERTEX-CFD on NARSIL CPU architecture
These instructions detail the process for building VERTEX-CFD for use on the NARSIL mod cluster CPU nodes. Similar instructions for a GPU build can be found [here](./install-vertexcfd-on-narsil-gpu.md).

## Initial Build
First, create project directories to clone the source repository and compile the code:
```
cd
mkdir -p projects/build
cd projects
```

Then, clone the source code from the internal ORNL GitLab repository. Enter your credentials if prompted.

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

Once the environment is configured, you are ready to build. Change to the build directory to begin:

```
cd ~/projects/build
```

Now, copy the build scripts from the repository. You will use the CPU version of the build script for the CPU nodes.

```
cp ../vertex-cfd/scripts/build/vertex_narsil_cpu_release.sh .
```

Ensure that `SOURCE` and `INSTALL` within the build script point to the desired source and installation directories, then execute the script.

```
./vertex_narsil_cpu_release.sh
```

This will copy the necessary files to your build directory. Now, the source code is ready to be compiled and installed. Note that you will use `ninja` instead of `make` for compilation on the CPU node.

```
ninja
ninja install
```

If the code compiles without error, it is ready to use. Executables are located within the `bin` directory of the build location. The `build` directory should be reserved for testing, while production runs should be carried out in `install` or another location.

## Rebuilding After Changes
After making changes or pulling updates from the head repository, you will again recompile the code from the build directory.

```
cd ~/projects/build
```

If any files have been created or removed since the previous build, rerun the build script:

```
./vertex_narsil_cuda_release.sh
```

Then, compile and install:

```
ninja
ninja install
```

If no errors are given, your code has successfully rebuilt with the desired changes and is ready to use. **Prior to pushing any changes to the repository, be sure to run the formatting tool:**

```
ninja format
```

## [RUNNING CASES WITH VERTEX-CFD](../run-vertexcfd/run-vertexcfd.md)
