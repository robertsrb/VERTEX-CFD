# Summary:

The demand for high-performance computational fluid dynamics and multiphysics software packages has grown in recent years as a response to efforts in complex engineering and research applications. While the widespread deployment of high-performance computing (HPC) resources has enabled larger, more complex simulations to be conducted, few commercial or open-source software packages are available which scale performantly on various computing architectures, and represent the multitude of physical processes relevant to these applications. The VERTEX initiative at Oak Ridge National Laboratory was created to address this technical gap, with a special emphasis on high-fidelity multiphysics modeling of coupled turbulent fluid flow, heat transfer, and magnetohydrodynamics for applications in fusion and fission energy, isotope separation and enrichment, and other spaces. The VERTEX-CFD module was developed to solve the governing equations of these problems using a high-order continuous Galerkin finite element framework, employing an artificial compressiblity method for pressure-velocity coupling and fully-implicit monolithic solvers. Special attention was paid during the development process to ensure performance portability across both CPU and GPU computing platforms. In this work, we present the results of a comprehensive verification and validation (V\&V) suite designed to assess the accuracy and convergence behavior of the VERTEX-CFD module for problems involving laminar, heated flows. Comparisons were made to closed-form analytical solutions as well as experiments to ensure both numerical and physical accuracy. The performance and scaling behavior of the software was examined for a representative problem on both CPU and GPU architectures, up to the scale of thousands of compute nodes. These analyses demonstrate the capabilities of the VERTEX-CFD, build trust in the solver accuracy, and provide the basis for future work.


# Statement of need and development plan

The core work of the cross-cutting VERTEX Laboratory Directed Research and Development (LDRD) initiative aims to create a new multiphysics simulation framework supporting physical phenomena key to Oak Ridge National Laboratory (ORNL) mission-critical challenges. ORNL has clearly demonstrated needs in modeling and simulation of gas dynamics, rarefied flow, plasma-surface interaction, electromagnetics, magneto-hydrodynamics (MHD), and thermal hydraulics for conducting fluids, collisionless and collisional plasma, and structural mechanics. The project is organized into four technical areas: VERTEX-CORE, VERTEX-MAXWELL, VERTEX-CFD, and VERTEX-CLOSURE. Each area focuses on a specific physics while relying on the common interface VERTEX-CORE.
As part of the VERTEX initiative, the primary mission of the VERTEX-CFD team is to develop modeling and simulation capabilities to accurately model the physics in fusion blanket design. It thus requires a multiphysics solver to implement the incompressible Navier-Stokes (NS) equation to conjugate a heat transfer model and an MHD solver. Solvers, finite element methods, and other relevant tools are provided by the [Trilinos package](https://trilinos.github.io/) \cite{trilinos-website}. The VERTEX-CFD solver is designed to scale on HPC platforms by leveraging Kokkos programming language to ensure compatibility with CPU and GPU architectures.

# Mathematics

# Conclusions

# Acknolegements

## [CPU BUILD INSTRUCTIONS](docs/install-vertexcfd/install-vertexcfd-on-narsil-cpu.md)

## [GPU BUILD INSTRUCTIONS](docs/install-vertexcfd/install-vertexcfd-on-narsil-gpu.md)

## [RUNNING CASES WITH VERTEX-CFD](docs/run-vertexcfd/run-incompressible-channel.md)
