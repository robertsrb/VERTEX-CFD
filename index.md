VERTEX-CFD is a performance portable software package for computational fluid dynamics (CFD) simulations on CPU or GPU architectures, built upon the [Trilinos](https://trilinos.github.io/) numerical library.

The core work of the cross-cutting VERTEX Laboratory Directed Research and Development (LDRD) initiative aims to create a new multiphysics simulation framework supporting physical phenomena key to Oak Ridge National Laboratory (ORNL) mission-critical challenges. ORNL has clearly demonstrated needs in modeling and simulation of gas dynamics, rarefied flow, plasma-surface interaction, electromagnetics, magneto-hydrodynamics (MHD), and thermal hydraulics for conducting fluids, collisionless and collisional plasma, and structural mechanics. The project is organized into four technical areas: VERTEX-CORE, VERTEX-MAXWELL, VERTEX-CFD, and VERTEX-CLOSURE. Each area focuses on a specific physics while relying on the common interface VERTEX-CORE.
As part of the VERTEX initiative, the primary mission of the VERTEX-CFD team is to develop modeling and simulation capabilities to accurately model the physics in fusion blanket design. It thus requires a multiphysics solver to implement the incompressible Navier-Stokes (NS) equation to conjugate a heat transfer model and an MHD solver. Solvers, finite element methods, and other relevant tools are provided by the [Trilinos package](https://trilinos.github.io/) \cite{trilinos-website}. The remainder of this paper presents current capabilities of the VERTEX-CFD package and provides a timeline for future work.

## [CPU BUILD INSTRUCTIONS](docs/install-vertexcfd/install-vertexcfd-on-narsil-cpu.md)

## [GPU BUILD INSTRUCTIONS](docs/install-vertexcfd/install-vertexcfd-on-narsil-gpu.md)

## [RUNNING CASES WITH VERTEX-CFD](docs/run-vertexcfd/run-incompressible-channel.md)
