---
title: 'VERTEX-CFD: A multiphysics platform for fusion applications'
tags:
  - Computational Fluid Dynamics
authors:
  - name: Marco
    affiliation: 1
    corresponding: true
affiliations:
  - name: Nuclear Energy and Fuel Cycle Division, Oak Ridge National Laboratory
    index: 1
date: 19 December 2024
bibliography: paper.bib
---

# Summary:

The demand for high-performance computational fluid dynamics and multiphysics software packages has grown in recent years as a response to efforts in complex engineering and research applications. While the widespread deployment of high-performance computing (HPC) resources has enabled larger, more complex simulations to be conducted, few commercial or open-source software packages are available which scale performantly on various computing architectures, and represent the multitude of physical processes relevant to these applications. The VERTEX initiative at Oak Ridge National Laboratory was created to address this technical gap, with a special emphasis on high-fidelity multiphysics modeling of coupled turbulent fluid flow, heat transfer, and magnetohydrodynamics for applications in fusion and fission energy, isotope separation and enrichment, and other spaces. The VERTEX-CFD module was developed to solve the governing equations of these problems using a high-order continuous Galerkin finite element framework, employing an artificial compressiblity method for pressure-velocity coupling and fully-implicit monolithic solvers. Special attention was paid during the development process to ensure performance portability across both CPU and GPU computing platforms. In this work, we present the results of a comprehensive verification and validation (V&V) suite designed to assess the accuracy and convergence behavior of the VERTEX-CFD module for problems involving laminar, heated flows. Comparisons were made to closed-form analytical solutions as well as experiments to ensure both numerical and physical accuracy. The performance and scaling behavior of the software was examined for a representative problem on both CPU and GPU architectures, up to the scale of thousands of compute nodes. These analyses demonstrate the capabilities of the VERTEX-CFD, build trust in the solver accuracy, and provide the basis for future work.


# Statement of need

The core work of the cross-cutting VERTEX Laboratory Directed Research and Development (LDRD) initiative aims to create a new multiphysics simulation framework supporting physical phenomena key to Oak Ridge National Laboratory (ORNL) mission-critical challenges. ORNL has clearly demonstrated needs in modeling and simulation of gas dynamics, rarefied flow, plasma-surface interaction, electromagnetics, magneto-hydrodynamics (MHD), and thermal hydraulics for conducting fluids, collisionless and collisional plasma, and structural mechanics. The project is organized into four technical areas: VERTEX-CORE, VERTEX-MAXWELL, VERTEX-CFD, and VERTEX-CLOSURE. Each area focuses on a specific physics while relying on the common interface VERTEX-CORE.
As part of the VERTEX initiative, the primary mission of the VERTEX-CFD team is to develop modeling and simulation capabilities to accurately model the physics in fusion blanket design. It thus requires a multiphysics solver to implement the incompressible Navier-Stokes (NS) equation to conjugate a heat transfer model and an MHD solver. Solvers, finite element methods, and other relevant tools are provided by the [Trilinos package](https://trilinos.github.io/) \cite{trilinos-website}. The VERTEX-CFD solver is designed to scale on HPC platforms by leveraging Kokkos programming language to ensure compatibility with various CPU and GPU architectures.
The long term objectives of the VERTEX initiative is to faciliate the addiiton of new physical models by relying on a plug-and-play architecture, and also guarentee the correctness of the implemented model over time. New physics and equations are easily added to the global tree and allow for quick deployment of new physical model on HPC platforms. Such approach can only be made possible by setting clear requirements and review process for all developers contributing to the project code: any changes and aditions to the source code is reviewed and tested before being merged. VERTEX-CFD solver is tested daily on a continuous integration (CI) workflow that is hosted on ORNL network.


# Current capabilities

VERTEX-CFD solver is still under active development and currently implement the following capabilities: incompressible Navier-Stokes equations, temperature equation, induction-less and full-induction MHD models, RANS turbulence models and WALE (LES) turbulence model. Each new physics is implemented in closure models with unit tests. Physical models and coupling between equations were verified and validated against bechmark problems taken from the published literature: isothermal flows, heated flows, transient and steady-state cases, turbulent cases. VERTEX-CFD solver has demonstrated second-order temporal and spatial accuracy. Scaling of the VERTEX-CFD solver was assessed on CPUs and GPUs architecture. It was found that strong and weak scaling were comparable to other CFD solvers alike NekRS. (ADD FIGURE).


# Conclusions

VERTEX-CFD is a CFD solver that relies on a finite element discretization method to solve for the incompressible Navier-Stokes equations coupled to a temperature equation and an electric potential equation. Reynolds Averaged Navier-Stokes turbulence models and large eddy simulation model are also available. The code relies on the  Trilinos package and offers a wide range of temporal integrators, solvers and preconditioners to run on CPU- and GPU-enabled platforms. The code was verified and validated for steady and unsteady incompressible flows with benchmark cases taken from the published literature: natural convection, viscous heating, laminar flow over a circle, and turbulent channels. It was also demonstrated that VERTEX-CFD solver scales on CPUs (Perlmutter) and GPUs (Perlmutter and Summit) architectures.


# Acknolegements

This work was funded by the Laboratory Directed Research and Development (LDRD) program at Oak Ridge National Laboratory, and the Scientific Discovery through Advanced Computing (SciDac) programm.


# Disclaimer

This manuscript has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the [DOE Public Access Plan](http://energy.gov/downloads/doe-public-access-plan).
