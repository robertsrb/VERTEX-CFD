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
