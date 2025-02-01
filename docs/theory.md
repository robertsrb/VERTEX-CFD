---
parent: VERTEX-CFD v1.0 User Guide
title: Theory
nav_order: 2
usemathjax: true
---

# Theory

---
VERTEX-CFD is a computational fluid dynamics (CFD) solver being developed at Oak Ridge National Laboratory to model fusion blanket designs. This multiphysics problem requires a robust and fast solver that implements the incompressible Navier--Stokes equations, heat transfer and conjugate heat transfer, turbulence models, and magneto-hydrodynamic equations. The VERTEX-CFD solver relies on Trilinos and its subsequent package for high-order temporal integrators and its discretization method and solver options. Both CPU and GPU hardware are supported with the Kokkos templated C++ package. Because of the multiphysics aspect of the project, which involves a wide range of spatial and temporal scales, Trilinos was chosen to rely on an implicit solver and to select numerical methods that scale well on high-performance computing systems.

## Governing equations

VERTEX-CFD implements the incompressible Navier-Stokes equations, a temperature equation, and a magneto-hydrodynamics (MHD) equation (induction-less equation).

$$
\begin{align}
    \partial_t \vec{u} + \vec{u} \cdot \nabla \vec{u} &= - \nabla p + \nu \nabla^2 \vec{u} + \vec{f}_b \tag{1} \\
    \partial_t T + \vec{u} \cdot \nabla T &= \alpha \nabla^2 T + S_T \tag{2}
\end{align}
$$


## The discretized equations

## Boundary conditions

## Initial conditions
