---
parent: VERTEX-CFD v1.0 User Guide
title: Theory
nav_order: 2
usemathjax: true
---

# Theory

---

## Governing equations

VERTEX-CFD implements the incompressible Navier-Stokes equations, a temperature equation, and a magneto-hydrodynamics (MHD) equation (induction-less equation).

$$
\nabla \cdot (\vec{u}) &= 0
\partial_t \vec{u} + \vec{u} \cdot \nabla \vec{u} &= - \nabla p + \nu \nabla^2 \vec{u} + \vec{f}_b \tag{1} \\
\partial_t T + \vec{u} \cdot \nabla T &= \alpha \nabla^2 T + S_T \tag{2} 
$$


## The discretized equations

The governing equations are discretized with a finite element method (FEM). The resulting ordinary differential equations (ODEs) are integrated with a fully implicit temporal integrators from the Tempus package. 

## Boundary conditions

## Initial conditions
