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


## The discretized equations

The governing equations are discretized with a finite element method (FEM). The resulting ordinary differential equations (ODEs) are integrated with a fully implicit temporal integrators from the Tempus package.

VERTEX-CFD employs a finite element discretization method and high-order implicit temporal integrators to integrate partial differential equations (PDEs). Numerical stability of the solution is ensured by the use of L-stable implicit temporal integrator and the use of appropriate mesh density.

## Boundary conditions

## Initial conditions
