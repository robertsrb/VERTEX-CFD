---
parent: VERTEX-CFD v1.0 User Guide
title: Theory
nav_order: 2
usemathjax: true
---

# Theory

---

## Governing equations

VERTEX-CFD implements the incompressible Navier-Stokes equations, a temperature equation, and a magneto-hydrodynamics (MHD) equation (induction-less equation). Coupling between the different equations is ensured by the Buoyancy force and the Lorentz force.

$$
\begin{align}
\left\{
\begin{matrix}
    \nabla \cdot \mathbf{u} = 0 \\
    \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho}\nabla P + \nu \Delta \mathbf{u} + \frac{1}{\rho}\mathbf{J} \times \mathbf{B^0} - \mathbf{g} \beta (T - T_0) \\
    \rho C_p \left( \partial_t T + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k \nabla T ) + q^{'''} \\
    \mathbf{J} = \sigma ( -\nabla \varphi + \mathbf{u} \times \mathbf{B^0} ) \\
    \nabla \cdot (\sigma \nabla \varphi) = \nabla \cdot [ \sigma \mathbf{u} \times \mathbf{B^0} ]
\end{matrix}
\right.
\end{align}
$$

The equations are recast in a conservative form and solved for the pressure $$P$$, the velocity $$\mathbf{u}$$, the temperature $$T$$, and the electric potential $$\varphi$$.

$$
\begin{align}
\left\{
\begin{matrix}
    \nabla \cdot \mathbf{u} = 0 \\
    \partial_t \rho \mathbf{u} + \rho (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla P + \rho \nu \Delta \mathbf{u} + f^L - \rho \mathbf{g} \beta (T - T_0) \\
    f^L = \mathbf{J} \times \mathbf{B^0} = \sigma \left( -\nabla \varphi \times \mathbf{B^0} + (\mathbf{B} \cdot \mathbf{u}) \cdot \mathbf{B^0} - ||\mathbf{B^0}||^2 \mathbf{u} \right) \\
    \rho C_p \left( \partial_t T + \mathbf{u} \cdot \nabla T \right) = \nabla \cdot (k \nabla T ) + q^{'''} \\
    \nabla \cdot (\sigma \nabla \varphi) = \nabla \cdot [ \sigma \mathbf{u} \times \mathbf{B^0} ]
\end{matrix}
\right.
\end{align}
$$


## The discretized equations

VERTEX-CFD employs a finite element discretization method and high-order implicit temporal integrators to integrate partial differential equations (PDEs). Numerical stability of the solution is ensured by the use of L-stable implicit temporal integrator and the use of appropriate mesh density.

## Boundary conditions

Boundary conditions are weakly imposed by computing numerical flux at the boundaries' provided boundary values.

## Initial conditions
