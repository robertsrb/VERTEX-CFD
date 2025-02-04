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

Boundary conditions are weakly imposed by computing numerical flux at the boundaries' provided boundary values. The boundary conditions implemented in VERTEX-CFD are listed below:

The boundary conditions implemented in VERTEX-CFD are listed below:

- **Periodic boundary**
- **Dirichlet with time-transient variation**
- **Symmetry for isothermal flow**
- **No-slip for viscous flow**
- **Rotating wall for isothermal flow**
- **Laminar flow**
- **Outflow with back pressure**
- **Conducting and isolating wall**

The vector solution is denoted by $$U_{bc} = (\phi_{p,{bc}}, \mathbf{u}_{bc}, T_{bc}, \varphi_{bc})$$ at the boundary. It should be noted that when the energy equation and the electric potential equation are not solved, the temperature $$T_{bc}$$ and the electric potential $$\varphi_{bc}$$ are ignored.

### Dirichlet boundary
The Dirichlet boundary condition denotes the Dirichlet boundary condition in VERTEX-CFD. The velocity is set equal to the user-specified values or Dirichlet values $$\mathbf{u}_D$$ while the Lagrange pressure and the boundary gradients are set to the interior values. The temperature is also set to a user-specified value $T_{bc}$. Linear ramping in time is also available and can be used to vary each primitive variable independently.

$$
\begin{equation}
\left\{ \
\begin{matrix}
    u_{i,bc}(\mathbf{r}, t) = u_{i,D}(t) \\
    \phi_{p,{bc}}(\mathbf{r}, t) = \phi_p(\mathbf{r}, t) \\
    \partial_i U_{bc}(\mathbf{r}, t) = \partial_i U(\mathbf{r}, t) \\
    T_{bc}(\mathbf{r}, t) = T_{D}
\end{matrix}
\right.
\end{equation}
$$

### Symmetry boundary condition
The symmetry boundary condition is a no-penetration condition. The normal component of the fluid velocity to the wall is zero while the tangential component is unrestricted. The same observation is valid for the temeprature gradient as well. Assuming the outward normal vector to a wall boundary is denoted by $$\mathbf{n}_{bc} = \left(n_{bc,x}, n_{bc,y}, n_{bc,z} \right)$$ in a three-dimensional computational domain, the boundary condition for the primitive variables reads:

$$
\begin{equation}
\left\{
\begin{matrix}
    \phi_{bc}(\mathbf{r}, t) &=& \phi(\mathbf{r}, t) \\
    \mathbf{u}_{bc}(\mathbf{r}, t) &=& \mathbf{u}(\mathbf{r}, t) - \left( \mathbf{u}(\mathbf{r}, t) \cdot \mathbf{n}_{bc} \right) \mathbf{n_{bc}} \\
    T_{bc}(\mathbf{r}, t) &=& T(\mathbf{r}, t) \\
    \varphi_{bc}(\mathbf{r}, t) &=& \varphi(\mathbf{r}, t)
\end{matrix}
\right.
\end{equation}
$$

The boundary gradients are function of the interior values:

$$
\begin{align}
    \partial_i U_{bc}(\mathbf{r}, t) =& \partial_i U(\mathbf{r}, t) \nonumber\\
    \partial_i T_{bc}(\mathbf{r}, t) =& \partial_i T(\mathbf{r}, t) - \left(\partial_i T(\mathbf{r}, t) \cdot \mathbf{n}_{bc} \right) n_{i} \\
    \partial_i \varphi_{bc}(\mathbf{r}, t) =& \partial_i \varphi(\mathbf{r}, t) - \left(\partial_i \varphi(\mathbf{r}, t) \cdot \mathbf{n}_{bc} \right) n_{i} \nonumber
\end{align}
$$


## Initial conditions
