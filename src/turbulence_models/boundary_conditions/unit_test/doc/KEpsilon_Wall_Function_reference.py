import numpy as np

vel = [1.5, -3.0, 4.5]
grad_k = [0.02, 0.04, 0.06]
grad_e = [0.03, 0.06, 0.09]
normals = [0.33, 0.66, 0.99]

tke_list = [0.25, 2.5]
eps = 3.1

yplus = 11.06
C_mu = 0.09
nu = 1e-4
kappa = 0.41

dim_list = [2, 3]

for dim in dim_list:
    print("Computing K-Epsilon wall function BCs in ", dim, "D\n")

    vel_dim = vel[:dim]
    normals_dim = normals[:dim]
    grad_k_dim = grad_k[:dim]
    grad_e_dim = grad_e[:dim]

    mag_u = np.linalg.norm(np.array(vel_dim))

    print("\tVelocity magnitude: ", mag_u, "\n")

    print("\tLimiting tke = ", pow(mag_u / yplus / pow(C_mu, 0.25), 2.0), "\n")

    for tke in tke_list:
        print("\tTesting tke value = ", tke, "\n")

        u_tau = max(pow(C_mu, 0.25) * pow(tke, 0.5), mag_u / yplus)

        print("\t\tu_tau = ", u_tau, "\n")

        nu_t = kappa * yplus * nu

        print("\t\tNeumann nu_t = ", nu_t, "\n")

        nu_t = C_mu * pow(tke, 2.0) / eps

        print("\t\tDirichlet nu_t = ", nu_t, "\n")

        e_bnd = pow(u_tau, 4.0) / nu_t

        print("\t\tboundary e for dirichlet condition: ", e_bnd)

        boundary_grad_k = grad_k_dim - np.array(np.dot(
            grad_k_dim, normals_dim)) * normals_dim
        boundary_grad_e = grad_e_dim + np.array(kappa * pow(u_tau, 5.0) / pow(
            nu_t, 2.0) - np.dot(grad_e_dim, normals_dim)) * normals_dim

        for d in range(dim):
            print("\t\tk boundary gradient component ", d, " = ",
                  boundary_grad_k[d], "\n")
            print("\t\te boundary gradient component ", d, " = ",
                  boundary_grad_e[d], "\n")

    print("\n")
