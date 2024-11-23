import numpy as np


def np_array_to_cpp_str(my_arr):
    my_str = np.array2string(my_arr, separator=', ')
    my_str = my_str.replace("[", "{")
    my_str = my_str.replace("]", "}")
    return my_str


def get_conducting_bc_exp_val(norm, u_bnd, b, b_bnd, b_ext, grad_b, psi,
                              psi_bnd, mu, eta, build_magn_corr, dir_magn_pot,
                              build_resistive):
    b_dot_n = np.dot(b - b_bnd, norm)
    grad_b_dot_n = []
    for i in range(len(grad_b)):
        grad_b_dot_n.append(np.dot(grad_b[i], norm))

    exp_b = b
    for i in range(len(b)):
        if (i < len(norm)):
            exp_b[i] -= b_dot_n * norm[i]

    if build_resistive:
        # i-loop over the magnetic field dimensions
        for i in range(len(grad_b)):
            # j-loop over the gradient/spatial dimensions
            for j in range(len(grad_b[i])):
                if i != j:
                    grad_b_dot_n[i] += mu / eta * norm[j] * (
                        u_bnd[i] * (b_ext[j] + b_bnd[j]) - u_bnd[j] *
                        (b_ext[i] + b_bnd[i]))

    exp_grad_b = grad_b
    for i in range(len(grad_b)):
        for j in range(len(grad_b[i])):
            if (j < len(norm)):
                exp_grad_b[i][j] -= grad_b_dot_n[i] * norm[j]

    print("exp_b =\n", np_array_to_cpp_str(exp_b), ";\n")
    print("exp_grad_b =\n", np_array_to_cpp_str(exp_grad_b), ";\n")
    if build_magn_corr:
        exp_psi = psi
        if dir_magn_pot:
            exp_psi = psi_bnd
        print("exp_psi =", exp_psi, ";\n")


def get_case_exp_vals(n_grad, n_space, build_magn_corr, dir_magn_pot,
                      build_resistive):
    u_bnd = np.array([3.0, -4.0, 5.0])
    b = np.array([1.25, 2.5, 3.75])
    b_bnd = np.array([1.1, 2.2, 3.3])
    b_ext = np.array([-0.05, 0.025, -0.0125])
    norm = np.array([.45, -.65, .35])
    grad_b = np.array([[11.0, -5.5, 2.75], [-6.0, 3.0, -1.5],
                       [3.25, -1.625, 0.8125]])
    psi = 4.4
    psi_bnd = 5.5
    mu = 0.12
    eta = 3.6

    print("\nFull Induction Conducting Wall Case:\n")
    print("\tnum_grad_dim = ", n_grad)
    print("\tnum_space_dim = ", n_space)
    print("\tbuild_magn_corr = ", build_magn_corr)
    print("\tdirichlet_scalar_magn_pot = ", dir_magn_pot)
    print("\tbuild_resistive_flux = ", build_resistive, "\n")

    get_conducting_bc_exp_val(norm[:n_grad], u_bnd[:n_space], b[:n_space],
                              b_bnd[:n_space], b_ext[:n_space],
                              grad_b[:n_space, :n_grad], psi, psi_bnd, mu, eta,
                              build_magn_corr, dir_magn_pot, build_resistive)


get_case_exp_vals(2, 2, False, False, False)
get_case_exp_vals(2, 2, False, False, True)
get_case_exp_vals(2, 2, True, False, False)
get_case_exp_vals(2, 2, True, False, True)
get_case_exp_vals(2, 2, True, True, False)
get_case_exp_vals(2, 2, True, True, True)

get_case_exp_vals(3, 3, False, False, False)
get_case_exp_vals(3, 3, False, False, True)
get_case_exp_vals(3, 3, True, False, False)
get_case_exp_vals(3, 3, True, False, True)
get_case_exp_vals(3, 3, True, True, False)
get_case_exp_vals(3, 3, True, True, True)
