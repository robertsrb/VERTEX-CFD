[Home page](https://code-int.ornl.gov/vertex/vertexcfd/-/wikis/home) -> [Run VertexCFD](run-vertexcfd.md)

Once VertexCFD is installed, an incompressible flow in a 2D channel is available in `installation_path/examples/incompressible/incompressible_2d_channel.xml`. The input file relies on a `xml` format. The example can be run with the command line `installation_path/bin/vertexcfd --i=incompressible_2d_channel.xml` and should produce the following output:

```
Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
  In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
  For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
  For unit testing set OMP_PROC_BIND=false

============================================================================
Time Integration Begin
Thu Mar 31 21:46:09 2022

  Stepper = Backward Euler
  Simulation Time Range  [0, 0.2]
----------------------------------------------------------------------------

Time Step = 1; Order = 1
CFL = 1.000e+00; dt = 6.186e-03; Time = 0.00000e+00
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   6.65e-02
           1   2.03e-02          1   6.45e-16
           2   4.66e-04          1   4.25e-16
           3   5.40e-07          1   5.36e-16
           4   3.19e-13          1   6.58e-16
Time step time to completion (s): 5.97e+00

Time Step = 2; Order = 1
CFL = 1.000e+00; dt = 5.644e-03; Time = 6.18583e-03
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   5.73e-02
           1   5.72e-03          1   4.68e-16
           2   5.40e-05          1   4.01e-16
           3   3.47e-09          1   7.18e-16
Time step time to completion (s): 8.96e+00

Time Step = 3; Order = 1
CFL = 1.000e+00; dt = 5.385e-03; Time = 1.18299e-02
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   5.37e-02
           1   4.14e-03          1   4.17e-16
           2   3.35e-05          1   3.47e-16
           3   2.01e-09          1   5.54e-16
Time step time to completion (s): 7.03e-01

Time Step = 4; Order = 1
CFL = 1.000e+00; dt = 5.249e-03; Time = 1.72150e-02
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   5.28e-02
           1   3.20e-03          1   4.06e-16
           2   1.69e-05          1   4.46e-16
           3   2.15e-10          1   4.99e-16
Time step time to completion (s): 1.19e+00
... ...
... ...
... ...
... ...
Time Step = 39; Order = 1
CFL = 1.000e+00; dt = 4.853e-03; Time = 1.88877e-01
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   3.59e-02
           1   1.16e-03          1   3.55e-16
           2   1.69e-06          1   3.86e-16
           3   8.47e-13          1   3.27e-16
Time step time to completion (s): 1.21e+01

Time Step = 40; Order = 1
CFL = 1.000e+00; dt = 4.853e-03; Time = 1.93731e-01
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   3.57e-02
           1   1.15e-03          1   3.89e-16
           2   1.68e-06          1   3.90e-16
           3   1.66e-12          1   3.20e-16
Time step time to completion (s): 1.73e+00
    41 *  (dt = 4.852e-03, new = 1.417e-03)  Adjusting dt to hit final time.

Time Step = 41; Order = 1
CFL = 1.000e+00; dt = 1.417e-03; Time = 1.98583e-01
 | Nonlinear | F 2-Norm | # Linear | R 2-Norm |
           0   3.56e-02
           1   3.11e-04          1   2.95e-16
           2   3.67e-08          1   2.73e-16
           3   4.40e-16          1   2.86e-16
Time step time to completion (s): 6.91e+00

----------------------------------------------------------------------------
Total runtime = 1.45e+02 sec
              = 2.42e+00 min
              = 0.04 hr
Thu Mar 31 21:51:51 2022
Time integration complete.
============================================================================
```

Once the job completed, the numerical solution can be read from the Exodus file `incompressible_2d_channel_solution.exo` using ParaView, through NoMachine client. ParaView is available on NARSIL at`/projects/vertex/opt/paraview/5.11.0/bin/paraview`.

![](uploads/run-incompressible-2d-channel/temperature-profile-paraview.png)*Visualization of the temperature profile predicted by VertexCFD with ParaView.*
