# Validated Capabilities

This is the authoritative capability summary. **V** means implemented and tested
for the stated formulation, **U** means unsupported/rejected, and **N** means not
validated (not a support claim). A coefficient table alone does not establish DAE
applicability. The evidence links describe the actual equations and limitations.
Project-relative evidence links require the parent checkout; library test links
remain usable in a standalone module checkout.

## Orders And Controllers

Dense order is the reported differential/global interpolation order. Zero means
no supported extension/embedding. RK4/SSP fallbacks use endpoint Hermite where
needed; DOP853 uses its own combined error estimator, not an ordinary embedding.
`test_pde_capabilities` exports coefficient metadata for the parent documentation
contract check. `current` means the accepted BDF order, including startup at one.

F = fixed steps; E = embedded/combined-estimate control (MATLAB, SciPy, digital
filters); B = BDF's own estimate/history feeding those controllers. Pseudo-time
merit control is separate from physical-time accuracy control and can operate
without an embedded estimator. Supported does not mean all controllers are
optimal for every stiff or constrained formulation.

| Method | Order | Embedded | Dense | Controllers | Evidence |
| --- | --- | --- | --- | --- | --- |
| EE | 1 | 0 | 1 | F | [RK](../tests/test_explicit_rk.cpp) |
| HE | 2 | 1 | 2 | F,E | [RK](../tests/test_explicit_rk.cpp) |
| RK23 | 3 | 2 | 3 | F,E | [dense](../tests/test_explicit_dense_output.cpp) |
| RK33SSP | 3 | 2 | 3 | F,E | [tables](../tests/test_butcher_tables_explicit.cpp) |
| RK43SSP | 3 | 2 | 3 | F,E | [tables](../tests/test_butcher_tables_explicit.cpp) |
| RK4 | 4 | 0 | 3 | F | [dense](../tests/test_explicit_dense_output.cpp) |
| RK64SSP | 4 | 3 | 3 | F,E | [tables](../tests/test_butcher_tables_explicit.cpp) |
| RK45 | 5 | 4 | 4 | F,E | [FSAL](../tests/test_explicit_fsal.cpp) |
| DOP853 | 8 | 0 | 7 | F,E | [adaptive](../tests/test_explicit_dop853_adaptive.cpp), [dense](../tests/test_explicit_dop853_dense_output.cpp) |
| IE | 1 | 0 | 1 | F | [DIRK](../tests/test_implicit_dirk.cpp) |
| IM | 2 | 0 | 2 | F | [dense](../tests/test_implicit_dense_output.cpp) |
| CN | 2 | 0 | 2 | F | [ESDIRK](../tests/test_implicit_esdirk.cpp) |
| SDIRK2(1)2 | 2 | 1 | 2 | F,E | [adaptive](../tests/test_implicit_adaptive.cpp) |
| ESDIRK2(1)3 | 2 | 1 | 2 | F,E | [adaptive](../tests/test_implicit_esdirk_adaptive.cpp) |
| SDIRK3(1)3 | 3 | 1 | 3 | F,E | [tables](../tests/test_butcher_tables_implicit.cpp) |
| ARK324L2SA | 3 | 2 | 3 | F,E | [adaptive](../tests/test_implicit_adaptive.cpp) |
| IMEX_EULER | 1 | 0 | 0 | F | [IMEX](../tests/test_implicit_imex.cpp) |
| IMEX_HEUN_TR2 | 2 | 0 | 0 | F | [IMEX](../tests/test_implicit_imex.cpp) |
| IMEX_ARS233 | 3 | 0 | 0 | F | [IMEX](../tests/test_implicit_imex.cpp) |
| IMEX_ARS222 | 2 | 0 | 0 | F | [IMEX](../tests/test_implicit_imex.cpp) |
| IMEX_ARK324L2SA | 3 | 2 | 3 | F,E | [adaptive](../tests/test_implicit_imex_adaptive.cpp), [dense](../tests/test_implicit_imex_dense_output.cpp) |
| ROS2 | 2 | 1 | 2 | F,E | [orders](../tests/test_implicit_rosenbrock.cpp) |
| ROS3 | 3 | 2 | 3 | F,E | [orders](../tests/test_implicit_rosenbrock_ros3.cpp) |
| ROS34PW2 | 3 | 2 | 3 | F,E | [W](../tests/test_implicit_rosenbrock_w.cpp) |
| ROSI2P2 | 3 | 2 | 3 | F,E | [index 2](../tests/test_implicit_rosenbrock_index2.cpp), [moving](../tests/test_implicit_rosenbrock_index2_moving.cpp) |
| BDF1 | 1 | 0 | 1 | F,B | [orders](../tests/test_implicit_bdf_orders.cpp) |
| BDF2 | 2 | 0 | 2 | F,B | [orders](../tests/test_implicit_bdf_orders.cpp) |
| BDF | 1-5 | 0 | current | B | [order/history](../tests/test_implicit_bdf_variable_order.cpp), [dense](../tests/test_implicit_bdf_dense.cpp) |

ROSI2P2's row reports the published table metadata. Its selected index-2 path
uses separate constraint-aware estimator weights with controller exponent two;
velocity order three does not imply order-three raw moving multipliers.

## Formulations

| Formulation | Status and restrictions | Evidence |
| --- | --- | --- |
| Explicit identity/nonsingular mass | V: `evaluate_rate` returns M^-1*f; mass solve and accuracy are problem-owned | [rate contract](../README.md#explicit-rk), [PDEs](../../../../source/nonlinear_operators/pde_validation/README.md) |
| DIRK/BDF variable nonsingular mass | V: shifted residual/Jacobian include mass derivatives; zero-diagonal stages need a consistent rate | [mass/DAE](../tests/test_implicit_mass_dae.cpp), [BDF](../tests/test_implicit_bdf_adaptive_mass_dae.cpp) |
| DIRK/BDF singular mass | V for tested consistent index-1/index-2 formulations and suitable methods; no singular mass inverse | [DAEs](../tests/test_implicit_mass_dae.cpp), [BDF DAEs](../tests/test_implicit_bdf_mass_dae.cpp), [mixed CH](../../../../source/nonlinear_operators/pde_validation/tests/test_cahn_hilliard.cpp) |
| IMEX nonsingular mass | V: both partitions use the same mass/state convention and independent contexts | [mass](../tests/test_implicit_imex_mass.cpp) |
| Reduced convex-split CH IMEX | V for concentration-only, constant mobility, periodic unforced double-well CH with IMEX_EULER; first order, mass conservation and energy dissipation; not general singular-mass IMEX | [formulation](../../../../source/nonlinear_operators/cahn_hilliard/README.md), [accuracy](../../../../source/nonlinear_operators/pde_validation/tests/test_cahn_hilliard_imex.cpp), [energy](../../../../source/nonlinear_operators/pde_validation/tests/test_cahn_hilliard_energy.cpp) |
| General singular-mass IMEX | U: no general constrained partition-rate reconstruction | [execution limits](../README.md#partition-execution) |
| Rosenbrock constant nonsingular mass | V: rate form or selected `mass_form`; exact Jacobian for ROS2/ROS3 | [mass](../tests/test_implicit_rosenbrock_mass.cpp), [time](../tests/test_implicit_rosenbrock_mass_nonautonomous.cpp) |
| Rosenbrock variable nonsingular mass | V in normalized rate form, including DM and M_t terms; not physical variable-mass `mass_form` | [variable mass](../tests/test_implicit_rosenbrock_variable_mass.cpp) |
| ROS34PW2 index-1 singular mass | V: autonomous constant mass, exact algebraic derivatives, coupled shifted solves; dense output U | [DAE](../tests/test_implicit_rosenbrock_dae.cpp), [guards](../tests/test_implicit_rosenbrock_dae_lifecycle.cpp) |
| ROSI2P2 index-2 singular mass | V for the gauged constant-mass exact-Jacobian Stokes-like formulation; moving constraints use clock augmentation | [derivation](rosenbrock_index2.md), [dense](../tests/test_implicit_rosenbrock_index2_dense.cpp) |
| General variable singular-mass Rosenbrock DAEs | U; broader nonlinear/moving constraints N | [limits](rosenbrock_mass.md) |
| Projection | V for identity-mass/rate formulations, fixed linear homogeneous P and documented compatibility conditions; void/bool apply; identity bypassed | [dispatch](../tests/test_implicit_operator_projection.cpp), [Stokes/Oseen](../../../../source/nonlinear_operators/stokes/README.md) |
| Nonlinear/moving projection or wall-pressure splitting | N; existing linear periodic projection is not a general Navier-Stokes method | [scope](../../../../source/nonlinear_operators/stokes/README.md) |
| Constrained dense samples | V only with problem-owned, validated reconstruction and gauge handling; raw interpolation is not constraint enforcement | [DIRK](../tests/test_implicit_dense_mass_dae.cpp), [BDF](../tests/test_implicit_bdf_dense_mass_dae.cpp) |

Scalar evidence here is real double precision. General AD, arbitrary precision,
and complex-valued timestepper/controller support are N in this audit. No claim
is inferred from scalar/vector traits alone.

## Backend And Application Evidence

Parent CI supplies required GCC/Clang serial/OpenMP runs, ASan/UBSan/leak checks,
header-isolation checks and capability-gated CUDA/HIP jobs. GPU jobs emit explicit
execution/skip records: an unavailable device is not a successful GPU validation.
HIP-on-NVIDIA and native AMD HIP are different execution evidence. Inspect the
uploaded compiler, runtime, architecture and result records for the exact run;
no documentation row asserts that every CI runner has every accelerator.

Large backend tests reject vector-sized host/device copies and check retained
workspace after warmup: [BDF](../tests/test_implicit_bdf_backend.cpp),
[Rosenbrock](../tests/test_implicit_rosenbrock_backend.cpp),
[IMEX](../tests/test_implicit_imex_backend.cpp), and the project PDE tests.
Scalar diagnostics/reductions and initial geometry setup are permitted. Workspace
is bounded by stages/order/Krylov basis, not trajectory length. Guard runs are
correctness checks, not equal-accuracy performance comparisons.

The project owns spatial discretization, PDE equations, solver assembly and
periodic-orbit matching. Library stepping support does not establish prepared
shooting derivatives. Project explicit/DIRK/BDF derivative evidence remains
separate; IMEX/Rosenbrock prepared shooting derivatives are not claimed here.
KS bifurcation references do not validate KS time integration. Broader CH
mobilities/boundaries, convex-split singular-mass IMEX, cavity Navier-Stokes and NLS remain separate
problem-integration stages.
