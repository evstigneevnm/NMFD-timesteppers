# NMFD Timesteppers

Header-only C++17 time integration in `nmfd::time_steppers`. Public headers live
under `include/nmfd/time_steppers`; simulations and verification programs are in
`tests`. The parent project's `source/time_stepper/legacy` remains frozen and is
not a dependency of this library.

Vector ownership uses `nmfd::detail::vector_wrap` from NMFD-operations. Scratch
vectors and RK stages are allocated once and reused. Explicit integration requires
only NMFD-operations and SCFD, not NMFD-newton or NMFD-linsolvers.

## Flat Development Builds

Use sibling checkouts `../nmfd-operations` and `../scfd`, or set
`NMFD_OPERATIONS_ROOT` and `SCFD_ROOT`. JSON is needed only by the Lorenz simulation;
set `JSON_INCLUDE` to a directory containing `nlohmann/json.hpp` (default `../json`).
Python comparisons additionally require NumPy, SciPy, and Matplotlib.

```sh
make -j2 test_cpu check_headers_cpu
make -j2 test_cuda check_headers_cuda CUDA_ARCH=75
make format-check
```

There are no nested dependency submodules or module-specific CI workflows during
this development stage. The parent project runs its existing separate timestepper
jobs against these sources and flat canonical dependencies. Standalone dependency
pins and module CI will be introduced later. Future implicit methods receive their
nonlinear solver by template injection; they must not introduce a Newton dependency
for explicit-only users.

## Explicit RK

- `runge_kutta/butcher_tables.h`: EE, HE, RK23, RK33SSP, RK43SSP, RK4, RK64SSP, RK45,
  DOP853,
  plus separately tested implicit/IMEX coefficient tables.
- `runge_kutta/explicit_time_step.h`: stage evaluation and bounded retry loop.
- `integration/time_step_adaptation_constant.h`: fixed proposed step magnitude.
- `integration/time_step_adaptation_matlab.h`: component-scaled embedded error
  control, first-rejection formula, repeated halving, and no growth after rejection.
- `integration/time_step_adaptation_scipy.h`: additive tolerance scaling, RMS error
  assessment, and SciPy-style power-law step selection with bounded growth/shrinkage.
- `integration/time_step_adaptation_filter.h`: seven digital-filter presets with
  committed error/step history, additive RMS assessment, and bounded startup/retries.
- `integration/time_integrator.h`: interval traversal, external management, commit.

An explicit problem supplies `bool evaluate_rate(state, rate)` and optionally `set_time(t)`.
The rate is already mass-inverted/projected when the formulation requires it.
There is no implicit-residual reinterpretation and no general singular-mass solve.
Return `false` on a failed rate solve; the stepper rejects/retries without publishing
a candidate. Nonfinite results are rejected separately; exceptions propagate.

The single-step method returns a pending candidate. The integrator calls
`finalize()` after external processing; failed candidates never replace the
committed output. `apply()` on the integrator starts a fresh trajectory.
Construction and `reset()` report `idle`. A successful single-step `apply()`
reports `converged`; `finalize()` retains that numerical result status.
External managers own their own reset policy and receive the actual step interval.
Constructors retain vector operations, problem, adaptation, step, and optional
external manager through `std::shared_ptr`. Required dependencies cannot be null.
Loggers, where used, remain borrowed `Log*` pointers and must outlive their users.
Problem objects and external managers use the same shared ownership for retained
dependencies. Callback time references are valid only during the current step's
external processing; continuous-integration handles remain borrowed views.
Evaluation methods borrow vectors; internal stages still use `vector_wrap`.
Do not share mutable instances between concurrent trajectories or create owner cycles.

Primary and embedded orders are separate. The error vector includes the step
size, and `error_order()` is its scaling exponent. `SDIRK3(1)3` retains the legacy
embedded weights, correctly identified as first order. RK64SSP retains published
decimal precision, with a table-specific validation tolerance. Missing stage
times are derived from row sums. Autonomous behavior belongs to the problem.

Table names:
- `RK23` is Bogacki-Shampine 3(2), stored with four derivatives including the
  endpoint derivative. Its embedded state error scales as `h^3`.
- `RK45` is Dormand-Prince 5(4), stored with seven stages.
- `RK4` is classical four-stage, fourth-order Runge-Kutta, without an embedded
  estimator or FSAL. Use constant adaptation; embedded-error controllers reject it.
- `DOP853` supplies the eighth-order primary formula, stored with twelve primary
  stages plus the endpoint derivative, and a combined E5/E3 estimator with error
  exponent eight. Use constant or SciPy adaptation. Its native seventh-order dense
  output adds three optional stages.
- `SDIRK2(1)2`, `ESDIRK2(1)3`, and `SDIRK3(1)3` use `p(q)s`: primary
  order, embedded order, total stage count. `IE`, `IM`, and `CN` are unchanged.
- IMEX names are `IMEX_EULER`, `IMEX_HEUN_TR2` (Heun plus trapezoidal),
  `IMEX_ARS233`, `IMEX_ARS222`, and the embedded `IMEX_ARK324L2SA` pair.
  ARS digits count implicit stages, explicit
  stages, and combined order as in the original paper, not padded table dimensions.
The previous names are retained only by the frozen legacy implementation.

Coefficient sources: [SSP embedded pairs](https://imrefekete.web.elte.hu/files/2022_FCS_JCAM.pdf),
[Bogacki-Shampine method](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/rk.py),
[DOP853 coefficients](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/dop853_coefficients.py),
[Dormand-Prince method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html).
Rooted-tree tests check nonlinear order, not just a scalar stability polynomial.

These IMEX tables are executed by `runge_kutta/imex_time_step.h`.
Adaptive initialization uses the configured initial step and bounded rejection,
not an extra initial derivative evaluation.

## Implicit RK

`runge_kutta/implicit_time_step.h` executes lower-triangular implicit RK tables,
including IE, IM, CN, SDIRK2(1)2, ESDIRK2(1)3, SDIRK3(1)3, ARK324L2SA and
user-supplied DIRK tables. The constructor accepts a method name through `params`,
or a `butcher_table` directly. Coupled IRK is rejected; entirely explicit tables
belong to the explicit engine.

The problem owns `previous_state()` and `known_rhs()` buffers and supplies:

```cpp
void set_time(scalar_type time);
void set_affine_shift(scalar_type a, scalar_type b);
void apply(const vector_type& state, vector_type& residual);
void set_linearization_point(const vector_type& state);
std::shared_ptr<const jacobi_operator_type> get_jacobi_operator();
```

The buffers have mutable and const reference accessors and must not alias one
another, caller states, or solver scratch. The residual is
`a*M(t,u)*(u-previous)-b*(f(t,u)+known_rhs)`. Its Jacobian includes derivatives of
a state-dependent mass. Trial residual calls must not alter frozen Jacobian data.

The stepper balances differential rows using `params::reference_time` (default one):
`s=max(abs(alpha),1/reference_time)`, `a=alpha/s`, `b=1/s`, and
`G=a*M(t,u)*(u-previous)-b*(f(t,u)+known_rhs)`. The problem evaluates this form directly,
not by multiplying an already formed large residual. Its Jacobian uses the same
coefficients, including the mass derivative. `set_affine_shift(a,b)` stores them
unchanged for the entire Newton solve and its backtracking trials.
Algebraic constraints and their Jacobian rows remain unscaled. General singular
mass formulations must supply their own appropriate equation partition/scaling;
the timestepper does not infer it from a mass-matrix diagonal or invert the mass.

This balances small and large stage steps, including negative shifts and the
zero-shift steady limit without changing Newton or solver tolerances.
It is not a globalization controller or a timestep-aware nonlinear
accuracy budget. Future globalization must measure progress in the original
physical residual and constraints, not compare scaled norms across stage shifts.

For stage i the stepper writes `previous=u_n+h*sum(a_ij*k_j)` and `known_rhs=0`,
then sets `time=t_n+c_i*h`. Nonzero diagonals use `alpha=1/(h*a_ii)` and the
existing stage residual. The supplied nonlinear solver is
bound once through `set_nonlinear_operator(problem)` and called with
`solve(stage_state)`. It must not be rebound or shared between concurrent
trajectories. No Newton headers are needed by the stepper itself.

Nonzero-diagonal stages require no mass inversion. For singular-mass DAEs, provide consistent initial
data and a suitable coupled stage solver/preconditioner. The tests use stiffly
accurate methods, committing the final stage directly to preserve its algebraic
constraints. The DAE regressions include a nonlinear index-1 system and a small
index-2 Stokes saddle system with a pressure multiplier. Other DIRK formulas are
not automatically appropriate for every DAE.
An external manager modifying a DAE endpoint must preserve/reconstruct consistency.

Zero-diagonal stages have their state already determined by `previous`. If their
rate is consumed by another stage, the endpoint, or the configured estimator,
the problem must additionally provide this optional method:

```cpp
bool evaluate_rate(const vector_type& state, vector_type& rate);
```

The two-argument `apply()` remains exclusively a residual evaluation.
In `evaluate_rate()`, use the time supplied by `set_time()` and
produce the physical derivative satisfying `M(t,state)*rate=f(t,state)`, not the
raw RHS or the shift-scaled residual. Ignore affine-shift data and do not modify
the input, stage buffers or prepared Jacobian. The problem owns any mass solver,
its accuracy settings and reusable backend workspace. Return `false` on numerical
failure; the stepper discards the rate, shrinks/retries when allowed, and leaves
the caller's output unchanged on failure. Nonfinite rates are rejected separately;
backend/configuration exceptions propagate. Missing rate support is rejected
during construction when the selected table/controller needs it. Unused dummy
rates need no provider. No rates are cached across attempts or committed steps.

For singular mass, provide a formulation-specific complete consistent derivative,
including differentiated constraints and gauges where required. The tests supply
the moving index-1 constraint's nonzero algebraic rate, but deliberately refuse
ESDIRK for the existing index-2 Stokes problem without such a provider. This is
not automatic support for arbitrary DAEs. Explicit stepping uses the same rate
contract, including for dense-output evaluations.

`ARK324L2SA` is the implicit four-stage Kennedy-Carpenter ARK3(2)4L[2]SA pair:
third-order primary and second-order embedding, with controller exponent three.
Coefficients follow the high-precision representation in
[SUNDIALS](https://github.com/LLNL/sundials/blob/main/src/arkode/arkode_butcher_dirk.def).
Rooted-tree tests verify both orders. CN has no embedded estimator and continues
to refuse error-requiring controllers instead of inventing an estimate.

The separate `test_implicit_esdirk`, `test_implicit_esdirk_adaptive` and
`test_implicit_esdirk_mass_dae` executables check residual/rate separation, stage
times (including interior explicit stages), rejection/rollback, primary and
embedded orders, all filter presets, MATLAB/SciPy adaptation, coupled/state/time-
dependent mass, consistent constraints, backward integration and restart. Parent
CI covers serial/OpenMP, capability-gated CUDA and sanitizers.

With dense output disabled, the stepper owns one stage-state vector and s-1 rates for stiffly accurate methods,
or s rates otherwise. An error-requiring controller with an embedded table adds
one state-error buffer and, when needed, the previously omitted endpoint rate.
Constant-step methods allocate neither addition. Dense-output storage is absent when disabled.
Scratch is reused across stages and trajectories; `apply()` publishes only a
successful candidate. The existing integrator commits through `finalize()`.
Failed solves report `failed_solver`; custom adaptation may retry via
`reject_step(dt)`, without assessing invalid candidates. Constant adaptation stops.
Configuration/backend exceptions propagate unchanged. Roundoff-close final steps
are clipped to the target to avoid an unresolved extra Newton solve.

Embedded pairs support `time_step_adaptation_matlab`,
`time_step_adaptation_scipy`, and `time_step_adaptation_filter`. The estimate is
the state error
`h*sum((b_i-bhat_i)*k_i)`; its exponent is `embedded_order()+1`, including two
for the first-order estimator of `SDIRK3(1)3`. The primary endpoint is still
committed, never the embedded state. Select the controller's error requirement
before constructing the stepper. Error-free custom controllers retain their
five-argument `assess()` interface. IE/IM report `error_estimate_unavailable`
with error-requiring controllers; no step-doubling estimator is implied.
Dense output is optional as described below.

`test_implicit_adaptive.cpp` checks independent scalar stage/error formulas,
primary and embedded orders, MATLAB/SciPy/PID tolerance refinement, nonlinear decay,
time/state-dependent mass, stiff forcing, repeated trajectories, failed/nonfinite
stages, minimum-step and attempt limits, modified endpoints, and allocation
failure cleanup. A moving index-1 constraint is checked at every committed
endpoint, in both time directions. This regression does not establish general
algebraic-error control or applicability to every singular-mass DAE.

The Lorenz, Rossler, Van der Pol, and mass/DAE regressions additionally run MATLAB,
SciPy, and all seven filter presets with SDIRK2(1)2 and SDIRK3(1)3 at two
tolerances. They check independent
finite-time references, tolerance refinement, rejected-step accounting, exact
target times, and repeatable trajectories. Classical Lorenz uses `epsilon=0`;
the fixed-step damped reference remains separate. Stiff Van der Pol covers fast
relaxation and early slow motion, not a full relaxation cycle. DAE constraints
are checked independently at every accepted endpoint, including a nonzero-time
restart for the moving manifold. Parent CI covers serial CPU, OpenMP, available
CUDA devices, and CPU address/undefined-behavior sanitizers.
The DIRK tests also cover coefficient bounds, nonunit reference times, nonzero
known RHS, invalid-configuration rollback, and the physical steady-limit balance.
The Lorenz test replays the captured roundoff-limited stage and crosses `t=0.404`
with the unchanged strict Newton settings. Mass/DAE tests check Jacobians across
both scaling regimes and verify that deliberately violated algebraic constraints
remain visible; actual small, large, and backward stages retain consistency.

```sh
make -j2 test_implicit_cpu
make -j2 test_implicit_cuda CUDA_ARCH=75
```

These tests additionally use sibling `../nmfd-newton` and `../nmfd-linsolvers`
checkouts (override `NMFD_NEWTON_ROOT` and `NMFD_LINSOLVERS_ROOT`). Default explicit
test targets still require neither solver library. Parent CI runs separate jobs
for DIRK order, mass/DAE, lifecycle, Lorenz, Rossler, and regular/stiff Van der Pol
on CPU/OpenMP and capability-gated CUDA.

## Interface Migration And IMEX Assembly

Rename explicit RHS providers from `void apply(state,rate)` to
`bool evaluate_rate(state,rate)`, returning success after the problem-owned mass
solve. Replace one-argument implicit setters with `set_affine_shift(a,b)` and
apply both coefficients consistently to differential residual/Jacobian rows.
Integrator, controller, commit and nonlinear-solver interfaces are unchanged.

`runge_kutta/imex_operator.h` provides a shared bundle for IMEX execution:

```cpp
using imex_operator_type = runge_kutta::imex_operator<explicit_type, implicit_type>;
std::shared_ptr<imex_operator_type> get_imex_operator(); // application-owned, cached
```

Its `explicit_part` supplies `evaluate_rate()` for `M*k_E=f_E`.
Its `implicit_part` supplies the shifted residual/Jacobian and optionally
`evaluate_rate()` for `M*k_I=f_I`. Rates exclude the other partition and stage
forcing. Retain shared model data without a back-pointer to the bundle/application;
bind solvers once at assembly.

## BDF

`bdf/implicit_time_step.h` supplies `BDF1` and `BDF2` through the same four
template dependencies and constructor arguments as implicit RK. Nested `params`
selects the method (default `BDF2`), maximum attempts and `reference_time`.
No problem, integrator or nonlinear-solver interface changes are required.

For BDF1, `previous_state()` is `u_n` and `alpha=1/h`. For equal-step BDF2 it is
`(4*u_n-u_{n-1})/3` and `alpha=3/(2*h)`. `known_rhs()` is zero. The existing
balanced `set_affine_shift(a,b)` contract applies, including the state-dependent
mass derivative and unscaled algebraic constraints. No rate or mass inverse is
required. `get_order()` reports the actual order used by the latest attempt.

BDF2 starts with backward Euler. Only an unchanged full `finalize(accepted,...)`
promotes history. Failed/rejected candidates and commits never advance it;
modified/shortened endpoints restart at BDF1. A changed actual step size or
direction also uses BDF1, then BDF2 resumes when an equal-step pair is available.
Roundoff-only endpoint clipping does not spuriously lower the order. Discontinuous
`set_time()` and `reset()` clear history; reset after independent state/model
changes. As with RK, the caller sets the next starting time after finalization.

Fixed-step workspace is one backend vector for BDF1 and three for BDF2: candidate, pending
input and older committed state. It is reused for in-place stepping, retries and
repeated trajectories. There are no vector-sized host transfers or per-step
allocations in the engine. Fixed mode remains the default; error-requiring
controllers return `error_estimate_unavailable` unless `params.adaptive=true`.

Adaptive mode uses variable-grid BDF coefficients and scaled committed divided
differences. With `r=h/h_previous`, BDF2 sets
`alpha=(1+2*r)/((1+r)*h)` and
`previous_state()=u_n+r*r/(1+2*r)*(u_n-u_{n-1})`.
The default usable step-ratio interval is `[0.2,2]`, configurable through
`minimum_step_ratio` and `maximum_step_ratio` (growth cannot exceed 2).
Growth is capped; extreme shrinkage or direction reversal restarts history.

For error-requiring controllers, startup estimates the full backward-Euler
candidate against two half steps, with error `2*(full-two_half)`. The full
candidate is retained. Subsequent estimates use polynomial defects corrected
for the computed candidate's contribution to the highest divided difference.
BDF2 begins after two committed intervals. The existing MATLAB/SciPy/filter
controllers receive state-unit errors with local-error exponents 2 and 3;
their interface is unchanged. Tightening local tolerances controls an estimate,
not an identical bound on accumulated global error.

Adaptive workspace with error estimation is four vectors for BDF1 and five for
BDF2, allocated once. Failed solves, rejected trials and failed commits do not
modify divided differences. Modified endpoints, independent model/state changes
and discontinuous time changes follow the same restart rules as fixed mode.
No mass inversion or `evaluate_rate()` is introduced. DAE error estimation remains
formulation-dependent; accepted algebraic constraints retain their own tests.

### Automatic Order 1-5

Opt in without changing the problem, solver, controller or integrator interfaces:

```cpp
step_type::params p;
p.method = "BDF";
p.adaptive = true;
p.maximum_order = 5;
```

`BDF` requires an error-requiring controller. `maximum_order` is 1-5 and applies
only to this mode; existing `BDF1`/`BDF2` behavior and defaults are unchanged.
Coefficients follow the actual grid, using a scaled Newton divided-difference
history. Start at order one, then raise by one after at least `max(3,q+1)` clean
commits at order `q`, provided sufficient history is available. Two failed trials
lower the retry order by one; four fall back to order one. Failed/rejected commits
never promote order, and modified endpoints restart both history and policy.
The current LTE is assessed once per trial with exponent `q+1`; the existing
controller alone owns error weighting and its timestep proposal.
Newton starts from the committed state, not the extrapolating polynomial: a small
predicted residual can conceal a larger pressure/state error in a poorly
conditioned constrained problem. The polynomial is still used for LTE estimation.
Residual tolerances must remain appropriate for the formulation and conditioning.

Automatic mode caps growth at 1.2 (or a smaller configured limit). Orders 3-5
require every adjacent ratio in their recent grid to lie in `[0.8,1.2]`;
otherwise the attempt uses at most order two. These are conservative grid rules,
not an optimal order selector or a stability-limit detector. BDF3-5 are not
A-stable; weakly damped modes need accuracy/stability validation for the problem.

History construction and each commit require O(qN) backend work. With a configured
cap `m`, automatic mode owns `3+m+(m>2)` vectors: candidate, pending input, error,
`m` divided differences and, for higher orders, one rotating scratch buffer.
This is nine vectors at cap five, allocated once, with no vector-sized host
transfers. No trajectory-sized history is retained. Cold-start error remains part
of the global error: formal order tests use exact histories separately, while
complete cold-start trajectories verify tolerance refinement. No general high-index
DAE accuracy guarantee is inferred from the manufactured constraint tests.

Run the serial/OpenMP and CUDA suites with `make test_bdf_cpu` and
`make test_bdf_cuda`. Separate formula, lifecycle, mass/DAE and backend tests
cover orders, stiff decay, non-autonomous times, backward integration, endpoint
modification, rollback, ownership and transfer guards. The DAE cases establish
capability for the tested formulations, not arbitrary DAE support.

### Dense Output

Append `true` as the fifth template argument to enable the existing borrowed
`get_continuous_integration()` view. Its `evaluate(theta,out)` accepts finite
`theta` in `[0,1]` and returns the sample time. It is available only for a
successful pending candidate and expires on finalize, reset or the next attempt;
the stepper must outlive the view. Endpoints are exact copies of saved states.

At the actual step order `q`, interpolate through the candidate and `q` old
nodes using the signed, nonuniform grid and scaled Newton differences. BDF1
is linear; BDF2 is quadratic. `dense_output_order()` reports the actual order,
including first-order startup/restart, not the configured cap or next proposal.
Accuracy includes startup, node and nonlinear-solve errors; degree alone does
not guarantee a global or arbitrary-DAE order. No extrapolation is provided.

Sampling is read-only O(qN) backend algebra, without allocations, rate/mass
solves, Newton calls or host transfers. Only fixed BDF1 needs an extra starting
snapshot; other modes reuse existing buffers. Disabled dense output preserves
the previous vector budgets. Shortened or externally modified commits invalidate
the view and restart history/order at BDF1; failed commits retain old history.
Raw interpolation need not satisfy nonlinear algebraic constraints. The
application reconstructs/validates constrained samples before event evaluation
or commitment; a reconstruction failure must not publish the candidate.

The visual runner supports `--sample-step` for BDF on ODE/nonsingular-mass
systems. Its metadata reports initial dense order one and
`dense_output_order_policy="used_bdf_order"`; actual automatic orders remain
in `orders_used`. Uniform DAE visualization still requires an explicit
problem-owned reconstruction policy. BDF section flights are covered, but
prepared BDF return-map derivatives are a separate implementation stage.

The visual runner accepts fixed and adaptive BDF cases without changing its defaults:

```sh
bash tests/run_all.sh --methods BDF1 BDF2 --controllers constant \
  --end 0.4 --step 0.0002
bash tests/run_all.sh --methods BDF1 BDF2 --controllers matlab scipy PID \
  --end 0.4 --rtol 1e-6 --atol 1e-8
bash tests/run_all.sh --methods BDF --bdf-max-order 5 --controllers matlab scipy PID \
  --end 0.4 --rtol 1e-7 --atol 1e-9
bash tests/run_all.sh --systems lorenz coupled_mass --methods BDF1 BDF2 BDF \
  --controllers scipy --end 0.4 --sample-step 0.017
```

## IMEX RK

`runge_kutta/imex_time_step.h` executes the fixed IMEX tables and `IMEX_ARK324L2SA` through
the ordinary stepper/integrator/controller interfaces:

```cpp
using step_type = runge_kutta::imex_time_step<Space, Problem, Controller, NonlinearSolver>;
auto step = std::make_shared<step_type>(space, problem, controller, solver,
                                       step_type::params{"IMEX_ARS222"});
```

The constructor calls `problem->get_imex_operator()` once, retains that shared
bundle (not the factory), and binds Newton to `implicit_part` once. The two
partitions must use the same state/mass convention and independent stage contexts.
Explicit `set_time(t)` is optional for autonomous providers; implicit `set_time(t)`
is required by the stage contract. No Newton headers are included by the engine.

At stage i, both partitions receive `t_i=t_n+c_i*h`. The engine writes
`previous=u_n+h*sum_{j<i}(aE_ij*kE_j+aI_ij*kI_j)` and `known_rhs=0`.
A nonzero implicit diagonal solves
`a*M(t_i,Y_i)*(Y_i-previous)-b*f_I(t_i,Y_i)=0`, with balanced
`a/b=1/(h*aI_ii)`, then recovers `kI_i=(Y_i-previous)/(h*aI_ii)` when consumed.
A consumed zero-diagonal implicit rate uses `implicit_part->evaluate_rate()`.
Consumed explicit rates use `explicit_part->evaluate_rate(Y_i,kE_i)`.
The endpoint combines both weight vectors; globally stiffly accurate pairs
reuse the final stage. Unused rates are neither evaluated nor allocated.

Mass solves and consistent partition derivatives remain problem-owned. Nonsingular
time/state-dependent coupled mass is tested; this does not establish general
singular-mass DAE support. No vector-sized data is transferred to the host by the engine.
Workspace is reused, and rates are rebuilt after rejection, reset or modified endpoints.
Failed rates/Newton stages follow bounded rejection without assessing or publishing
an invalid candidate; backend/configuration exceptions propagate.

`IMEX_ARK324L2SA` adds the Kennedy-Carpenter ARK3(2)4L[2]SA pair, using the
existing implicit table and the corresponding [SUNDIALS explicit coefficients](https://github.com/LLNL/sundials/blob/main/src/arkode/arkode_butcher_erk.def).
Its combined state error is
`h*sum((bE-bhatE)*kE+(bI-bhatI)*kI)`, formed before the controller's norm/scaling.
The primary order is three, the embedding order two, and the controller exponent
three. MATLAB, SciPy and all digital-filter controllers use their existing interfaces.
Only error-requiring configurations allocate the extra error vector and any
embedding-only rates. Fixed-step configurations retain their original storage.

Combined embedding metadata is explicit: the optional fourth argument of
`imex_butcher_table(e,i,order,embedded_order)` enables a validated paired estimate.
Separate partition embeddings do not automatically certify a combined one.
Custom pairs must satisfy the colored order conditions; tests cover the published
pair's primary and embedded coupling conditions. Old three-argument construction
is unchanged. An error-requiring controller without a paired estimate reports
`error_estimate_unavailable` before stage work.

`IMEX_ARK324L2SA` also opts into combined endpoint-Hermite dense output of order
three. Enable the fifth template argument and use the existing borrowed
`get_continuous_integration().evaluate(theta,out)` view. Other named IMEX tables
still reject `DenseOutput=true`; neither partition's interpolant is a combined one.
Fixed-step and error-free merit controllers retain their existing acceptance contracts.

`test_implicit_imex.cpp`, `test_implicit_imex_mass.cpp`, and
`test_implicit_imex_lifecycle.cpp` cover limiting/mixed orders, nonlinear and timed
splits, stiff tracking, mass Jacobians/solves, rollback, ownership and workspace reuse.
`test_implicit_imex_backend.cpp` runs matrix-free Newton/GMRES on 8192 unknowns
with vector-sized host transfers blocked, including adaptive runs.
`test_implicit_imex_adaptive.cpp` checks combined primary/embedded orders, error
coefficients, controller refinement, stiff tracking, backward integration and restart.
Mass and lifecycle tests also cover adaptive recovery, modified endpoints and
error-workspace reuse. Parent CI covers GCC/Clang CPU/OpenMP,
sanitizers and capability-gated CUDA/HIP.

```sh
make -j2 test_imex_cpu
make -j2 test_imex_cuda CUDA_ARCH=75
```

## Rosenbrock ROS2/ROS3

`rosenbrock/time_step.h` provides fixed/adaptive ROS2 and ROS3 for identity-mass
ODEs. Its separate `rosenbrock/coefficients.h` stores the increment-form
[ROS2 coefficients](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html#ros-2).
Select `params.method="ROS2"` (default, order 2/1) or `"ROS3"` (order 3/2).
Both are L-stable. ROS3 uses the full-precision constants from
[KPP's Ros3 routine](https://github.com/KineticPreProcessor/KPP/blob/main/int/rosenbrock.f90),
not the rounded documentation table. MATLAB, SciPy and all seven digital-filter
controllers use `sum(e_i*k_i)` with exponent two for ROS2, three for ROS3;
increments already include the step size. Estimation reuses RHS scratch.
`table()` returns a method-neutral `method_table` with `stage_count()`, `order()`,
`embedded_order()`, `error_order()` and zero-based coefficient accessors.
The original standalone ROS2 `coefficients` descriptor is unchanged, and
`table().gamma` remains available.

```cpp
using step_type = nmfd::time_steppers::rosenbrock::time_step<
    VectorOperations, Problem, TimeStepAdaptation, LinearSolver>;
```

Supply shared operations, problem, controller and linear solver handles. Reuse
the existing shifted-stage problem methods and `bool evaluate_rate(state,rate)`.
The solver supplies `set_operator(shared_jacobian)` and `bool solve(rhs,out)`;
no nonlinear solver is constructed, and public Rosenbrock headers do not include
NMFD-newton or a particular linear solver.
For each attempt, the engine sets the previous state to its input, zeros the
known RHS and freezes the exact Jacobian at the base time/state. All stage
solves use that same balanced shifted operator; stage-rate calls must preserve
its prepared data. ROS2 has two stages; ROS3 has three, reusing the second rate
in stage three. Both evaluate two rates. Changed steps always reprepare/rebind
the operator.

The existing apply/finalize/reset and controller interfaces remain unchanged.
Four backend workspace vectors for ROS2, five for ROS3, are allocated once, with no dense or trajectory
storage. Failures never publish partial output. Error-requiring controllers
without the extended error-assessment signature report `error_estimate_unavailable`.
`reference_time` defaults to one.
The default rate path requires the documented identity-rate shifted operator.
For autonomous constant nonsingular mass, select the residual form below.
DAEs are not supported. ROS2/ROS3 retain the exact-Jacobian
requirement. The separately selected W-method below permits approximations.

Autonomous problems need no additional method or storage. With
`params.autonomous=false`, the problem must additionally provide:

```cpp
bool evaluate_time_derivative(const vector_type& state, vector_type& out);
```

This is the partial derivative of the physical rate with respect to time at
fixed state, evaluated at the current `set_time()` value. It must not change
the state, stage buffers or prepared Jacobian. A false result requests bounded
rejection, nonfinite results are rejected, and exceptions propagate. Missing
support rejects construction; no finite-difference or zero approximation is
silently substituted. One additional backend vector retains the base derivative
for all stages. Stage two evaluates the rate at the endpoint for ROS2 or at
`t+gamma*h` for ROS3, using the actual clipped/backward step. Base time is
restored even on failure or exception. ROS3's third stage reuses that rate
without another evaluation or dedicated rate-cache vector.

```sh
make -j2 test_rosenbrock_cpu check_headers_cpu
make -j2 test_rosenbrock_cuda CUDA_ARCH=75
```

Separate tests cover analytical linear/nonlinear primary and embedded orders,
coupled GMRES/direct solves, stiff damping, small/large shifts, stage/rate
failures, bounded retries, backward steps, modified endpoints, ownership and
partial-construction cleanup. An 8192-component backend test guards both bulk
transfer paths and verifies fixed workspace with no allocations after warmup.
Adaptive tests cover all controllers, primary/embedded state differences,
tolerance refinement and repeated trajectories. Non-autonomous tests use a
manufactured exponential solution with stiff or time-varying Jacobians and
exercise partial derivatives, stage timing and failed-provider rollback.
The parent CI also runs Clang, sanitizers and available CUDA/HIP backends.

With the fifth template argument `true`, ROS2/ROS3 provide borrowed dense views
through the existing `get_continuous_integration().evaluate(theta,out)` contract.
Their native polynomial extensions report orders two/three and preserve both
endpoints exactly. Sampling performs backend algebra only; finalization and reset
expire views. ROS2 adds one snapshot vector and no rates/solves. ROS3 adds two
vectors, one endpoint rate and one solve using the same prepared operator, only
after error acceptance. Disabled dense output adds no backend workspace, rate
evaluations or solves.
Classical interpolation order is not a claim of stiffness-uniform accuracy or
positivity in unresolved initial layers. See the [derivation and limits](docs/rosenbrock_dense_output.md).

The ODE validation runner accepts both methods for identity-mass systems, without
changing its default method matrix or enabling unsupported mass/DAE cases:

```sh
bash tests/run_all.sh --systems lorenz rossler vdp vdp_stiff time_coefficient stiff_tracking \
  --methods ROS2 ROS3 --controllers scipy --end 0.4 --sample-step 0.017
```

### Rosenbrock/W ROS34PW2

Select `params.method="ROS34PW2"` for the four-stage third-order
[Rang/Angermann W-method](https://petsc.org/main/manualpages/TS/TSROSWRA34PW2/)
with a second-order embedded estimator. Its physical `evaluate_rate()` and
optional non-autonomous `evaluate_time_derivative()` contracts are unchanged.
The problem may supply `a*I-b*B` from `get_jacobi_operator()`, with an approximate
or problem-owned lagged Jacobian B frozen throughout each attempt. Preparation
and solver binding still occur on every attempt; changed timesteps must update
the shift even when B is reused. Linear-solve failures are still failures.

All existing controllers use the combined state defect with exponent three.
Without dense output, each successful attempt uses four rates, four solves and one operator binding.
Six backend workspace vectors are allocated once, seven for non-autonomous runs;
there is no per-step vector allocation or vector-sized host/device transfer.
Classical W order does not guarantee stability for an arbitrary approximation.
The physical rate and partial time derivative must not be approximated merely
because the stage Jacobian is approximate.

`table().supports_approximate_jacobian()` reports this validated capability.
`DenseOutput=true` enables a separately derived third-order W-compatible
continuous extension through the existing borrowed view. After error acceptance,
two extra solves prepare native slope increments with the same frozen operator;
only one additional endpoint rate is evaluated. Three additional backend vectors
retain the input and slopes; the first slope buffer also caches the original
base rate. Error-rejected trials skip dense-only solves and endpoint-rate work.
Sampling invokes no callbacks or solves, and finalize/reset expires views.
Endpoints are copied exactly and the primary/embedded formulas are unchanged.
This is not the ROS3 or published exact-Jacobian interpolant. See
[coefficient conversion, order conditions and limits](docs/rosenbrock_w_methods.md).
Separate W tests cover the original table, nonlinear and noncommuting systems,
zero/perturbed/lagged approximations, all controllers, non-autonomous backward
integration, fourth-stage failures, retry shifts and modified commits. They run
through the existing Rosenbrock CPU/OpenMP/CUDA targets and parent CI.
The visual runner accepts `--methods ROS34PW2 --sample-step 0.017` for the
same six identity-mass systems as ROS2/ROS3; default method selections are unchanged.
Separate dense tests check all eight independent W conditions, interior orders,
bounded stiff samples, exact endpoints, failed slope/rate preparation and view
expiry. Classical continuous order does not certify stiffness-uniform accuracy
or positivity in unresolved layers. Singular/time- or state-dependent mass
remains unsupported.

### Constant-Mass Residual Form

Set `params.mass_form=true` for autonomous `M*u'=f(u)` with constant nonsingular
M. All three tables support fixed/adaptive steps and their native dense output.
The default is false, preserving existing rate-based execution and costs.
No constructor, integrator, controller, solver or problem-method signatures change.
Mass form needs only the existing shifted residual `apply`, reference buffers,
and prepared Jacobian; `evaluate_rate` is not required or called.

The problem supplies `R(U)=a*M*(U-previous_state)-b*(f(U)+known_rhs)` and the
frozen operator `a*M-b*B`. ROS2/ROS3 require B=J_f; ROS34PW2 permits its validated
approximation. Stage residual calls must preserve the prepared operator.
The engine sets `known_rhs=0` and changes the reference buffer to encode the
mass-weighted stage coupling. It neither constructs/inverts M nor requires a
mass-action callback. See [derivation and limits](docs/rosenbrock_mass.md).

Mass form performs two/three/four residuals and shifted solves for
ROS2/ROS3/ROS34PW2. Enabled dense output adds zero/one/two residuals and solves
after error acceptance. Storage is unchanged, with no per-step vector allocations
or vector-sized host/device copies; sampling remains backend-only O(N) algebra.
Non-autonomous selection is rejected explicitly: the rate-time-derivative
callback is not a physical-force derivative. Singular and variable mass are
not validated capabilities, even if their shifted operators happen to solve.

## Non-Autonomous Verification

`test_explicit_nonautonomous.cpp` and `test_implicit_nonautonomous.cpp` use shared
analytical scalar models from `nonautonomous_test_problems.h`:

- `u'=t*u`, exact `exp(t*t/2)`: time-dependent RHS/Jacobian coefficients.
- `(1+t)*u'=-u`, exact `1/(1+t)`: explicitly time-dependent positive mass.
- `u'=-kappa*(u-sin(t))+cos(t)`, exact `sin(t)`: regular and stiff forcing.

The tests start at t=0.25, refine three step sizes, and repeat each trajectory.
Explicit tests audit every stage time and keep `h*kappa<=0.5` for stiff tracking.
Implicit tests check Jacobians at multiple times/shifts and include kappa=10000.
Classical order is checked above numerical roundoff/solve-error floors; stiff-limit
tracking checks accuracy and refinement without assuming uniform classical order.

`test_implicit_mass_dae.cpp` additionally verifies the time-dependent mass and
`x'=-x-z+sin(t)*x`, `0=z-sin(t)*x`. The exact constrained state is
`(exp(-t),exp(-t)*sin(t))`; external management checks the moving constraint at
accepted endpoints, including repeated trajectories starting at t=0.25.
Each new ODE test has separate parent CPU/OpenMP and capability-gated CUDA jobs.
The sanitizer job also includes both non-autonomous test executables.

## FSAL Reuse

Endpoint derivative reuse is opt-in for RK23, RK45, and the extended DOP853 table:

```cpp
step_type::params p;
p.method = "RK45";
p.reuse_endpoint_rate = true;
// operations, problem, and adaptation are shared_ptr instances.
step_type step(operations, problem, adaptation, p);
```

Consecutive inputs must follow the committed trajectory. After changing the RHS,
parameters, forcing, or input state independently, call `step.invalidate_rhs_cache()`
or `step.reset()`. Explicit invalidation preserves adaptation history and blocks
promotion of an already pending endpoint. An unchanged, full `accepted` commit
promotes the endpoint derivative by exchanging logical stage-buffer indices;
there are no additional state vectors or vector copies. Modified/shortened
endpoints, failed commits, resets, and time discontinuities discard reuse.
Internal retries retain a finite starting derivative, never a rejected endpoint.
Problems must provide deterministic RHS evaluations for fixed state, time, and
parameters; leave reuse disabled for evaluations with untracked mutable effects.
Dense-output views expire before the stage-buffer roles change.

`test_explicit_fsal_{cpu,cpu_omp,cuda}.bin` checks RHS counts, cached/uncached
agreement, memory, invalidation, retries, and external section refinement.
Separate `timestepper-fsal-cpu` and capability-gated `timestepper-fsal-cuda`
workflow jobs run these tests.

## Dense Output

Enable explicit interpolation using the fourth template argument; the default remains off:

```cpp
using step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<
    vector_operations_type, problem_type, adaptation_type, true>;
// In external_manager::apply(), before finalization:
auto dense = step.get_continuous_integration();
const auto sample_time = dense.evaluate(theta, sample); // theta in [0,1]
```

The manager can retain the stepper; each interpolation view borrows its pending step.
No changes to the problem or integration interface
are required. A view is valid only for its pending step; `finalize()`, `reset()`,
or the next `apply()` invalidate it, including when a newer step is pending.
The stepper must stay alive at the same address while a view is used.

| Method | Interpolant | Dense order | Extra vectors | Extra RHS calls per accepted step |
| --- | --- | --- | --- | --- |
| EE | Native linear | 1 | 1 | 0 |
| HE | Native quadratic | 2 | 1 | 0 |
| RK23 | Native cubic | 3 | 1 | 0 |
| RK45 | Shampine quartic | 4 | 1 | 0 |
| RK4 | Endpoint cubic Hermite | 3 | 2 | 1 |
| DOP853 | Native degree-seven polynomial | 7 | 4 | 3 |
| RK33SSP, RK43SSP, RK64SSP | Endpoint cubic Hermite | 3 | 2 | 1 |

Native extension coefficients are stored and validated in the Butcher table.
`dense_outout_stage_count()` counts all interpolant derivatives; `dense_outout_a()`
and `dense_outout_c()` describe ordinary and additional stage rows/times. DOP853
retains thirteen ordinary derivatives and prepares three additional derivatives
only after numerical acceptance, before publishing the pending candidate.
The initial-state snapshot doubles as stage-state scratch. No seven-vector
polynomial cache is needed, and failed extra stages follow the normal retry path.
The fallback is not advertised as a native SSP extension and does not guarantee
SSP/positivity preservation. `step.dense_output_order()` reports the actual order;
in particular RK64SSP remains fourth order at endpoints but has third-order dense
output. Embedded error control still controls the step, not a separate dense error.
Evaluation needs no new vectors or RHS calls; it reuses the pending stages and
endpoint. With dense output disabled, no dense vectors or extra RHS calls exist.
These interpolants apply to explicit ODE/reduced-flow formulations, not arbitrary
algebraic DAE states.

### Implicit Dense Output

Enable the fifth template argument; all existing four-argument uses remain valid:

```cpp
using step_type = nmfd::time_steppers::runge_kutta::implicit_time_step<
    vector_operations_type, problem_type, adaptation_type, nonlinear_solver_type, true>;
auto dense = step.get_continuous_integration(); // after successful apply(), before finalize()
const auto sample_time = dense.evaluate(theta, sample);
```

| Method | Extension | Dense order | Additional consistent-rate calls |
| --- | --- | --- | --- |
| IE | Linear | 1 | 0 |
| IM | Quadratic with initial derivative | 2 | 1 |
| CN, SDIRK2(1)2, ESDIRK2(1)3 | Quadratic using stored stages | 2 | 0 |
| SDIRK3(1)3 | Endpoint cubic Hermite | 3 | 1 |
| ARK324L2SA | Endpoint cubic Hermite, reusing the first stage | 3 | 0 |

The additional derivative is evaluated at `(t_n,u_n)` through the optional
`bool Problem::evaluate_rate(state, rate)`, including the appropriate mass solve
or complete consistent DAE derivative. Missing required support is rejected during
construction. Custom tables must provide their own continuous extension; no
unreported fallback is invented. Coefficients are checked against rooted-tree
continuous-order conditions, not merely scalar test equations.

Enabled storage is one work vector, all ordinary and additional extension rates,
one initial-state snapshot, and the error vector only when configured. This
retains the endpoint derivative otherwise omitted for stiffly accurate methods.
Extra rate stages run only after error acceptance, before publishing the candidate.
Failures/nonfinite rates follow bounded numerical rejection; backend exceptions
propagate. Preparation restores the final prepared stage time (the midpoint for
IM), without changing its reference/RHS buffers or Jacobian. Dense evaluation uses
stored rates only, allocates no vectors, and makes no problem calls or vector-sized
device/host transfers. Disabled storage and rate-call budgets are unchanged.

Views have the same pending-step lifetime as explicit RK, including rejection,
reset, modified/shortened endpoints and failed external commits. Sampling either
endpoint copies the saved numerical state exactly. Adaptive control still targets
the step error, not an independently estimated interior error.

### IMEX Dense Output

```cpp
using step_type = runge_kutta::imex_time_step<Space, Problem, Controller, NonlinearSolver, true>;
// Select IMEX_ARK324L2SA. After successful apply(), before finalize():
auto dense = step.get_continuous_integration();
auto time = dense.evaluate(theta, sample);
```

The cubic extension combines the initial/accepted states with total rates
`kE+kI` at those states. The first stage supplies the initial rates. Both partition
providers evaluate fresh rates at the actual combined endpoint only after error
acceptance; its last internal stage is generally a different state. Required mass
solves remain problem-owned. Failed/nonfinite endpoint rates enter normal bounded
rejection; exceptions propagate. Preparation restores both partition times without
rewriting their reference/RHS buffers or frozen Jacobian.

For ARK, enabling dense output adds one initial-state snapshot and one extra
`evaluate_rate()` call per partition per numerically accepted attempt. Retired
stage buffers hold the two total rates, so no extra rate vectors are allocated.
Sampling uses a fixed number of backend vector operations, no allocations, no
problem calls and no vector-sized host/device transfers. Disabling dense output
preserves the existing workspace and call budgets. Views have the same lifetime
and exact endpoint-copy semantics as the explicit/DIRK views above.

Combined metadata is opt-in: the fifth `imex_butcher_table` constructor argument
may select `dense_output_type::endpoint_hermite` for a validated pair of order at
least three with an initial-state first stage. Continuous colored-tree tests check
the combined extension through order three; independent partition extensions do
not enable it. This implementation covers ODE/nonsingular-mass formulations, not
general singular-mass DAEs. It controls endpoint error, not a separate dense error.

`test_implicit_imex_dense_output.cpp` checks forward/backward interior order,
partition limits, timed nonlinear splitting, mass systems and restart.
`test_implicit_imex_dense_lifecycle.cpp` covers view lifetime, actual endpoint
rates, failures, context restoration, bounded retries, optional capabilities and
allocation/release budgets. The 8192-unknown backend test blocks vector-sized host
transfers during dense preparation and sampling too.

For DAEs, raw polynomial samples need not satisfy algebraic constraints. The
application or external manager must perform formulation-specific reconstruction
at `sample_time` before constrained event evaluation/commit. The tests demonstrate
nonlinear and moving index-1 constraints with problem-owned reconstruction and
rollback on its failure. They do not establish arbitrary DAE algebraic order.
No additional mandatory problem or integration callbacks are introduced.

`test_implicit_dense_output`, `test_implicit_dense_lifecycle`, and
`test_implicit_dense_mass_dae` cover local/global interior order, non-autonomous
and backward sampling, unchanged adaptation, storage and construction failures,
view expiry, section refinement, mass systems and constrained reconstruction.
Parent CI runs these on serial/OpenMP, capability-gated CUDA and ASan/UBSan.

## Tests

Each Lorenz, Rossler, and Van der Pol test contains its own problem and assembly.
They use the legacy equations/initial values, fixed-step refinement, and adaptive
endpoint references computed using SciPy DOP853 and independently checked with
Radau (`rtol=3e-14`, `atol=1e-14`). SciPy is not needed to run the C++ tests.
The Lorenz problem includes the legacy cubic damping and asymmetry parameters.
Its stored-reference tests explicitly set `epsilon=0.0055`; the simulation uses
the example's current default (`epsilon=0` for classical Lorenz).

```bash
make test_rk_tables_explicit.bin test_rk_tables_implicit.bin \
     test_explicit_rk_cpu_omp.bin test_explicit_lorenz_cpu_omp.bin \
     test_explicit_rossler_cpu_omp.bin test_explicit_vdp_cpu_omp.bin
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_cpu_omp.bin
```

Use `_cpu.bin` for serial or `_cuda.bin` with the configured CUDA toolchain.
The parent workflow has separate CPU jobs for explicit tables, implicit tables, RK
lifecycle, and each ODE. CUDA tests are capability-gated.

Dense-output tests have their own CPU jobs and a capability-gated CUDA job:
`test_explicit_dense_output_{cpu,cpu_omp,cuda}.bin` checks order, lifetime,
storage, rejection, backward integration, and external section refinement.
`test_explicit_{lorenz,rossler,vdp}_dense_output_{cpu,cpu_omp,cuda}.bin` samples
off-step times through an external manager, checks independent 40-digit Decimal
RK4 references, restarts, and commits a shortened endpoint. Build these targets
with the same Makefile as above.

## SciPy-Style Adaptation

`time_step_adaptation_scipy<VectorOperations>` uses the same initialization,
assessment, update, and rejection interface as the MATLAB controller. It accepts
only `error_ratio < 1`, uses safety `0.9`, limits growth to `10` and shrinkage to
`0.2`, and prevents immediate growth after a rejected attempt. Ordinary embedded
errors use RMS scaling with `atol + rtol*max(abs(previous), abs(candidate))`.
The controller owns no state vectors; a generic `transform_reduce_sum` reuses
vector-operation scratch storage. The existing MATLAB controller is unchanged.

An optional second error pointer in `assess()` enables the DOP853 combined RMS
assessment, requiring error exponent `8`. Both inputs are state-error vectors,
already multiplied by the attempted step size. DOP853 forms both defects from its
ordinary derivatives and dispatches to this overload through a C++17 capability
check. Controllers without dual assessment cannot adapt DOP853.

The five configuration fields match the MATLAB controller. Relative tolerance is
clamped to at least `100*epsilon(scalar_type)`, exposed by `relative_tolerance()`.
Unlike SciPy's complete solver, initialization uses a configured step rather than
an RHS-based estimate, and the minimum step is configured rather than derived
from ten floating-point ULPs. Failed numerical attempts shrink independently;
an inadmissibly small retry returns failure. Modified endpoints limit the next
step and fresh integrations reset rejection history.

```bash
make test_explicit_scipy_adaptation_cpu_omp.bin
OMP_NUM_THREADS=8 ./build/test_explicit_scipy_adaptation_cpu_omp.bin
```

Use `_cpu.bin` or `_cuda.bin` for other backends. Generic reduction tests use
`test_transform_reduce_sum_{cpu,cpu_omp,cuda}.bin`. CI has separate reduction and
SciPy adaptation CPU jobs plus capability-gated CUDA coverage.

## Filter Adaptation

`integration/time_step_adaptation_filter.h` provides I, H211, PC, PID (default),
H312, PPID, and H321 presets. Select one through `params::controller`; the other
five parameters match SciPy adaptation. Scaling is additive RMS, acceptance is
strictly below one, and the estimator already contains its step-size factor.
The error powers use the supplied state-error exponent (`embedded_order()+1`),
not the primary method order. These presets follow the recurrence and coefficients
in [Kennedy and Carpenter, Eq. 93 and Table 8](https://ntrs.nasa.gov/api/citations/20160005923/downloads/20160005923.pdf).

For normalized errors `E` and actual attempted step magnitudes `h`, the proposal is
`0.9*h_n*E_n^(-alpha)*E_(n-1)^beta*E_(n-2)^(-gamma)` times
`(h_n/h_(n-1))^a*(h_(n-1)/h_(n-2))^b`. It is evaluated in the log domain with
growth limits `[0.2,10]` and configured step bounds. I control is used until enough
accepted samples exist, and for error retries; numerical failures shrink by `0.2`.
There is no immediate growth after rejection. Exactly zero error permits bounded
growth without taking a logarithm of zero.

History contains only two committed scalar samples, never rejected trials or
predicted step sizes. Assessment stages a sample; `update(accepted,...)` commits
it. A modified endpoint, a time discontinuity, a direction or estimator-order
change, `reset()`, or initialization invalidates history. External modifications
also bound the next step by the elapsed interval. This is error control, not
steady-state globalization or a nonlinear-tolerance policy.

```cpp
using controller_type = nmfd::time_steppers::integration::time_step_adaptation_filter<vector_operations_type>;
controller_type::params p;
p.controller = "H211";
auto control = std::make_shared<controller_type>(operations, p);
```

RK23, RK45, and embedded DIRK pairs use this controller without stepper changes.
DOP853's combined dual estimator is not supported; use SciPy adaptation for it.
The controller allocates no vectors and uses the backend reduction directly,
without vector-sized host/device copies. `test_explicit_filter_adaptation` checks
all recurrences independently for float/double, startup, committed history,
rejection, bounds, modified endpoints, forward/backward RK, and repeated runs.
Parent CI adds serial/OpenMP and capability-gated CUDA jobs plus sanitizer
coverage; implicit system jobs exercise every preset with both SDIRK methods.

The `tests/implicit_{lorenz,rossler,vdp}_problem.h` and
`tests/implicit_mass_dae_problems.h` headers share the regression equations and
analytical Jacobians with external diagnostic drivers. They are test models,
not additional public solver interfaces; no executable `.cpp` inclusion or
duplicated numerical formulas is needed.

## Pseudo-Time Globalization

`integration/time_step_adaptation_globalization.h` supplies a separate policy for
relaxing toward steady equations. It does not control physical-time accuracy or
change Newton's convergence settings. Its existing adaptation interface works
with IE and the supported DIRK methods without an embedded estimator.

```cpp
using controller_type = nmfd::time_steppers::integration::time_step_adaptation_globalization<
    vector_operations_type, steady_merit_type>;
controller_type::params p;
p.initial_step = .01;
p.maximum_step = 1e6;
auto control = std::make_shared<controller_type>(steady_merit, p);
```

`steady_merit` is a shared functor whose `operator()(const vector_type&)` returns a
nonnegative physical steady-residual measure. It must not use the shift-dependent
stage residual. Keep the physical time and equation parameters fixed in both the
problem and merit; the stepper's advancing time is pseudo time. The policy is
forward-only. Use an external manager to stop at the desired steady tolerance.

For committed merit `m` and trial merit `m_trial`, the default accepts every
successfully solved, finite, admissible candidate with finite nonnegative merit,
including temporary merit growth. `maximum_merit_growth = 0` (the default)
disables growth-based rejection. Set it to a finite value at least one only to
enable an explicit growth guard; one requests monotone merit. A decaying physical
trajectory need not have monotonically decaying residuals, so that guard may
prevent progress even for stable systems.

The next proposal is `h*(m/m_trial)^residual_power`, with factor bounds `[0.5,2]`
and the configured positive step bounds. Thus merit growth normally shrinks the
next step rather than rejecting the current one. The exponent defaults to one.
Zero trial merit permits bounded growth; two zero merits keep the step unchanged;
zero-to-nonzero merit uses the minimum factor. Nonfinite trial merit, a failed
stage, or an enabled growth guard triggers `rejection_factor` shrinking (default
`0.5`), with no immediate growth after retry. Failure at the minimum step is
returned, never accepted. Budget exhaustion is not evidence of an equilibrium.

The policy retains only scalar history and a shared merit handle, with one merit
evaluation per trial after caching the committed baseline. Initialization/reset,
pseudo-time jumps and modified endpoints invalidate that cache. Reset after an
independent state or merit-definition change. Failed stages bypass assessment;
rejected candidates never become the baseline. Merit reductions stay on the
selected backend; the controller allocates no vectors and copies no vector data.

`test_implicit_globalization_{cpu,cpu_omp,cuda}.bin` checks parameter validation,
bounded growth/shrinkage, zero/nonfinite merits, retries, modified/failed commits,
reset and steady convergence of a nonlinear ODE and a singular-mass constrained
DAE with IE and both SDIRK methods. Per-system steady regressions are separate:

- `test_implicit_nonnormal_steady_state`: stable linear dynamics with initially
  increasing residual, optional monotone-guard failure, restart and budget checks.
- `test_implicit_lorenz_steady_state`: all three unstable classical Lorenz roots
  (`epsilon=0`), plus damped Lorenz (`epsilon=0.05`) in both attracting basins
  compared with analytical equilibria and ordinary error-controlled RK45.
- `test_implicit_rossler_steady_state`: both unstable analytical equilibria.
- `test_implicit_vdp_steady_state`: the unstable origin for `mu=1` and `mu=1000`.

The unstable cases use perturbed seeds and initial pseudo-steps of 1 (Lorenz),
100 (Rossler), and 10/1e6 (Van der Pol). These are controlled root searches with
IE and both SDIRK methods, not a guarantee of convergence from arbitrary seeds
or with default small steps. Physical residuals must reach `1e-8` (`5e-10` for
stiff Van der Pol), with independent component errors below `1e-6`. Repeated
solves check controller reset and preservation of the initial state.
These are equilibrium comparisons, not assertions about identical transient
trajectories or settling times; the legacy `epsilon=0.0055` physical-time
regressions remain unchanged. Merit work
uses the physical RHS kernels and backend reductions without vector-sized host
copies. Parent CI includes CPU/OpenMP, capability-gated CUDA and sanitizer
coverage. Ordinary integration still uses its independently selected constant,
MATLAB, SciPy or filter controller. No timestepper method signatures change.

## Visual ODE Benchmarks

From the parent project root (or from the standalone library checkout), run:

```bash
bash source/contrib/nmfd-timesteppers/tests/run_all.sh
# In a standalone checkout: bash tests/run_all.sh
```

The default output is `build/ode_benchmarks/`, relative to the caller's working
directory. Use `--build-dir PATH` to override it. One shared executable lives in
`bin/`; every selected system receives `config.json` and an executable `run.sh`.
Runs are preserved under `<system>/runs/<batch>/`, with per-case logs, trajectories,
independent references, component/phase/step/error/constraint plots, and
`summary.json`. Batch results are retained in `reports/`.

```bash
# Force rebuilding and assembling only. No trajectories, plots, or GPU initialization.
bash source/contrib/nmfd-timesteppers/tests/run_all.sh -B

# Run only Lorenz, or launch a previously assembled case.
bash source/contrib/nmfd-timesteppers/tests/run_all.sh --system lorenz
bash build/ode_benchmarks/lorenz/run.sh

# Optional backends and controller selection.
bash source/contrib/nmfd-timesteppers/tests/run_all.sh --backend cpu_omp --threads 2
bash source/contrib/nmfd-timesteppers/tests/run_all.sh --backend cuda --controllers H211 PID H321

# Compare adaptive IMEX with DIRK on the same equations and independent references.
bash source/contrib/nmfd-timesteppers/tests/run_all.sh \
  --systems lorenz rossler vdp vdp_stiff coupled_mass state_mass time_mass time_coefficient stiff_tracking \
  --methods 'SDIRK2(1)2' IMEX_ARK324L2SA --controllers scipy

# The four non-embedded IMEX tables use constant stepping (--step sets h).
bash source/contrib/nmfd-timesteppers/tests/run_all.sh --system coupled_mass \
  --methods IMEX_EULER IMEX_HEUN_TR2 IMEX_ARS222 IMEX_ARS233 --controllers constant

# Optional uniform dense samples, in a separate *.dense.dat trajectory.
bash source/contrib/nmfd-timesteppers/tests/run_all.sh --system lorenz \
  --methods IMEX_ARK324L2SA --controllers scipy --sample-step 0.01
```

Normal invocations rebuild only stale binaries and leave unchanged assembly files
untouched. Compiler dependency files track the included timestepper, model, NMFD,
SCFD, and JSON headers. Changes to the Makefile or recorded compiler/dependency
settings also invalidate the build. `-B` forces a rebuild for the selected backend
and **does not run benchmarks**; it needs only Python's standard library and the
C++ dependencies. Normal execution additionally requires NumPy, SciPy, and
Matplotlib. Set `PYTHON` explicitly when needed. Sibling dependency paths and
compiler settings follow the standalone Makefile's existing environment variables.

Defaults retain the twelve-system matrix: Lorenz and Rossler to `t=50`, Van der
Pol to `t=20` (`mu=1`) or `t=10` (`mu=1000`), and the manufactured mass/DAE and
non-autonomous cases. Both SDIRK methods and all nine MATLAB/SciPy/filter settings
are selected, with `IMEX_ARK324L2SA` added for the nine ODE/nonsingular-mass systems,
giving 297 trajectories. Singular-mass DAE cases explicitly skip IMEX; an IMEX-only
selection for them is rejected before building. Default adaptation is `rtol=1e-5`,
`atol=1e-7`, initial step `0.001`, and step bounds `[1e-8,0.05]`; Newton tolerances
remain `2e-10` absolute and `1e-12` relative. Use `--help` for interval and other
overrides. Analytical initial states are computed consistently at `--start`.

IMEX splits preserve the original full equations and parameters. Lorenz and Rossler
use explicit nonlinear coupling and implicit linear terms; Van der Pol keeps its
stiff damping implicit. Stiff tracking treats forcing explicitly and decay implicitly.
Coupled mass splits off-diagonal/diagonal physical RHS terms; scalar mass and
time-coefficient cases use a 1:3 explicit/implicit RHS split. Both rates solve the
same problem-owned mass system. Configurations and trajectory metadata record
the split; reports and plots compare the full state, not individual partitions.
`--sample-step` enables ARK dense output explicitly. Uniform samples and accepted
steps retain separate row counts; plots/reference checks cover sampled interiors
without replacing accepted-step statistics or relaxing the endpoint checks.
The default remains accepted-step output with dense storage disabled.
`test_implicit_imex_benchmarks.cpp` checks rate sums against the original equations
and finite-differences every shifted implicit Jacobian, including mass derivatives.

Lorenz/Rossler accuracy is checked over the first two time units; later chaotic
divergence is plotted, not treated as an integration failure. Other systems use
the full interval. The diagnostic bound is `max(abs(error)/(1+abs(reference))) <=
0.01`, configurable with `--max-relative-error`. Every trajectory must reach its
full target with finite states; DAE constraints must remain below `2e-9`. These
diagnostic bounds do not replace or relax the existing CI accuracy regressions.

## DOP853

Use the existing explicit step with constant adaptation and dense output disabled:

```cpp
using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_constant<vector_operations_type>;
using step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<
    vector_operations_type, problem_type, adaptation_type>;
adaptation_type adaptation({0.25});
step_type step(operations, problem, adaptation, {"DOP853"});
```

The coefficient data is isolated in `runge_kutta/dop853_coefficients.h`, preserving
the published digits as `long double` literals. Rooted-tree tests verify all
conditions through order eight and a failure at order nine, plus the fifth- and
third-order companion formulas. No changes to the
problem, step, adaptation, or integrator interfaces are required.

For adaptive DOP853, replace constant adaptation with
`time_step_adaptation_scipy<vector_operations_type>`. The step owns two reusable
E5/E3 defect buffers only in that configuration. Fixed DOP853 owns neither, and
ordinary embedded methods keep one error buffer. Error estimation adds no RHS
calls; dense-only stages run after acceptance. An incompatible adaptive controller
returns `error_estimate_unavailable` before evaluating the RHS or changing output.
There is no fabricated `embedded_b` pair: `has_error_estimate()` and the estimator
kind distinguish this case. Diagnostic `error_estimate(0)`/`error_estimate(1)`
return raw E5/E3 state defects; neither is the combined error alone. The no-argument
getter remains for ordinary embedded pairs. Native dense output still reports
order seven and remains independently optional.

```bash
make test_explicit_dop853_cpu.bin test_explicit_dop853_cpu_omp.bin
OMP_NUM_THREADS=8 ./build/test_explicit_dop853_cpu_omp.bin
```

The same source builds as `test_explicit_dop853_cuda.bin`. Its analytical linear,
non-autonomous, and nonlinear cases check eighth-order refinement above roundoff,
RHS counts, forward/backward endpoints, restart, aliasing, capability guards,
step budgets, and rollback after nonfinite stages. CI has separate
`timestepper-dop853-fixed-cpu` and capability-gated `timestepper-dop853-fixed-cuda`
jobs. Frozen legacy files are not involved.

`test_explicit_dop853_adaptive_{cpu,cpu_omp,cuda}.bin` checks signed E5/E3 defects
against SciPy values, optional buffer counts, tolerance refinement, restart,
backward integration, attempt/minimum-step limits, and dense-stage recovery.
Separate adaptive CPU/CUDA CI jobs also exercise the CUDA Lorenz simulation.

Native DOP853 dense output has separate CPU, SciPy-comparison, and capability-gated
CUDA jobs. Analytical tests verify local eighth-order error, forward/backward
sampling, four optional buffers, three extra RHS calls, view lifetime, shortened
endpoints, aliasing, retry, and rollback on each failed extra stage. The existing
Lorenz, Rossler, and Van der Pol external-manager tests also exercise fixed DOP853.

```bash
make test_explicit_dop853_dense_output_cpu_omp.bin
OMP_NUM_THREADS=8 ./build/test_explicit_dop853_dense_output_cpu_omp.bin build/dop853_dense_samples.txt
python3 tests/compare_explicit_dop853_dense_output.py build/dop853_dense_samples.txt
```

## Lorenz Simulation Comparison

The separate `test_explicit_lorenz_simulation.cpp` executable records accepted
endpoints through `ExternalManagement`, without changing the problem or integrator.
The manager buffers independent host snapshots in a `std::vector`, acquired through
scoped read-only vector views. It creates and writes the output file only after
successful integration. RAM usage therefore grows with the recorded step count.
The user supplies a positive integration end time and an output filename:

```bash
make test_explicit_lorenz_simulation_cpu_omp.bin
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_simulation.dat
python3 tests/compare_explicit_lorenz_simulation.py \
     build/lorenz_simulation.dat
```

An optional method argument selects `RK23` or `DOP853` instead of the default `RK45`:

```bash
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_rk23.dat RK23
python3 tests/compare_explicit_lorenz_simulation.py \
     build/lorenz_rk23.dat
```

An additional controller argument selects SciPy-style adaptation. RK45/RK23
default to MATLAB-style; DOP853 defaults to SciPy and rejects MATLAB-style control:

```bash
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_scipy.dat RK45 scipy
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_dop853.dat DOP853 scipy
```

The same Makefile target supports `_cpu.bin` and `_cuda.bin`. Python requires
NumPy, SciPy, and Matplotlib. The trajectory stores its parameters, initial state,
integration interval, backend, and tolerances in a JSON comment header, followed
by `time x y z` rows. A completion footer identifies successfully finished runs;
the comparator rejects incomplete, nonfinite, or inconsistent data.

The Python script uses the saved method name directly for SciPy RK45, RK23, or DOP853.
Classical RK4 instead uses an independent NumPy implementation on the recorded
constant-step grid, including a clipped final step. It produces the same
trajectory, difference, summary, and plot files; reference tolerances do not apply
to this fixed-step comparison. Analytical order tests and high-accuracy ODE
fixtures separately verify numerical accuracy.

```sh
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_rk4.dat RK4 constant 0.001
python3 tests/compare_explicit_lorenz_simulation.py \
     build/lorenz_rk4.dat
```

RK4 defaults to constant adaptation with step size `0.001`. The optional final
argument sets the constant step or the initial step for adaptive methods.
`timestepper-rk4-simulation-python-cpu` separately checks serial/OMP recordings,
Python agreement, endpoint clipping, and rejection of adaptive RK4 settings.
It uses tighter tolerances and samples at the C++ output times. Adaptive controllers
differ, so matching accepted-step counts or bitwise-identical states is not required.
It writes `_python.dat`, `_error.dat`, `_summary.json`, `_comparison.png`, and
`_errors.png` next to the trajectory. Pointwise absolute/relative checks determine
the exit status; `--check-atol` and `--check-rtol` set their thresholds.

For long chaotic simulations use `--diagnostic-only` to report and plot trajectory
divergence without labeling it a passing accuracy test. Input validation and
reference-solver failures still fail. `--no-plots` skips Matplotlib output.
CI uses a separate `timestepper-lorenz-simulation-python-cpu` job for short serial
and OMP comparisons of all supported method/controller combinations, input-validation tests, and
plot artifacts.
