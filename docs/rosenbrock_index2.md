# Gauged Index-2 Rosenbrock Execution

Select `method="ROSI2P2"`, `mass_form=true`, `index2_dae=true`.
Defaults and existing method signatures are unchanged. This opt-in capability
uses a constant singular mass and an exact, complete coupled Jacobian, including
constraint and pressure-gauge rows. It does not invert the singular mass.

The four-stage stiffly accurate primary table is Table 3.2 of
[Rang/Angermann, New Rosenbrock methods of order 3 for PDAEs of index 2](https://dml.cz/bitstream/handle/10338.dmlcz/700435/Equadiff_11-2005-2_46.pdf).
It is not an arbitrary-Jacobian W method. The existing offline conversion
`a=A*Gamma^-1`, `c=-lower(Gamma^-1)`, `m=b^T*Gamma^-1` is unchanged.
Tests independently recover the original table and check ODE, stiff-accuracy,
E3, F3a/F3b/F3c and PDE moment conditions. No scalar matrix inverse is formed
by the engine at runtime.

## Problem And Gauge

For a Stokes-like semi-explicit formulation with linear multipliers, supply

```text
Md*u' = f(u,p),  0 = g(u),  0 = gauge(p).
R = [ a*Md*(u-previous_u)-b*f(u,p) ; g(u) ; gauge(p) ].
```

Use the existing `set_affine_shift(a,b)`, `apply`, reference buffers and
`get_jacobi_operator()`. Keep algebraic rows unscaled and exact. The prepared
operator freezes the base point and shift throughout every stage; residual
evaluations must not overwrite that snapshot. Solve the complete shifted system
with the injected direct/Krylov solver. Nullspace handling remains problem-owned;
a compatible Krylov residual alone does not prove pressure uniqueness.

Check initial constraints, differentiated (hidden) consistency and the gauge
before integration. Check their independent tolerances before commitment through
the existing controller/external-manager boundary. The engine does not inspect
DAE index, choose a gauge, project states or certify arbitrary user operators.
Externally modified endpoints require the same consistency checks.

## Moving Constraints Without Reinterpreting q_t

Represent time dependence in the problem by an additional clock T with T'=1.
Then M remains constant and singular, while f(u,p,T) and g(u,T) form an
autonomous augmented problem. Initialize T consistently with the desired initial
time. Include all force and constraint clock columns in the exact Jacobian.
Advance the clock on the same backend as the other differential variables.

For example, differentiating g(u,T)=0 gives g_u*u'+g_T=0. This determines the
hidden consistency equations used to initialize/reconstruct pressure. Higher
derivatives may be needed to evaluate a full consistent pressure rate.
No `evaluate_time_derivative` callback is required by this formulation, and
the existing q_t callback is not repurposed as a general DAE derivative.
Native non-autonomous singular-mass execution is still refused by index-2 mode.

The tests use both mean-zero and pinned gauges on autonomous and moving
Stokes-like systems. Autonomous velocities/pressures converge at order three.
For the moving example, velocities converge at order three but raw multipliers
and hidden defects converge at order two. Constraint/gauge equations remain at
roundoff. Reconstructed pressure recovers third-order accuracy. Do not infer
third-order raw multipliers from the primary ODE order.

## DAE Error Estimate

The published embedded weights satisfy ODE order two, but fail the index-2
constraint moments. Using them unchanged gives first-order pressure defects in
the moving example. They remain unchanged for ordinary ODE execution.

Index-2 mode instead uses separately derived weights with original-coordinate
`bhat_4=1/2` and

```text
bhat^T*1=1,  bhat^T*B*1=1/2,  bhat^T*B^-1*c_time^2=1,
B=A+Gamma, c_time=A*1.
```

For this table the weights also satisfy `bhat^T*B^-2*c_time^2=2`.
The four weights are approximately
`(.8910881528826726,-.2681447044143851,-.1229434484682875,.5)`.
They are converted offline to `index2_e=(b-bhat)^T*Gamma^-1`.
Independent algebraic checks and signed refinement tests verify O(h^3)
differential and O(h^2) pressure estimates on the supported examples.
The existing controller receives exponent order **two**, reflecting pressure
defects rather than the ODE embedded order plus one. All MATLAB/SciPy/filter
presets retain their interfaces. No extra rates, solves or vectors are used.
State-error acceptance still does not replace independent consistency checks.

## Optional Constrained Dense Output

Only `DenseOutput=true` requires the existing optional
`bool evaluate_rate(state,rate)`. The problem supplies the full physical rate,
including differentiated constraints and the selected gauge. It must preserve
input, reference buffers and the frozen operator. It may refuse states outside
its explicitly configured consistency budget; the engine never silently projects
an endpoint to obtain a rate. The test provider checks primary/hidden/gauge
defects independently before evaluating its analytical rate.

After error acceptance, evaluate q0 and q1 at the two numerical endpoints and
retain h*q0 and h*q1. Use the cubic Hermite polynomial

```text
P(theta) = (1-3*theta^2+2*theta^3)*u0
         +(3*theta^2-2*theta^3)*u1
         +theta*(1-theta)^2*h*q0-theta^2*(1-theta)*h*q1.
```

The endpoint change is represented by the existing stage increments to avoid
subtracting nearby states. This is not the ROS34PW2 W interpolant. The scalar
`dense_output_order()` reports the classical differential order three; raw
algebraic components are not automatically third-order or constraint-consistent.
Endpoint copies are exact, including the unmodified numerical pressure.

Before using interior DAE samples for events or commitment, the external manager
must invoke a problem-owned constrained reconstruction as needed. In the tests,
reconstruct the dependent velocity from g and pressure from the differentiated
constraint plus its gauge. That operation is backend-only and restores primary
and hidden consistency; no universal pressure interpolation rule is introduced.
Do not apply arbitrary projection and assume it preserves order for other DAEs.
Validate reconstruction accuracy and conditioning for the application.

False/nonfinite rates or exceptions never publish a candidate/view. Sampling
does not call the problem or solver. Rejection, reset, or an event-shortened
`finalize(accepted_modified,...)` invalidates the borrowed view. The next attempt
prepares the actual reconstructed committed state, not the old full endpoint.

## Cost And Limits

Fixed/adaptive stepping retains six backend vectors and uses four residuals and
full shifted solves per attempt. Dense output adds three vectors and two
problem-owned rate evaluations after acceptance, **no extra shifted solves**.
The disabled feature has no rate calls or dense vector storage. There is no
per-attempt allocation or trajectory retention. Engine algebra is O(s^2*N),
storage O(s*N), with s=4; sampling/reconstruction in the block-local examples
are O(N). The supplied coupled solves, mass/constraint solves and reconstruction
may have additional application-dependent costs. Scalar reductions remain, but
no vector-sized host/device copies are introduced. Large guarded blocks test
fixed/adaptive, moving/autonomous and dense-enabled/disabled execution.

These gates establish bounded constant-mass, linear-multiplier Stokes-like
capabilities, not arbitrary nonlinear multiplier coupling, general variable-mass
DAEs, index reduction, stiffness-uniform accuracy or a complete Navier-Stokes
discretization. The moving gate has a time-dependent constraint offset and a
fixed velocity constraint gradient; rotating/nonlinear velocity constraint
manifolds need separate validation. Ordinary ODEs, index-1 ROS34PW2, and their dense formulas remain
unchanged. Periodic-orbit/event policies remain outside the library.
