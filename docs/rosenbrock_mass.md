# Constant-Mass Rosenbrock Execution

The optional `params.mass_form=true` path supports autonomous equations

    M*u' = f(u),

with constant nonsingular M. Identity-rate execution remains the default.
ROS2/ROS3 use B=J_f; ROS34PW2 may use its validated approximate B. No existing
method signature changes. No rate, force, or mass-action callback is added.

## Stage Residual

With the existing transformed coefficients, prepare once per attempt:

    A = a*M - b*B,
    alpha = 1/(h*gamma), s = max(abs(alpha), 1/reference_time),
    a = alpha/s, b = 1/s.

Prepare at the input state with `previous_state=input` and `known_rhs=0`, so
the stage Jacobian has precisely this shift. Each stage has

    U_i = u_n + sum_j A_ij*y_j,
    q_i = gamma*sum_j C_ij*y_j.

Set `previous_state=U_i+q_i`. The existing residual then gives

    -R(U_i) = b*f(U_i) + a*M*q_i,
    (a*M-b*B)*y_i = -R(U_i).

This is the constant-mass transformed Rosenbrock equation documented in
[PETSc TSROSW](https://petsc.org/release/manualpages/TS/TSROSW/). Residuals are
constructed with balanced a,b directly, not by evaluating an unbalanced
equation and scaling afterward. Algebraically, multiplication by M inverse
reduces these stages to the existing rate scheme for q=M^{-1}f and
B_q=M^{-1}B. No explicit inverse is formed by the engine.

Only the problem's reference buffer changes between stages. Its residual
evaluation must preserve the operator snapshot prepared at the base state.
Every new attempt reinitializes both reference buffers and rebinds its shifted
operator, including changed timesteps and externally modified endpoints.
The endpoint and embedded error remain sums of the same state increments.

## Continuous Extension

ROS2 uses the existing increment polynomial without additional solves.
For ROS3, the endpoint correction uses the same residual construction with

    U = u_1, q = -gamma*sum_i m_i*y_i.

For ROS34PW2, let w_s be the two native dense-coupling rows from
`rosenbrock_w_methods.md` and let Q_s=sum_i w_si*y_i. For s=0,1 evaluate
the residual at u_n,u_1 with previous_state=U+Q_s. Solve

    (a*M-b*B)*z_s = b*f(U) + a*M*Q_s,
    L_s = (z_s-Q_s)/gamma.

Left multiplication by M inverse gives the already validated W-compatible
slope equations. Thus the same continuous weights and classical orders apply
for constant nonsingular mass. Sampling performs only backend algebra.
This does not establish stiffness-uniform interpolation accuracy.

## Cost And Failure Contract

Ordinary stages perform 2/3/4 residuals and shifted solves for ROS2/ROS3/ROS34PW2.
ROS3 cannot simply reuse its previous residual: the mass-weighted coupling is
different even though the physical force is shared. Dense output adds 0/1/2
residuals and solves, only after error acceptance. Unlike identity-rate W dense
output, mass-form W dense preparation reevaluates both endpoint residuals.

Retained workspace is unchanged: 4/5/6 backend vectors without dense output,
5/7/9 with dense output. The problem and injected solver own their separate
storage. No vector allocation occurs per attempt; all O(N) vector work stays
on the selected backend. Costs of the supplied residual/linear solver remain
problem-dependent. No matrix assembly, full trajectory retention or vector-sized
host/device transfer is introduced.

Nonfinite residuals and failed/nonfinite solves follow the existing bounded
rejection policy. Exceptions propagate. Neither failures nor dense-only failures
publish partial candidates or views. Existing finalize/reset invalidation applies.

Forming previous_state=U+q and then U-previous_state introduces ordinary
state-sized cancellation when q is tiny relative to U. Balanced equations avoid
1/h amplification in the evaluated residual, but do not remove the floating-point
floor. Tests compare mass/rate forms over signed small/large steps, mass scales
and reference times; agreement is numerical, not bitwise equivalence.

## Capability Limits

`mass_form=true` with `autonomous=false` is rejected before workspace allocation.
The existing partial-time callback describes M-inverted rate derivatives, not
the physical f_t required here. No conversion is inferred.

State/time-dependent and singular mass, DAEs, index reduction and constrained
dense samples are outside this capability. No claim is inferred from a successful
shifted solve, and the engine cannot detect these properties of an arbitrary
matrix-free problem. Callers must satisfy the documented constant-mass contract.
