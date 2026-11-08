# Constant-Mass Rosenbrock Execution

The optional `params.mass_form=true` path supports equations

    M*u' = f(t,u),

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

The problem's reference buffer and evaluation time change between stages.
Residual evaluation must preserve the operator snapshot prepared at the base state.
Every new attempt reinitializes both reference buffers and rebinds its shifted
operator, including changed timesteps and externally modified endpoints.
The endpoint and embedded error remain sums of the same state increments.

## Non-Autonomous Terms

Select `params.autonomous=false` and supply the existing optional method
`bool evaluate_time_derivative(state,out)`. Its meaning is unchanged: the
partial derivative q_t of the physical rate q=M^{-1}f, at fixed state and the
current problem time. It is not f_t, a total derivative along a trajectory,
or an approximation inferred from the W Jacobian. `evaluate_rate` is still
not required by mass-form execution.

For constant nonsingular M, M*q_t=f_t. The standard transformed stage has an
additional h*d_i*f_t term, with d_i the table's time-derivative coefficient
(see [KPP's stage equations](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html)).
Use the same residual construction with

    U_i = u_n + sum_j A_ij*y_j, t_i = t_n + c_i*h,
    q_i = gamma*sum_j C_ij*y_j + h^2*gamma*d_i*q_t,
    previous_state = U_i+q_i.

Then -R(t_i,U_i)=b*f(t_i,U_i)+a*gamma*M*sum(C_ij*y_j)+b*h*d_i*f_t,
since a*h*gamma=b. The engine evaluates q_t once per attempt at (t_n,u_n),
keeps it unchanged through all stages/dense corrections, and recomputes it
after every retry. An existing RHS scratch vector holds h*q_t before adding
(h*gamma*d_i)*(h*q_t) to the reference; no scalar h^2 is formed. This avoids
premature h^2 overflow/underflow when the state-sized defect remains finite.
Ordinary floating-point range and state-sized cancellation still apply.

Residuals use each table's actual stage time and the clipped/signed timestep.
Restore base time after every evaluation, including exceptions and nonfinite
results. The prepared operator retains its base-time snapshot. Missing/incorrect
callback signatures are rejected on construction; false/nonfinite q_t follows
bounded numerical rejection, and exceptions propagate before publication.

## Continuous Extension

ROS2 uses the existing increment polynomial without additional solves.
For ROS3, the endpoint correction uses the same residual construction with

    U = u_1, t = t_n+h, q = -gamma*sum_i m_i*y_i.

For ROS34PW2, let w_s be the two native dense-coupling rows from
`rosenbrock_w_methods.md` and let Q_s=sum_i w_si*y_i. For s=0,1 evaluate
the residual at (t_n,u_n),(t_n+h,u_1) with previous_state=U+Q_s. Solve

    (a*M-b*B)*z_s = b*f(U) + a*M*Q_s,
    L_s = (z_s-Q_s)/gamma.

Left multiplication by M inverse gives the already validated W-compatible
slope equations. Thus the same continuous weights and classical orders apply
for constant nonsingular mass. Sampling performs only backend algebra.
These dense-only stages have zero additional partial-time coefficient, as in
the validated rate-form extensions; they do not request another q_t evaluation.
This does not establish stiffness-uniform interpolation accuracy.

## Cost And Failure Contract

Ordinary stages perform 2/3/4 residuals and shifted solves for ROS2/ROS3/ROS34PW2.
ROS3 cannot simply reuse its previous residual: the mass-weighted coupling is
different even though the physical force is shared. Dense output adds 0/1/2
residuals and solves, only after error acceptance. Unlike identity-rate W dense
output, mass-form W dense preparation reevaluates both endpoint residuals.

Retained workspace is unchanged: 4/5/6 backend vectors without dense output,
5/7/9 with dense output. The problem and injected solver own their separate
storage. Non-autonomous runs add only the existing q_t vector and one callback
per attempt. Each consumed time term adds two backend vector operations using
RHS scratch. No vector allocation occurs per attempt; all O(N) vector work stays
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

State/time-dependent and singular mass, DAEs, index reduction and constrained
dense samples are outside this capability. No claim is inferred from a successful
shifted solve, and the engine cannot detect these properties of an arbitrary
matrix-free problem. Callers must satisfy the documented constant-mass contract.
