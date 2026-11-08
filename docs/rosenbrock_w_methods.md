# Rosenbrock/W Methods

`ROS34PW2` uses the four-stage Rang/Angermann RA34PW2 table, order three with
an embedded method of order two. Original coefficients are tabulated in
[PETSc's source](https://petsc.org/main/src/ts/impls/rosw/rosw.c.html);
[the method documentation](https://petsc.org/main/manualpages/TS/TSROSWRA34PW2/)
identifies the 2005 paper. This implementation currently supports identity-mass
ODEs only, not the paper's broader DAE setting.

## Coefficient Convention

Write the original stages as

    K_i = h*f(t+c_i*h, u+sum(A_ij*K_j))
          + h*B*sum(Gamma_ij*K_j) + h^2*g_i*f_t,
    c = A*1, g = Gamma*1,
    u_high = u + b^T*K, u_low = u + bhat^T*K.

Here B approximates J=f_u at the base point. It is fixed throughout an attempt;
f and f_t remain the physical rate and its partial time derivative at fixed state.
The problem owns B and its approximation policy. The linear solver receives
the balanced shifted operator, not a modified nonlinear equation.

The engine's increments are y=Gamma*K. With common diagonal gamma, define

    a = A*Gamma^-1, c_lower = -lower(Gamma^-1),
    m = b^T*Gamma^-1, e = (b-bhat)^T*Gamma^-1.

Then each stage uses

    (I/(h*gamma)-B)*y_i
       = f(t+c_i*h, u+sum(a_ij*y_j))
         + sum(c_lower_ij*y_j)/h + h*g_i*f_t.

Table constants are converted offline with 70-digit decimal arithmetic from
the published coefficients, then stored as doubles. No matrix inverse is formed
at runtime. The existing balancing scales both sides consistently. Preparation
and binding occur for every attempt, including retries and modified endpoints.
Reusing B never implies reusing an obsolete timestep shift.

## Independent W Conditions

Let alpha=A*1 and g=Gamma*1. For third order with independent J and B,
the original endpoint weights satisfy

    b^T*1 = 1,
    b^T*alpha = 1/2, b^T*g = 0,
    b^T*A*alpha = 1/6, b^T*A*g = 0,
    b^T*Gamma*alpha = 0, b^T*Gamma*g = 0,
    b^T*(alpha .* alpha) = 1/3.

These correspond to F, JF, BF, JJF, JBF, BJF, BBF and F''(F,F).
The embedded weights satisfy the first three conditions. Time augmentation
uses the original row sums; it does not evaluate time at transformed row sums.
The coefficient tests recover the original table from the engine's constants,
compare it to the independently tabulated source, and check these conditions.
Nonlinear logistic, noncommuting coupled decay and manufactured non-autonomous
exponential solutions verify the resulting physical orders on each backend.

## Limits And Cost

With the exact Jacobian, the primary method is L-stable. W order preservation
with an approximation is not unconditional stability: a poor B may impose small
steps. B=0 order tests are nonstiff tests, not demonstrations of stiff stability.
Linear solvers must still solve their shifted systems to suitable accuracy;
failed or nonfinite solves follow bounded rejection, never unconditional acceptance.

There are four rates/solves and one preparation per successful attempt. Retain
six backend vectors, or seven with f_t; the estimator reuses RHS scratch.
Vector algebra is O(s^2*N), s=4, and storage O(s*N), plus injected problem/solver
costs. There are no per-step vector allocations or vector-sized host transfers.
Preparing B, factorization and Krylov iterations are additional costs.

## Dense Output Is Separate

The published continuous extension has third order for an exact J, but its
interior weights do not satisfy the separate third-order JBF/BJF/BBF conditions
for arbitrary B. Combining these terms as though B=J conceals the failure.
Neither that formula nor the existing ROS3 extension establishes third-order
W-compatible interpolation. This increment therefore refuses dense output at
construction and in direct continuous-weight queries. Developing and validating
a W-compatible extension is the next increment; ordinary ROS2/ROS3 dense output
is unchanged.
