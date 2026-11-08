# Rosenbrock/W Methods

`ROS34PW2` uses the four-stage Rang/Angermann RA34PW2 table, order three with
an embedded method of order two. Original coefficients are tabulated in
[PETSc's source](https://petsc.org/main/src/ts/impls/rosw/rosw.c.html);
[the method documentation](https://petsc.org/main/manualpages/TS/TSROSWRA34PW2/)
identifies the 2005 paper. This implementation supports identity-rate ODEs and
optional constant nonsingular mass, not the paper's broader DAE setting.

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

Without dense output, there are four rates/solves and one preparation per successful attempt. Retain
six backend vectors, or seven with f_t; the estimator reuses RHS scratch.
Vector algebra is O(s^2*N), s=4, and storage O(s*N), plus injected problem/solver
costs. There are no per-step vector allocations or vector-sized host transfers.
Preparing B, factorization and Krylov iterations are additional costs.

## W-Compatible Dense Output

The published continuous extension has third order for an exact J, but its
interior weights do not satisfy the separate third-order JBF/BJF/BBF conditions
for arbitrary B. Combining these terms as though B=J conceals the failure.
Neither that formula nor the existing ROS3 extension establishes third-order
W-compatible interpolation. The following extension is independently derived
for this implementation; it is not the published RA34PW2 continuous extension.
The primary and embedded endpoint formulas are unchanged.

### Two Additional Slope Stages

Let C have rows 1^T, alpha^T and g^T for the original four-stage table.
Choose the minimum-Euclidean-norm rows v0 and v1 satisfying

    C*v0 = (-gamma, 0, 0)^T,
    C*v1 = (-gamma, -gamma, 0)^T.

They are computed offline as C^T*(C*C^T)^-1 times the respective right-hand
sides, using 70-digit decimal arithmetic. This is a fixed scalar coefficient
calculation, never a runtime dense solve. Append the original-form stages

    (I-h*gamma*B)*L0 = h*f(t,u)     + h*B*sum(v0_j*K_j),
    (I-h*gamma*B)*L1 = h*f(t+h,u1) + h*B*sum(v1_j*K_j).

Both have the same diagonal gamma and frozen B as the original stages.
Their new Gamma rows are (v0, gamma, 0) and (v1, 0, gamma), respectively.
Their A rows are zero and (b^T,0,0). The new Gamma row sums are zero, so no
additional partial-time derivative terms are needed, even for non-autonomous
systems. These stages approximate h*f(t,u) and h*f(t+h,u1) with O(h^4) error
for smooth ODEs with fixed bounded B as h tends to zero. They are not raw
endpoint rates; their shifted solves cancel the spurious B terms.

To avoid requesting a separate B action, store w_s=v_s^T*Gamma^-1 and
q_s=sum(w_sj*y_j), where y are the existing transformed increments. Then

    (I/(h*gamma)-B)*z_s = f_s + q_s/(h*gamma),
    L_s = (z_s-q_s)/gamma.

The engine applies the same balancing to both sides of this equation. It
retains the original base rate during ordinary stepping, then prepares L0 and
L1 only after embedded-error acceptance. Only the endpoint rate is evaluated
again. No new problem or solver methods are required.

### Continuous Polynomial And Order

For theta in [0,1], use

    P(theta) = H00*u + H01*u1 + H10*L0 + H11*L1,
    H00 = 1-3*theta^2+2*theta^3, H01 = 3*theta^2-2*theta^3,
    H10 = theta*(1-theta)^2, H11 = -theta^2*(1-theta).

The polynomial uses a Hermite basis with native W slope stages, not raw physical
slopes. The weights on the six original-form stages are
(H01*b^T,H10,H11). For every theta they satisfy all eight independent W
conditions above with right-hand sides

    (theta, theta^2/2, 0, theta^3/6, 0, 0, 0, theta^3/3).

For example, the new stages have alpha=(0,1), A*alpha=(0,1/2), A*g=0,
Gamma*alpha=Gamma*g=0 and squared alpha=(0,1). Substituting their polynomial
weights proves the conditions without identifying B with J. Time augmentation
then covers the partial-time terms using the same existing f_t contract.
This establishes classical order three and O(h^4) one-step interior errors,
not stiffness-uniform accuracy. Linear-solve errors must remain suitably small.

Evaluation uses the existing secant form, with u1-u represented by sum(m_i*y_i)
rather than subtracting nearby endpoint states. Endpoints are copied exactly.
The borrowed view expires on finalize/reset, including rejection or shortening.
Sampling performs only backend vector algebra, never rate evaluations or solves.

### Cost, Failure And Limits

With DenseOutput=false the original six/seven backend vectors and four
rates/solves per attempt are unchanged. With DenseOutput=true retain three
additional vectors: the previous state and two slope stages. The base-rate
cache reuses the first slope buffer. Each error-accepted candidate costs one
extra endpoint rate and two solves, with no new operator binding. Thus a
successful autonomous step uses nine vectors, five rates and six solves;
non-autonomous runs add their existing f_t vector. Error-rejected trials perform
no dense-only solves or endpoint-rate calls. Sampling remains O(N); storage
remains O(N), excluding injected problem/solver costs. No per-step vector
allocation, trajectory storage or vector-sized host/device transfer is added.

Failures in either additional solve, slope reconstruction or endpoint rate never
publish partial output or a view. Numerical failures follow bounded rejection;
exceptions propagate, restoring base time after endpoint-rate calls. Every
retry rebuilds all ordinary and dense data from the unchanged committed state.

The extension is rational and has bounded scalar stiff-decay samples with B=J;
tests also exercise B=J/2 and B=2*J. This is not an assertion of stability for
arbitrary B, positivity, monotonicity or accuracy inside unresolved initial
layers. Constant nonsingular mass is supported by the optional
[residual form](rosenbrock_mass.md). Variable/singular mass, DAEs and prepared
periodic-orbit derivatives remain unsupported.
Ordinary ROS2/ROS3 continuous extensions are unchanged.
