# Rosenbrock Continuous Extensions

ROS2 and ROS3 use native increment polynomials, not endpoint-rate Hermite
interpolation. The derivation below assumes the exact base Jacobian and identity
mass; autonomous constant nonsingular mass uses the equivalent
[residual-form corrections](rosenbrock_mass.md). The original endpoint and error estimator do not
change. Their coefficients are the [KPP increment-form tables](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html).

## Polynomial

For a pending step of signed length h, with numerical endpoints u0 and u1,

```text
P(theta) = (1-theta)*u0 + theta*u1
         + theta*(1-theta)*sum((D_i + theta*E_i)*k_i),  0 <= theta <= 1.
```

Endpoint evaluation copies the stored endpoints exactly. Interior evaluation uses
only backend vector algebra; it invokes neither the problem nor the solver.

ROS2 uses its two original increments, E=0, and

```text
D_1 = D_2 = 1 / (2*gamma*(2*gamma-1)).
```

ROS3 adds one dense-only increment after the embedded-error assessment accepts:

```text
A*k4 = f(t+h,u1) - sum(m_i*k_i)/h,
A    = I/(gamma*h) - J(t,u0).
```

The same prepared operator and solver binding are retained. Both sides use the
existing balanced shift. The secant term is assembled from increments, not by
subtracting almost equal endpoint states. This stage has a4j=m_j, c4j=-m_j and
gamma4=0. Its time is t+h, including clipped and backward steps. Base time is
restored after the endpoint rate, including failure and exception paths.

The ROS3 D/E constants in `rosenbrock/continuous_integration.h` solve the order
conditions below, evaluated with 70-digit decimal arithmetic and stored as double
precision. They are derived for these particular ROS3 coefficients; they are not
a generic extension for other Rosenbrock methods.

## Order Conditions

For an autonomous equation with exact base Jacobian J=F', expand each increment:

```text
k_i = h*d_i*F + h^2*e_i*J*F
    + h^3*(p_i*J^2*F + q_i*F''(F,F)) + O(h^4).

d_i = gamma*(1 + sum(c_ij*d_j))
e_i = gamma*(d_i + sum(a_ij*d_j) + sum(c_ij*e_j))
p_i = gamma*(e_i + sum(a_ij*e_j) + sum(c_ij*p_j))
q_i = gamma*(sum(a_ij*d_j)^2/2 + sum(c_ij*q_j)).
```

Let B have rows d,e,p,q, and append m4=0 for ROS3. The cubic extension satisfies

```text
B*m = (1, 1/2, 1/6, 1/6)
B*D = (0,-1/2,-1/6,-1/6)
B*E = (0,   0,-1/6,-1/6).
```

Consequently its weights reproduce theta, theta^2/2, theta^3/6 and theta^3/6
for the four elementary differentials. ROS2 imposes only the first two rows,
giving the quadratic formula above. Tests independently reconstruct these rows
from the stepping coefficients and check the identities at multiple coordinates.

Non-autonomous equations are handled by time augmentation. The table's stage
times equal sum(a_ij*d_j), and its partial-time weights gamma_i equal d_i. The
existing h*gamma_i*f_t terms therefore implement the augmented Jacobian's time
column. The dense-only stage has d4=0, so it requires no additional f_t term.

## Accuracy And Limits

Reported continuous orders are two and three, respectively, for smooth ODEs in
the classical accuracy regime. Stiff-decay interpolation is bounded as negative
lambda*h grows; unlike raw endpoint-slope Hermite interpolation, no h*f term
causes unbounded samples. This does not imply stiffness-uniform accuracy,
positivity, monotonicity, or resolution of an initial layer inside a large step.
The derivation alone does not establish DAE or approximate-Jacobian W support.
The separate constant-mass extension and W-compatible interpolant have their
own derivations and regression gates.

## Cost And Lifetime

`DenseOutput=false` retains the original workspace and rate/solve counts. Enabled
ROS2 adds one input snapshot and no evaluations or solves; enabled ROS3 adds that
snapshot and k4, one endpoint rate and one shifted solve per accepted candidate.
Non-autonomous runs retain their existing single f_t vector.

Interpolation and storage are O(N) with fixed stage counts. The additional solve
has the complexity of the supplied linear solver. No per-step backend allocation,
vector-sized host transfer, or trajectory history is introduced.

Views borrow one pending candidate. Finalization (including rejection or modified
endpoints), reset, and subsequent attempts invalidate them. A failed dense-only
stage never publishes output; it follows the controller's normal bounded retry
path from the unchanged committed input.
