# HJB Derivation Appendix

This appendix keeps the Week 2 Avellaneda-Stoikov core and extends it in the
three directions needed for Week 3: the Ho-Stoll dealer foundation, a
queue-reactive correction, and a Glosten-Milgrom adverse-selection layer. The
goal is not to present a fully general market-microstructure equilibrium. The
goal is to show, cleanly, how the inventory-risk term implemented in
`src/p2/hjb_solver.py`, the queue-state machinery in `src/p2/queue.py`, and the
Bayesian information filter in `src/p2/glosten_milgrom.py` fit together.

## 0. Ho-Stoll (1981) foundation
Ho and Stoll (1981) study a single dealer who faces Poisson customer arrivals and
chooses bid and ask quotes to maximize expected terminal utility of wealth under
CARA preferences. The shortest path from that setting into Avellaneda-Stoikov is
the one-period certainty-equivalent calculation: isolate the inventory-risk
compensation term first, then layer execution intensity on top. That is exactly
the quantity Avellaneda and Stoikov later reuse in their finite-horizon
reservation-price formula (Avellaneda-Stoikov 2008, eq. (7)).

Let the terminal asset value be

\[
S_T = S_0 + \sigma W_T,
\]

so that conditional on time \(0\), \(S_T \sim \mathcal{N}(S_0, \sigma^2 T)\). A
dealer with cash \(x\), inventory \(q\), and CARA utility
\(U(w) = -\exp(-\gamma w)\) evaluates terminal wealth
\(W_T = x + q S_T\) by

\[
\mathbb{E}\left[-e^{-\gamma(x + q S_T)}\right]
= -\exp\left(-\gamma(x + q S_0) + \frac{\gamma^2 q^2 \sigma^2 T}{2}\right).
\]

Because the exponential preserves certainty equivalents, the dealer is
indifferent between the random position and the deterministic certainty
equivalent

\[
\mathrm{CE}(x,q)
= x + q S_0 - \frac{\gamma}{2} q^2 \sigma^2 T.
\]

That quadratic penalty is the Ho-Stoll inventory-risk term. It is the exact same
penalty structure that reappears in the finite-horizon Avellaneda-Stoikov
quadratic approximation. The paper itself derives it inside a broader continuous
time Poisson dealer problem; the one-period expression above is the clean
specialization that exposes the spread component carried into later models
(Ho-Stoll 1981, dealer optimization under return uncertainty).

Now define the dealer's reservation bid \(b^{HS}(q)\) as the price paid now to
acquire one extra share while staying indifferent:

\[
\mathrm{CE}(x,q)
= \mathrm{CE}(x - b^{HS}(q), q+1).
\]

Substituting the certainty equivalents gives

\[
x + q S_0 - \frac{\gamma}{2} q^2 \sigma^2 T
= x - b^{HS}(q) + (q+1)S_0 - \frac{\gamma}{2}(q+1)^2 \sigma^2 T,
\]

so

\[
b^{HS}(q)
= S_0 - \frac{\gamma \sigma^2 T}{2}(2q+1).
\]

Likewise the reservation ask \(a^{HS}(q)\) solves

\[
\mathrm{CE}(x,q)
= \mathrm{CE}(x + a^{HS}(q), q-1),
\]

which yields

\[
a^{HS}(q)
= S_0 + \frac{\gamma \sigma^2 T}{2}(1-2q).
\]

Two quantities matter immediately:

\[
r^{HS}(q)
= \frac{a^{HS}(q) + b^{HS}(q)}{2}
= S_0 - q \gamma \sigma^2 T,
\]

\[
\Delta^{HS}
= a^{HS}(q) - b^{HS}(q)
= \gamma \sigma^2 T.
\]

So the classical Ho-Stoll inventory spread is approximately
\(\gamma \sigma^2 T\), independent of \(q\), while the reservation price carries
the inventory skew \(-q \gamma \sigma^2 T\). That decomposition is the main
bridge into Avellaneda-Stoikov: the spread compensates variance, and the center
price compensates existing inventory.

Where do customer arrivals enter? If a customer arrives over a short interval
\([t,t+\Delta t]\) with Poisson probability \(\nu \Delta t + o(\Delta t)\), the
dealer chooses quotes to trade off fill probability against those certainty
equivalent valuations. Ho-Stoll treat this tradeoff dynamically. In the
one-period simplification above, the inventory-risk component is already visible:
arrival intensity affects how often the dealer earns the spread, but the minimal
inventory compensation embedded in the spread is still \(\gamma \sigma^2 T\). That
is the term preserved in Avellaneda-Stoikov (2008, eqs. (7)-(9)).

## 1. Avellaneda-Stoikov control problem
Avellaneda and Stoikov (2008) keep the same CARA objective but move from a
single-period certainty-equivalent argument to a continuous-time control problem
with endogenous limit-order execution. The mid price follows arithmetic Brownian
motion,

\[
dS_t = \sigma dW_t,
\]

and the market maker posts bid and ask quotes \(b_t\) and \(a_t\). It is more
convenient to work with quote distances from the mid,

\[
\delta_t^b = S_t - b_t, \qquad
\delta_t^a = a_t - S_t.
\]

Market sell orders hitting our bid and market buy orders hitting our ask are
modeled as independent Poisson processes with exponentially decaying intensities,

\[
\lambda^b(\delta_t^b) = A e^{-\kappa \delta_t^b}, \qquad
\lambda^a(\delta_t^a) = A e^{-\kappa \delta_t^a}.
\]

This is the exact intensity form used in the repo's closed-form solver. A fill at
the bid increases inventory and reduces cash by the bid price; a fill at the ask
reduces inventory and increases cash by the ask price:

\[
dq_t = dN_t^b - dN_t^a,
\]

\[
dX_t = -b_t \, dN_t^b + a_t \, dN_t^a.
\]

Terminal wealth is marked to market,

\[
W_T = X_T + q_T S_T,
\]

and the control problem is

\[
\sup_{(\delta^b,\delta^a)}
\mathbb{E}_t\left[-e^{-\gamma(X_T + q_T S_T)}\right].
\]

The finite-horizon formulas implemented in `src/p2/hjb_solver.py` are the closed
form approximations of this control problem, with the final quote equations
reported in Avellaneda-Stoikov (2008, eqs. (7)-(9)).

## 2. Value function and HJB
Define the value function

\[
V(x,s,q,t)
= \sup_{(\delta^b,\delta^a)}
\mathbb{E}_{t,x,s,q}\left[-e^{-\gamma(X_T + q_T S_T)}\right].
\]

Dynamic programming gives the Hamilton-Jacobi-Bellman equation

\[
\partial_t V
+ \frac{1}{2}\sigma^2 \partial_{ss} V
+ \max_{\delta^b} \lambda^b(\delta^b)
\left[V(x-b,s,q+1,t) - V(x,s,q,t)\right]
+ \max_{\delta^a} \lambda^a(\delta^a)
\left[V(x+a,s,q-1,t) - V(x,s,q,t)\right]
= 0,
\]

with terminal condition

\[
V(x,s,q,T) = -e^{-\gamma(x + q s)}.
\]

This is the continuous-time counterpart of the Ho-Stoll dealer problem: diffusion
risk enters through \(\partial_{ss}V\), while transaction opportunities enter
through the jump terms. The Poisson structure is the same basic market-making
primitive in both papers. What changes in Avellaneda-Stoikov is that the arrival
rate is now explicitly controlled through \(\delta^b\) and \(\delta^a\).

## 3. Exponential utility reduction
Exponential utility suggests the separable ansatz

\[
V(x,s,q,t) = -e^{-\gamma(x + q s + \phi(q,t))}.
\]

Here \(\phi(q,t)\) collects the continuation value beyond the linear marked to
market term \(x + q s\). The ansatz is consistent with the frozen-inventory
indifference calculation in Ho-Stoll and with the reservation-price step used by
Avellaneda-Stoikov to derive eqs. (7)-(9).

Substituting into the bid jump term requires one careful sign check. A bid fill
occurs at price \(b = s - \delta^b\), so

\[
V(x-b,s,q+1,t)
= -e^{-\gamma(x + q s + \delta^b + \phi(q+1,t))}.
\]

Relative to the current state, the exponent increment is
\(\delta^b + \phi(q+1,t) - \phi(q,t)\), not
\(-\delta^b + \phi(q+1,t) - \phi(q,t)\). The analogous ask increment is
\(\delta^a + \phi(q-1,t) - \phi(q,t)\).

The common factor is \(V=-e^{-\gamma(x+qs+\phi)}<0\). Dividing by it therefore
reverses the ordering inside each control term: a maximum before division is a
minimum afterward. Equivalently, negating the minimized jump contribution and
scaling by \(1/\gamma\) gives the following positive maximization form:

\[
0
= \partial_t \phi(q,t)
- \frac{1}{2}\gamma \sigma^2 q^2
+ \max_{\delta^b} \frac{A}{\gamma} e^{-\kappa \delta^b}
\left(
1 - e^{-\gamma(\delta^b + \phi(q+1,t) - \phi(q,t))}
\right)
+ \max_{\delta^a} \frac{A}{\gamma} e^{-\kappa \delta^a}
\left(
1 - e^{-\gamma(\delta^a + \phi(q-1,t) - \phi(q,t))}
\right).
\]

This is the reduced HJB from which the closed-form quote distances follow.

## 4. First-order conditions and closed-form quote distances
Define the bid-side continuation gap

\[
\Delta_b \phi(q,t) = \phi(q+1,t) - \phi(q,t).
\]

The bid contribution to the HJB is then

\[
f_b(\delta^b)
= \frac{A}{\gamma} e^{-\kappa \delta^b}
\left(
1 - e^{-\gamma(\delta^b + \Delta_b \phi(q,t))}
\right).
\]

Differentiating and setting the derivative to zero gives

\[
0
= \frac{A}{\gamma} e^{-\kappa \delta^b}
\left[
-\kappa \left(1 - e^{-\gamma(\delta^b + \Delta_b \phi)}\right)
+ \gamma e^{-\gamma(\delta^b + \Delta_b \phi)}
\right].
\]

Solving that scalar first-order condition yields

\[
\delta_t^{b,*}
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
- \phi(q+1,t) + \phi(q,t).
\]

Likewise, with
\(\Delta_a \phi(q,t) = \phi(q-1,t) - \phi(q,t)\),

\[
\delta_t^{a,*}
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
- \phi(q-1,t) + \phi(q,t)
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
+ \phi(q,t) - \phi(q-1,t).
\]

Those are the exact closed-form distance formulas for the exponential-intensity
case. The remaining step is to approximate \(\phi\) in a way that exposes the
inventory term explicitly.

One-line symbolic verification of the bid-side first-order condition:

```python
import sympy as sp; δ, γ, κ, Δ = sp.symbols('δ γ κ Δ', positive=True); f = sp.exp(-κ*δ)*(1 - sp.exp(-γ*(δ + Δ))); sp.solve(sp.diff(f, δ), δ)[0]
```

Sympy returns
\(-\Delta + \gamma^{-1}\log(1+\gamma/\kappa)\), which is exactly the closed-form
\(\delta_t^{b,*}\) above.

## 5. Quadratic approximation and reservation price
Avellaneda and Stoikov close the system with the quadratic approximation

\[
\phi(q,t) \approx -\frac{1}{2} q^2 \gamma \sigma^2 (T-t).
\]

That is the finite-horizon continuation-value analogue of the Ho-Stoll certainty
equivalent. Substituting it into the bid and ask formulas gives

\[
\phi(q+1,t) - \phi(q,t)
= -\gamma \sigma^2 (T-t)\left(q + \frac{1}{2}\right),
\]

\[
\phi(q,t) - \phi(q-1,t)
= -\gamma \sigma^2 (T-t)\left(q - \frac{1}{2}\right).
\]

Therefore

\[
\delta_t^{b,*}
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
+ \gamma \sigma^2 (T-t)\left(q + \frac{1}{2}\right),
\]

\[
\delta_t^{a,*}
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
- \gamma \sigma^2 (T-t)\left(q - \frac{1}{2}\right).
\]

Now average the bid and ask quotes:

\[
b_t^* = S_t - \delta_t^{b,*}, \qquad
a_t^* = S_t + \delta_t^{a,*}.
\]

Their midpoint is

\[
r_t
= \frac{a_t^* + b_t^*}{2}
= S_t - q \gamma \sigma^2 (T-t).
\]

This is exactly the finite-horizon reservation price reported by
Avellaneda-Stoikov (2008, eq. (7)) and implemented in
`src/p2/hjb_solver.py::reservation_price`.

The structural interpretation is the same as in Ho-Stoll:

- If \(q>0\), the dealer is long inventory, so \(r_t < S_t\) and the quoting
  center shifts downward to encourage selling.
- If \(q<0\), the dealer is short inventory, so \(r_t > S_t\) and the quoting
  center shifts upward to encourage buying back.
- As \(t \to T\), the inventory-risk horizon disappears and \(r_t \to S_t\).

The Ho-Stoll and Avellaneda-Stoikov reservation prices are therefore the same
object at different levels of dynamic sophistication: Ho-Stoll derive it as a
one-period certainty equivalent; Avellaneda-Stoikov derive the finite-horizon
dynamic analogue through the HJB and then report it in eq. (7).

## 6. Optimal spread
Adding the optimal bid and ask distances gives the total spread

\[
\Delta_t^*
= \delta_t^{b,*} + \delta_t^{a,*}
= \gamma \sigma^2 (T-t)
+ \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right).
\]

This is the Avellaneda-Stoikov finite-horizon spread formula reported in eqs. (8)
and (9), depending on whether one writes total spread or half-spread. The first
term, \(\gamma \sigma^2 (T-t)\), is the direct descendant of the Ho-Stoll
inventory compensation. The second term,
\(\frac{2}{\gamma}\log(1+\gamma/\kappa)\), is the order-arrival elasticity markup
generated by the exponential intensity model.

The repo uses the one-sided half-spread

\[
h_t^* = \frac{\Delta_t^*}{2},
\]

so that quote placement is

\[
b_t^* = r_t - h_t^*, \qquad
a_t^* = r_t + h_t^*.
\]

That is why `src/p2/hjb_solver.py` exposes `optimal_total_spread(...)` and
`optimal_spread(...)` separately. The former matches the paper's total spread,
while the latter is the half-spread actually used to place bid and ask quotes in
code.

## 7. Small-\(\gamma\) limit
The risk-neutral limit is immediate from
\(\log(1+\gamma/\kappa) \sim \gamma/\kappa\):

\[
\Delta_t^*
\to \frac{2}{\kappa}
\qquad \text{as } \gamma \to 0.
\]

Hence the one-sided distance converges to

\[
h_t^* \to \frac{1}{\kappa}.
\]

So when risk aversion vanishes, the quote width is driven entirely by the decay
of the fill-intensity curve. This is why the test suite checks that
`optimal_spread(...)` approaches \(1/\kappa\) as \(\gamma \to 0\): it is the
risk-neutral benchmark implied directly by Avellaneda-Stoikov eqs. (8)-(9).

## 7.1 GLFT asymptotic quotes with a finite order size
Avellaneda-Stoikov closes in a terminal-horizon problem, so its quote width
carries the factor \(\gamma \sigma^2 \tau\) and widens without bound as the
horizon grows. Gueant, Lehalle, and Fernandez-Tapia (2013) solve the same
inventory problem in the stationary regime and give asymptotic quotes that no
longer depend on \(\tau\). They also carry the order size \(\Delta\) explicitly
instead of assuming unit orders, which matters here because the crypto contract
quotes 0.01 BTC against intensities calibrated per unit of depth.

Write the risk term as \(\gamma\Delta\) and define

\[
c_1 = \frac{1}{\gamma\Delta}\,\log\!\left(1 + \frac{\gamma\Delta}{\kappa}\right),
\qquad
c_2 = \sqrt{\frac{\gamma}{2 A \Delta \kappa}
  \left(1 + \frac{\gamma\Delta}{\kappa}\right)^{\frac{\kappa}{\gamma\Delta} + 1}}.
\]

The reservation price and one-sided distance are then

\[
r_t = s_t - q_t\,\sigma\,c_2,
\qquad
h^{\mathrm{GLFT}} = c_1 + \frac{\Delta}{2}\,\sigma\,c_2,
\]

with quotes \(r_t \pm h^{\mathrm{GLFT}}\). Two properties matter for the study.
The inventory skew is linear in \(q\) exactly as in section 5, but its
coefficient \(\sigma c_2\) is constant through the session rather than decaying
with \(\tau\). And \(c_1\) reduces to the risk-neutral \(1/\kappa\) of section 7
as \(\gamma\Delta \to 0\), so the whole expression is the finite-risk,
finite-size generalisation of the same quote width.

`src/p2/research_models.py:glft_quotes` implements these expressions directly,
computing \(c_2\) through `log1p` and an exponential of
\(\left(\kappa/(\gamma\Delta) + 1\right)\log(1 + \gamma\Delta/\kappa)\) so the
power stays finite when \(\gamma\Delta\) is small relative to \(\kappa\).

The preregistered order-flow variant adds a skew on top of these quotes. With
touch imbalance \(I \in [-1, 1]\) and a coefficient \(\beta\), both quotes shift
by \(\beta\,\text{tick}\,I\), in the spirit of the order-flow-imbalance price
response of Cont, Kukanov, and Stoikov (2014). The shift moves the pair
together, so it changes where the maker leans without changing the width that
\(c_1\) and \(c_2\) set.

## 8. Queue-reactive correction
The Avellaneda-Stoikov intensity model is one-dimensional: execution depends only
on quote distance \(\delta\). Huang, Lehalle, and Rosenbaum (2015) replace that
view with a state-conditioned queueing picture. During periods of constant
reference price, order-flow intensities depend on the current limit-order-book
state, not only on price distance. In their notation, the LOB evolves as a
continuous-time Markov jump process whose order-flow intensities are functions of
the current queue state (Huang-Lehalle-Rosenbaum 2015, Section 2.1 and lines
42-43, 93-95 of the arXiv version).

That is the conceptual basis of `src/p2/queue.py`. The repo does three concrete
things in that style:

1. `calibrate_queue_reactive(...)` fits depth-conditioned arrival, cancellation,
   and add rates.
2. `QueueReactiveModel.fill_probability(...)` maps queue depth and own FIFO
   position into a passive-order fill probability over a short horizon.
3. `expected_time_to_fill(...)` turns the same state into an execution-delay
   statistic.

To expose the correction analytically, enlarge the value function to include queue
state \(Q_t\):

\[
V(x,s,q,Q,t)
= \sup_{(\delta^b,\delta^a)}
\mathbb{E}_{t,x,s,q,Q}\left[-e^{-\gamma(X_T + q_T S_T)}\right].
\]

Here \(Q_t\) can stand for the queue depth at the posted level, our own FIFO
position, or a small vector containing both bid- and ask-side queue descriptors.
Following Huang-Lehalle-Rosenbaum, suppose execution intensity is conditioned on
that state:

\[
\lambda^b(\delta,Q) = A_b(Q)e^{-\kappa \delta}, \qquad
\lambda^a(\delta,Q) = A_a(Q)e^{-\kappa \delta}.
\]

With the exponential ansatz

\[
V(x,s,q,Q,t) = -e^{-\gamma(x + q s + \phi(q,Q,t))},
\]

the reduced HJB contains an additional queue-state generator term
\(\mathcal{L}^Q \phi\), plus fill terms of the form

\[
\max_{\delta^b} A_b(Q)e^{-\kappa \delta^b}
\left(
e^{-\gamma(\delta^b + \phi(q+1,Q_b^{\mathrm{fill}},t) - \phi(q,Q,t))} - 1
\right),
\]

and similarly on the ask side. The first-order condition is then

\[
\delta_{b,\mathrm{QR}}^*(Q)
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
- \phi(q+1,Q_b^{\mathrm{fill}},t) + \phi(q,Q,t),
\]

\[
\delta_{a,\mathrm{QR}}^*(Q)
= \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right)
+ \phi(q,Q,t) - \phi(q-1,Q_a^{\mathrm{fill}},t).
\]

There is one subtle but important point here. If \(Q\) only rescales the
intensity multiplicatively through \(A(Q)\), then \(A(Q)\) cancels out of the
pointwise first-order condition. The quote correction does not come from the
level of intensity alone. It comes from the queue-conditioned continuation value
\(\phi(q,Q,t)\): front-of-queue versus back-of-queue states change the marginal
value of waiting for a passive fill.

Write

\[
\phi(q,Q,t) = \phi_0(q,t) + \psi(q,Q,t),
\]

where \(\phi_0\) is the plain Avellaneda-Stoikov term and \(\psi\) is the
queue-state correction. Then

\[
\delta_{b,\mathrm{QR}}^*(Q)
= \delta_{b,\mathrm{AVS}}^*
+ \chi_b(Q,t),
\qquad
\chi_b(Q,t)
= -\psi(q+1,Q_b^{\mathrm{fill}},t) + \psi(q,Q,t),
\]

\[
\delta_{a,\mathrm{QR}}^*(Q)
= \delta_{a,\mathrm{AVS}}^*
+ \chi_a(Q,t),
\qquad
\chi_a(Q,t)
= \psi(q,Q,t) - \psi(q-1,Q_a^{\mathrm{fill}},t).
\]

Those \(\chi\) terms are the queue-reactive correction. When our passive order is
deep in queue, `expected_time_to_fill(...)` increases and
`fill_probability(...)` decreases; the continuation value of waiting worsens, so
\(\chi\) is typically positive and quotes widen. Near the front of the queue, the
opposite happens and quotes can tighten. This is exactly the Huang-Lehalle-
Rosenbaum intuition translated into an Avellaneda-Stoikov HJB.

For implementation, a convenient approximation is to map the queue module's
short-horizon fill statistics into the HJB coefficients directly. Over a small
time step \(\Delta t\), one can treat
\(A_b(Q) \approx \pi_b^{\mathrm{fill}}(Q) / \Delta t\) and
\(A_a(Q) \approx \pi_a^{\mathrm{fill}}(Q) / \Delta t\), where
\(\pi^{\mathrm{fill}}\) comes from `QueueReactiveModel.fill_probability(...)`, and
then use `expected_time_to_fill(...)` as a proxy for the waiting-cost component
inside \(\psi(q,Q,t)\). That does not solve the full queue-reactive HJB in closed
form, but it does produce a state-measurable correction with a direct path from
the paper's queueing logic to the repo's current engineering primitives.

## 9. Glosten-Milgrom adverse-selection coupling
Glosten and Milgrom (1985) study a different friction: not inventory risk but
informational asymmetry. The asset value is latent, traders may be informed, and
the dealer updates beliefs from order flow. The repo's
`src/p2/glosten_milgrom.py` implements the binary-state version:

\[
V \in \{v_L, v_H\}, \qquad
p_t = \mathbb{P}(V = v_H \mid \mathcal{F}_t).
\]

Let \(\mu \in (0,1)\) be the informed-trader fraction. Informed traders buy when
\(V=v_H\) and sell when \(V=v_L\). Uninformed traders buy and sell with equal
probability. Therefore

\[
\mathbb{P}(y_{t+1}=+1 \mid V=v_H) = \frac{1+\mu}{2}, \qquad
\mathbb{P}(y_{t+1}=+1 \mid V=v_L) = \frac{1-\mu}{2},
\]

\[
\mathbb{P}(y_{t+1}=-1 \mid V=v_H) = \frac{1-\mu}{2}, \qquad
\mathbb{P}(y_{t+1}=-1 \mid V=v_L) = \frac{1+\mu}{2},
\]

where \(y_{t+1}=+1\) denotes a buy order and \(y_{t+1}=-1\) a sell order. Bayes'
rule then gives the posterior after a buy:

\[
p_{t+1}^{(+)}
= \frac{p_t (1+\mu)}
{p_t (1+\mu) + (1-p_t)(1-\mu)},
\]

and after a sell:

\[
p_{t+1}^{(-)}
= \frac{p_t (1-\mu)}
{p_t (1-\mu) + (1-p_t)(1+\mu)}.
\]

This update is much cleaner in log-odds form. Define

\[
\ell_t = \log\frac{p_t}{1-p_t}, \qquad
\eta = \log\frac{1+\mu}{1-\mu}.
\]

Then

\[
\ell_{t+1} = \ell_t + y_{t+1}\eta.
\]

So each buy order increments log-odds by a constant and each sell order
decrements it by the same constant. That is exactly what
`GlostenMilgromLayer.posterior(...)` and
`GlostenMilgromLayer.update_posterior_sequence(...)` implement.

One-line symbolic verification of the buy-update odds ratio:

```python
import sympy as sp; p, mu = sp.symbols('p mu', positive=True); post = p*(1+mu)/(p*(1+mu) + (1-p)*(1-mu)); sp.simplify((post/(1-post)) / ((p/(1-p))*((1+mu)/(1-mu))))
```

Sympy returns \(1\), confirming that a buy multiplies odds by
\((1+\mu)/(1-\mu)\).

How does this couple to Avellaneda-Stoikov? The repo's first coupling is through
effective arrival intensities. The baseline Avellaneda-Stoikov intensity
\(A e^{-\kappa \delta}\) is tilted by the posterior:

\[
\lambda_{\mathrm{buy}}(\delta,p_t)
= A e^{-\kappa \delta}\left(1 - \mu + 2\mu p_t\right),
\]

\[
\lambda_{\mathrm{sell}}(\delta,p_t)
= A e^{-\kappa \delta}\left(1 - \mu + 2\mu (1-p_t)\right).
\]

In the repo's naming, `lambda_buy` is the arrival rate of buy market orders, so it
hits our ask. `lambda_sell` is the arrival rate of sell market orders, so it hits
our bid. At \(p_t = 1/2\), both factors equal \(1\), and the model collapses back
to the Avellaneda-Stoikov baseline exactly; this is tested explicitly in
`tests/test_glosten_milgrom.py`.

The more structural coupling is through the dealer's subjective fair value. The
posterior mean latent value is

\[
m_t = \mathbb{E}[V \mid \mathcal{F}_t]
= p_t v_H + (1-p_t) v_L.
\]

Replacing the exogenous mid \(S_t\) by the dealer's posterior mean gives the
coupled reservation price

\[
r_t^{\mathrm{AVS+GM}}
= m_t - q_t \gamma \sigma^2 (T-t).
\]

This last equation is an inference from the two source models, not a formula
printed verbatim in either paper. The logic is nonetheless direct:

- The Avellaneda-Stoikov term \(-q_t \gamma \sigma^2 (T-t)\) shifts the
  reservation price because inventory is risky.
- The Glosten-Milgrom term \(m_t - S_t\) shifts the reservation price because
  order flow reveals information about fundamental value.

Both act on the same object, the dealer's subjective indifference price, so to
first order they add. When order flow turns buy-heavy and \(p_t\) rises, the
posterior mean \(m_t\) rises, pushing the reservation price upward. A short
inventory position does the same through the inventory term. A long inventory
position pushes the other way. That is the cleanest way to read the interaction:
inventory risk and adverse selection are distinct frictions, but they both enter
as shifts in the quoting center.

## 10. Repo-level interpretation
The repo currently implements the three pieces at different depths:

1. `src/p2/hjb_solver.py` implements the closed-form Avellaneda-Stoikov
   reservation price and spread from eqs. (7)-(9).
2. `src/p2/queue.py` implements queue-state estimation and queue-conditioned fill
   approximations in the Huang-Lehalle-Rosenbaum style.
3. `src/p2/glosten_milgrom.py` implements the Bayesian posterior update and the
   posterior-tilted intensity layer.

What is already rigorous is the decomposition:

\[
\text{quote center}
= \text{mid or posterior mean}
- \text{inventory penalty}
+ \text{information correction},
\]

\[
\text{quote width}
= \text{Avellaneda-Stoikov half-spread}
+ \text{queue-state correction}.
\]

What is not yet claimed here is a single fully solved HJB with all of those
states simultaneously active. The appendix shows how the pieces line up
mathematically and why the current implementation choices in the repo are
coherent.

## References
- Ho, T. and H. Stoll (1981), *Optimal dealer pricing under transactions and return uncertainty*.
- Avellaneda, M. and S. Stoikov (2008), *High-frequency trading in a limit order book*.
- Huang, W., C.-A. Lehalle, and M. Rosenbaum (2015), *Simulating and analyzing order book data: The queue-reactive model*.
- Glosten, L. and P. Milgrom (1985), *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders*.
- Gueant, O., C.-A. Lehalle, and J. Fernandez-Tapia (2013), *Dealing with the inventory risk: a solution to the market making problem*.
- Cont, R., A. Kukanov, and S. Stoikov (2014), *The price impact of order book events*.
