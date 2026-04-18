# Avellaneda-Stoikov HJB Derivation

## 1. Control Problem
Let the mid price follow arithmetic Brownian motion

\[
dS_t = \sigma dW_t.
\]

The market maker posts bid and ask quotes \(b_t\) and \(a_t\), or equivalently distances

\[
\delta_t^b = S_t - b_t, \qquad \delta_t^a = a_t - S_t.
\]

Buy and sell market-order arrivals are modeled as independent Poisson processes with intensities

\[
\lambda^b(\delta_t^b) = A e^{-\kappa \delta_t^b}, \qquad
\lambda^a(\delta_t^a) = A e^{-\kappa \delta_t^a}.
\]

If the bid fills, inventory increases by one share and cash decreases by the bid price. If the ask fills, inventory decreases by one share and cash increases by the ask price. Denoting inventory by \(q_t\) and cash by \(X_t\),

\[
dq_t = dN_t^b - dN_t^a,
\]

\[
dX_t = -b_t \, dN_t^b + a_t \, dN_t^a.
\]

The terminal marked-to-market wealth is

\[
W_T = X_T + q_T S_T.
\]

The objective is to maximize exponential utility

\[
\sup_{(\delta^b,\delta^a)} \mathbb{E}_t\left[-e^{-\gamma W_T}\right].
\]

## 2. Value Function and HJB
Define the value function

\[
V(x,s,q,t) = \sup_{(\delta^b,\delta^a)} \mathbb{E}_{t,x,s,q}\left[-e^{-\gamma(X_T + q_T S_T)}\right].
\]

The dynamic programming equation is

\[
\partial_t V + \frac{1}{2}\sigma^2 \partial_{ss} V
+ \max_{\delta^b} \lambda^b(\delta^b)\left[V(x-b,s,q+1,t)-V(x,s,q,t)\right]
+ \max_{\delta^a} \lambda^a(\delta^a)\left[V(x+a,s,q-1,t)-V(x,s,q,t)\right] = 0
\]

with terminal condition

\[
V(x,s,q,T) = -e^{-\gamma(x + q s)}.
\]

## 3. Exponential Utility Ansatz
Following Avellaneda and Stoikov, use the ansatz

\[
V(x,s,q,t) = -e^{-\gamma(x + q s + \theta(q,t))}.
\]

This extracts the linear dependence on cash and marked-to-market inventory, leaving \(\theta(q,t)\) to capture the inventory penalty induced by risk aversion and the remaining horizon.

Differentiating the ansatz and substituting into the HJB gives

\[
\partial_t \theta(q,t) - \frac{1}{2}\gamma \sigma^2 q^2
+ \max_{\delta^b} \lambda^b(\delta^b)\left(e^{-\gamma(-\delta^b + \theta(q+1,t)-\theta(q,t))}-1\right)
+ \max_{\delta^a} \lambda^a(\delta^a)\left(e^{-\gamma(\delta^a + \theta(q-1,t)-\theta(q,t))}-1\right) = 0.
\]

## 4. First-Order Conditions
Consider the bid side. The bid contribution is

\[
f_b(\delta^b) = A e^{-\kappa \delta^b}\left(e^{-\gamma(-\delta^b + \theta(q+1,t)-\theta(q,t))}-1\right).
\]

Differentiating with respect to \(\delta^b\) and setting the derivative to zero yields

\[
\delta_t^{b,*} = \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right) - \theta(q+1,t) + \theta(q,t).
\]

Analogously for the ask side,

\[
\delta_t^{a,*} = \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right) + \theta(q,t) - \theta(q-1,t).
\]

Under the quadratic approximation used in the original paper, \(\theta(q,t)\) is approximately

\[
\theta(q,t) \approx -\frac{1}{2} q^2 \gamma \sigma^2 (T-t),
\]

which makes the inventory dependence explicit.

## 5. Reservation Price
The reservation price is the indifference price of holding one more unit of inventory. In the quadratic approximation it becomes

\[
r(q,t) = S_t - q \gamma \sigma^2 (T-t).
\]

This is Eq. (7) in the paper and is implemented directly in `src/p2/hjb_solver.py` as `reservation_price(...)`.

Interpretation:

- long inventory shifts the reservation price below mid, encouraging selling
- short inventory shifts it above mid, encouraging buying back
- the shift decays to zero as \(t \to T\)

## 6. Optimal Spread
Combining the optimal bid and ask offsets gives the total spread

\[
\Delta^*(t) = \delta_t^{b,*} + \delta_t^{a,*}
= \gamma \sigma^2 (T-t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right).
\]

This is the total distance between optimal ask and optimal bid. The code uses the one-sided half-spread

\[
\frac{\Delta^*(t)}{2},
\]

so quote placement is

\[
b_t = r_t - \frac{\Delta^*(t)}{2}, \qquad
a_t = r_t + \frac{\Delta^*(t)}{2}.
\]

This convention is implemented in `optimal_spread`, `optimal_bid`, and `optimal_ask`.

## 7. Small-\(\gamma\) Limit
As \(\gamma \to 0\),

\[
\ln\left(1 + \frac{\gamma}{\kappa}\right) \sim \frac{\gamma}{\kappa},
\]

so

\[
\Delta^*(t) \to \frac{2}{\kappa}.
\]

Therefore the half-spread converges to

\[
\frac{1}{\kappa},
\]

which is exactly the risk-neutral quoting distance used in the test suite.

## 8. Repo-Level Interpretation
The simulator in this repo keeps the analytical core fixed and extends the execution environment with three explicit practical choices:

1. Inventory hard limits are enforced by turning off bid quotes at \(+Q_{\max}\) and ask quotes at \(-Q_{\max}\).
2. Adverse selection is represented by an immediate mid-price jump after fills.
3. LOBSTER replay is kept as a top-of-book approximation and is not presented as a full queueing model.

Those extensions sit outside the closed-form derivation. The derivation above is only the analytical AVS core used by `hjb_solver.py`.
