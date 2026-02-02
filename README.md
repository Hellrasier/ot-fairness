# Fairness with Optimal Transport

## Purpose
This repository aims to build an optimal transport (OT) library to compute and
enforce algorithmic fairness in regression and classification. The core idea is
to post-process model outputs so that their distributions match across sensitive
groups, while controlling the accuracy-fairness trade-off.

## Approach in this repo
We use OT-based post-processing, especially Wasserstein barycenters, to map
group-conditional prediction distributions to a common target distribution.

High-level pipeline:
1. Train a base model on training data (any regressor or classifier).
2. Use a calibration split to estimate group-conditional prediction
   distributions.
3. Compute OT maps (e.g., via Sinkhorn) to a barycenter distribution.
4. Apply group-specific monotone maps to test predictions.
5. Evaluate accuracy and fairness metrics and sweep trade-off parameters.

## Notebooks and code
- `fairness.ipynb` (regression): US Census (folktables) for income prediction in
  California. Filters by age, hours worked, and positive income; collapses small
  nationality groups into "other". Trains LightGBM, then applies OT
  post-processing using `equipy` Wasserstein barycenters on calibration data.
  Tracks unfairness and risk as a function of the fairness-accuracy trade-off.
- `fairness_class.ipynb` (classification): COMPAS and Bank Marketing datasets.
  Trains LightGBM, then applies OT post-processing using both `equipy` and the
  custom Sinkhorn pipeline in `sinkhorn_fairness.py`. Uses intersectional
  sensitive groups and logit-space maps for probabilistic outputs.
- `sinkhorn_fairness.py`: OT utilities for fairness via Sinkhorn barycenters on
  1D grids, stable barycentric projections, monotone map construction, and
  application of learned maps to new predictions.

## Fairness definition: Demographic Parity
Let $S$ be a sensitive attribute, $X$ non-sensitive features, and the prediction be
$Y$ (possibly randomized). Demographic Parity (DP) requires that the
distribution of predictions does not depend on S:

For all $s, s' \in S$ and all $t \in R$:
$$ P(Y \leq t | S = s) = P(Y \leq t | S = s'). $$

For binary classification, this reduces to:
$$ P(Y = 1 | S = s) = P(Y_hat = 1 | S = s') \quad \text{for all} \,\, s, s'. $$

## Empirical Risk Minimization (ERM)
Let D = {(x_i, s_i, y_i)}_{i=1}^n be the dataset.

Classification ERM (score function f; 0-1 or log loss):
min_f (1/n) * sum_{i=1}^n L(y_i, f(x_i))
with a classifier y_hat_i = 1{f(x_i) >= 1/2} when f outputs probabilities.

Regression ERM (squared loss):
min_f (1/n) * sum_{i=1}^n (y_i - f(x_i))^2.

Fairness can be enforced by adding a constraint (e.g., DP) or by post-processing
f(x_i) to equalize group-conditional prediction distributions.

## Papers followed (short comments)
- Chzhen et al., "Fair Regression with Wasserstein Barycenters" (arXiv:2006.07286):
  establishes that the optimal DP-fair regressor has a prediction distribution
  equal to the Wasserstein barycenter of group-conditional predictions, enabling
  a simple post-processing strategy.
- Xian et al., "Fair and Optimal Classification via Post-Processing" (ICML 2023,
  arXiv:2211.01528 / PMLR v202 xian23b): characterizes the DP fairness-accuracy
  trade-off for classification and shows an optimal post-processing rule based on
  Wasserstein barycenters of score distributions.
- Hu et al., "Fairness in Multi-Task Learning via Wasserstein Barycenters"
  (arXiv:2306.10155): extends strong DP to multi-task settings and derives
  closed-form optimal fair predictors with a post-processing estimator.
- Oneto and Chiappa, "Fairness in Machine Learning" (arXiv:2012.15816):
  survey and framework discussion; highlights OT as a way to impose distribution-
  level constraints and connects to causal reasoning.
- Le Gouic et al., "Projection to Fairness in Statistical Learning"
  (arXiv:2005.11720): defines projection to fairness as the closest fair estimator
  and uses OT to construct it as a post-processing step with quantified accuracy
  cost.
- Fernandes Machado et al., "EquiPy: Sequential Fairness using Optimal Transport
  in Python" (arXiv:2503.09866): a model-agnostic library for fairness across
  multiple sensitive variables, aligned with our use of `equipy`.

