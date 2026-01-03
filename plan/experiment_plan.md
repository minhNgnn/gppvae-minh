## Datasets / setup

Face dataset with views: 90L, 60L, 45L, 30L, 0, 30R, 45R, 60R, 90R. NOTE: Views should be observed to enable geometric constraint.

## Kernels:

- Full-rank kernel (as used in the original paper)
- RBF kernel (smoothness assumption)

## Experiments / data splits:

### 0. Sanity check (original-style i.i.d. split)

As a baseline replication, I also ran the original evaluation protocol with a standard random train/test split across images (all angles present in training). In this setting, the full-rank kernel achieved lower test MSE than the smooth kernels, which seems expected because the task does not require cross-view generalization. This motivates focusing on view-held-out regimes below.

### 1. Interpolation regime (in-range generalization)

- Train on a subset of discrete angles (e.g. {90L, 60L, 30L, 0, 30R, 60R, 90R})
- Evaluate on held-out intermediate views ({45L, 45R}), which lie strictly within the training range.
- Compare reconstruction performance across kernels

Corresponding research question: "Does imposing smoothness via structured kernels improve or degrade performance when test views lie within the training range?"

### 2. Extrapolation regime (out-of-range stress test)

- Train on a central subset of angles (e.g. {30L, 0, 30R} or {45L, 30L, 0, 30R, 45R})
- Evaluate on more extreme views ({60L, 90L, 60R, 90R})
- Compare how different kernels generalize under distribution shift

Corresponding research question: "How do different kernel-induced inductive biases influence extrapolation to unseen extreme views?"

### 3. Regularized full-rank control (optional)

- Evaluate a regularized full-rank kernel under both regimes
- This helps determine whether differences are due to overfitting/capacity or to the inductive bias imposed by kernel structure

Corresponding research question: "Are the observed differences primarily driven by model capacity or by kernel-induced inductive bias?”

## Training regime

- The VAE parameters and GP kernel parameters will be trained jointly, following the optimization setup in the original GPP-VAE paper.
- Learning rates comparable to those reported in the paper.
- Each experiment will have VAE trained for 200 - 500 epochs while GPP VAE trained for 500 - 1000 epochs
- 3 different random seeds and report mean and variance of the results.

## Evaluation metrics

- Mean Squared Error (MSE) on held-out views
- Optionally SSIM for qualitative support
- Per-angle MSE for extrapolation to highlight behavior at extreme views