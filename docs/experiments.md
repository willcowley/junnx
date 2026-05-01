# Experiments
These have been run via the optional `experiments` module. `data` dependencies are also required.
## 1D Regression

This reproduces Fig. 1a of [Rudner et al.](https://arxiv.org/abs/2312.17199v1). The comparison
with the Monte-Carlo (MC) Dropout technique of [Gal & Ghahramani](https://arxiv.org/abs/1506.02142)
indicates that whilst MC Dropout performs well around data (and indeed in interpolation regions)
in out-of-distribution (OOD) cases MC Dropout predictions become more difficult to reason about. The
FSVI predictions, meanwhile, revert towards the functional prior (in this case a Matern52 kernel) OOD.

The experiment uses a simple MLP architecture with 2 hidden layers of 128 neurons. An Adam optimiser
is used with a learning rate of 1e-3.
```bash
JAX_ENABLE_X64=1 poetry run python experiments/scripts/regression1d.py
```
![regression1d_sample_fsvi_1999.svg](assets/regression1d_sample_fsvi_1999.svg)
MC dropout overrides:
```bash
trainer/loss_fn=nll net=mcdropout_mlp
```
![regression1d_mcdropout_nll_1999.svg](assets/regression1d_mcdropout_nll_1999.svg)

## 2D Classification
This reproduces Fig. 1b and 1c of [Rudner et al.](https://arxiv.org/abs/2312.17199v1) (see also Fig. C.1).
Again the comparison to MC Dropout reveals that MC Dropout is overconfident away from regimes seen in
training.

The experiment uses a simple MLP architecture with 2 hidden layers of 32 neurons. An Adam optimiser
is used with a learning rate of 1e-3.
```bash
JAX_ENABLE_X64=1 poetry run python experiments/scripts/classification2d.py
```
![classification2d_samples_fsvi_3999.svg](assets/classification2d_samples_fsvi_3999.svg)
![classification2d_mcdropout_nll_3999.svg](assets/classification2d_mcdropout_nll_3999.svg)

## OOD Image Classification
### MNIST/FashionMNIST
This reproduces elements of Table 1. from [Rudner et al.](https://arxiv.org/abs/2312.17199v1) The models are
trained on the MNIST dataset. Accuracy and Expected Calibration Error (ECE) are measured on the validation
set. An entropy-based AUROC metric evaluated using FashionMNIST is used to determine OOD detection
performance. FSVI significantly outperforms MC dropout in this regard.

A LeNet
([LeCun et al.](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)) architecture is used
for both methods. An SGD optimiser with cosine learning rate decay is used, per Rudner et al.
```bash
JAX_ENABLE_X64=1 poetry run python experiments/scripts/mnist_ood.py -m net.key.seed=0,1,2,3,4,5,6,7,8,9
```
| Method     | Accuracy ↑  | ECE ↓      | AUROC (FM) ↑    |
|------------|-------------|------------|-----------------|
| FSVI       | 98.53±0.041 | 0.01±0.001 | **96.09±1.107** |
| MC Dropout | 98.33±0.090 | 0.01±0.001 | 85.53±4.066     |
