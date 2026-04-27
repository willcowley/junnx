# Experiments

## 1D Regression
```bash
JAX_ENABLE_X64=1 poetry run python experiments/scripts/regression1d.py
```
![regression1d_sample_fsvi_1999.svg](assets/regression1d_sample_fsvi_1999.svg)
```bash
trainer/loss_fn=nll net=mcdropout_mlp
```
![regression1d_mcdropout_nll_1999.svg](assets/regression1d_mcdropout_nll_1999.svg)

## 2D Classification
```bash
JAX_ENABLE_X64=1 poetry run python experiments/scripts/classification2d.py
```
![classification2d_samples_fsvi_3999.svg](assets/classification2d_samples_fsvi_3999.svg)
```bash
trainer/loss_fn=nll net=mcdropout_mlp
```
![classification2d_mcdropout_nll_3999.svg](assets/classification2d_mcdropout_nll_3999.svg)

## OOD Image Classification
### MNIST/FashionMNIST
```bash
JAX_ENABLE_X64=1 poetry run python experiments/scripts/mnist_ood.py -m net.key.seed=0,1,2,3,4,5,6,7,8,9
```