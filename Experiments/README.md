# Experiments

This directory contains experiment code, configs, and utilities used to run and reproduce the experiments for the FOS Benchmark.

## Training

First open [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) and specify the baseline model you want to train and the domains that you have in your FOS Benchmark. Then run `train_baseline.py`.

## Evaluation

First open [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) and specify the baseline model you want to evaluate, the negative sampling and neighbor sampling strategies that you want to use, and the domains that you have included in your FOS Benchmark. Then run `evaluate_baseline.py`.

## Ablation Study

To evaluate the contribution of each semantic node feature, you can omit a feature from train/test data by specifying its name [in `config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml#L27)

## TODO
- include pre-trained baseline model checkpoints
