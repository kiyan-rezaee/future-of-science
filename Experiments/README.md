# Experiments

This directory contains experiment code, configs, and utilities used to run and reproduce the experiments for the FOS Benchmark.

## Training

First open the `train_baseline.py` script and edit the top configurations section to specify the baseline model you want to train and the domains that you have included in your FOS Benchmark. Then run `train_baseline.py`.

## Evaluation

First open the `evaluate_baseline.py` script and edit the top configurations section to specify the baseline model you want to evaluate, the negative sampling and neighbor sampling strategies that you want to use, and the domains that you have included in your FOS Benchmark. Then run `evaluate_baseline.py`.

## TODO
- include pre-trained model checkpoints