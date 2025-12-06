# Experiments

This directory contains experiment code, configs, and utilities used to run and reproduce the experiments for the FOS Benchmark.

## Training

First open [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) and specify the baseline model you want to train and the domains that you have in your FOS Benchmark. Then run `train_baseline.py`.

## Evaluation

First open [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) and specify the baseline model you want to evaluate, the negative sampling and neighbor sampling strategies that you want to use, and the domains that you have included in your FOS Benchmark. Then run `evaluate_baseline.py`.

## Ablation Study

To evaluate the contribution of each semantic node feature, you can omit a feature from train/test data by specifying its name [in `config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml#L27)

## Discussion (Real-World Applications)

You can use the `Discussion_Real_World_Application.ipynb` notebook for identifying emerging interdisciplinary connections in scientific research. You can get high-confidence predictions of your preferred model where predicted cross-field links were appeared for the first time in test period. Just open the notebook and run the cells to generate two files:

- `*_discussion.csv`: This CSV file contain the model's predicted emerging cross-field links with their corresponding confidence.
- `*_OpenAlex_Results.json`: This JSON file includes the publications that contain the predicted cross-field links.

## TODO
- include pre-trained baseline model checkpoints
