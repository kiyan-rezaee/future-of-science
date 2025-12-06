# The FOS (Future Of Science) Benchmark

This repository is the official implementation of the paper "**FOS: A Large-Scale Temporal Graph Benchmark for Scientific Interdisciplinary Link Prediction**" (arXiv: https://www.arxiv.org/abs/2511.18631).

It provides the code and artifacts needed to reproduce the FOS benchmark, run evaluations, and reproduce baseline experiments.

## Repository Contents
- FOS benchmark construction (dataset creation and preprocessing)
- Training and Evaluation pipeline for the temporal link prediction task
- Baseline model implementations and experiment scripts

## Quick Start
1. Clone this repository.
2. Install dependencies (`pip install -r requirements.txt`).
3. Edit [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml).
4. Use the scripts in [FOS_Benchmark](https://github.com/kiyan-rezaee/future-of-science/blob/main/FOS_Benchmark) to generate the dataset and benchmark. Or download our pre-built benchmark at [our HuggingFace dataset](https://huggingface.co/datasets/Morteza24/future-of-science) and place the `Art_Business/` directory in `./FOS_Benchmark/`.
5. Use the [Experiments](https://github.com/kiyan-rezaee/future-of-science/blob/main/Experiments) directory to train and evaluate and reproduce baseline results. And to perform ablation studies and analyse the real-world applications of FOS.

## TODO
- Test the consistency of codes after the refactor
- Provide CLI tools (argparse) for common workflows
- Add progress bars using tqdm for long-running operations

## Acknowledgements
We would like to express our gratitude to the creators of the following repository:

https://github.com/yule-BUAA/DyGLib/

Their efforts and contributions to open-source development have been invaluable to this project.
