# DrugCL

DrugCL is a contrastive-learning framework for drug–disease association prediction.
It learns drug representations from heterogeneous drug similarity / disease networks
and predicts novel drug–disease associations.

## Overview

- Contrastive learning over drug similarity graphs
- Multi-view drug/disease feature fusion
- End-to-end trainable with PyTorch Geometric

## Requirements

- Python > 3.8.1
- PyTorch
- PyTorch Geometric
- scikit-learn

## Installation

```bash
git clone https://github.com/YKONGCO/DrugCL.git
cd DrugCL
pip install -r DrugCL/requirements.txt
```

## Usage

```bash
cd DrugCL
python main.py
```

Dataset, hyper-parameters and device can be configured via the `config1` dict in
`main.py`. Available datasets: `lrssl`, `Ldataset`, `Gdataset`, `Cdataset`.

## Project structure

```
DrugCL/
├── config.py
├── data.py
├── evaluate.py
├── layers.py
├── main.py
├── model.py
├── train.py
├── utils.py
├── requirements.txt
└── raw_data/
```

## License

MIT License.
