
# Fuzzy Integrals for Network Traffic Prediction

Python implementation of Choquet-based fuzzy integrals using the Power Measure for network traffic prediction.

This repository provides the source code and dataset used in the network traffic prediction experiments described in the paper *"Evaluating the Power Measure with Choquet-Based Generalizations with Different Applications"*.

## Project Structure

```text
.
├── main.py
├── models.py
├── experiment.py
├── plots.py
├── requirements.txt
|── compilado.csv
└── README.md
```

## Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements.txt`

## Installation

Clone the repository:

```bash
git clone https://github.com/dennerguti/fuzzy-integrals-network-traffic.git
cd fuzzy-integrals-network-traffic
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The experiments use network traffic data derived from the CIC-IDS2017 dataset.

The processed dataset is provided in:

```text
compilado.csv
```

No additional dataset download is required.

## Usage

Run the experiments using:

```bash
python main.py --input compilado.csv
```

The implementation includes:

- Poisson Moving Average (PMA) baseline.
- Choquet integral and its generalizations.
- Power Measure with different values of q.
- Sliding-window-based network traffic prediction.
- Mean Absolute Error (MAE) evaluation.
- Experimental results and visualizations.

## Reference

This repository accompanies the research paper:

**Evaluating the Power Measure with Choquet-Based Generalizations with Different Applications**

For theoretical foundations, methodology, and experimental analysis, please refer to the paper.
