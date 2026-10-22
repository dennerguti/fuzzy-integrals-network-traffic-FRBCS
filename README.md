# Evaluating the Power Measure with Choquet-Based Generalizations with different applications

Python implementation of a Poisson-based weighted moving average (PMA) and nine Power Measure-based fuzzy integral variants, evaluated by one-step-ahead prediction MAE.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Run

Place your dataset locally (for example `data/compilado.csv`) and run:

```bash
python main.py --input data/compilado.csv --column 1 --output outputs
```

The default input column is the **second column** (zero-based index 1), as in the original notebook. The CSV is read with pandas defaults (header row and comma separator). Adapt loading if your dataset uses another format.

The experiment tests windows 5–70 in steps of 5, q=0.1–1.0 in steps of 0.1, and nine methods (Choquet, CC-min, CT, CF-AVG, CF-nonAVG, CF1F2, dCF, dCC, dXC), plus PMA. Predictions are rounded with Python `round` before MAE calculation, matching the notebook.

## Files

- `models.py`: PMA weights and fuzzy-integral formulas.
- `experiment.py`: sliding-window evaluation and CSV export.
- `plots.py`: plotting utilities and numerical method comparisons.
- `main.py`: command-line entry point.

Outputs include `resultados_completos.csv` (long format) and `resultados_por_funcao.csv` (wide format, semicolon separator and decimal comma).

## Optional charts

```python
import pandas as pd
from plots import comparar_metodos_q, plotar_metodos, plotar_funcao_por_q

df = pd.read_csv("outputs/resultados_completos.csv")
plotar_metodos(df, valores_q=[0.5])
plotar_funcao_por_q(df, metodo_escolhido="Choquet_dCC")
tabela = comparar_metodos_q(df, "Choquet_dCC", 0.2, "Choquet_CT", 0.1)
print(tabela)
```

**Reproducibility note:** The original mathematical expressions and rounding policy were preserved. The notebook did not include the underlying `compilado.csv`, so results cannot be independently reproduced without that dataset. No model-validation or methodology changes were introduced.
