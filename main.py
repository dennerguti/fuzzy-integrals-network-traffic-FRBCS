"""Run the BRACIS experiments on a local CSV file."""
import argparse
from pathlib import Path
import pandas as pd
from experiment import executar_experimento, exportar_resultados


def main():
    parser = argparse.ArgumentParser(description="BRACIS sliding-window aggregation experiments")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--column", type=int, default=1, help="Zero-based data column index (default: 1)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dados = pd.read_csv(args.input)
    if not 0 <= args.column < len(dados.columns):
        parser.error("Column index outside CSV columns")
    valores = pd.to_numeric(dados.iloc[:, args.column], errors="raise").tolist()
    resultados = executar_experimento(valores)
    resultados.to_csv(args.output / "resultados_completos.csv", index=False)
    exportar_resultados(resultados, args.output / "resultados_por_funcao.csv")
    print(f"Resultados salvos em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
