"""Sliding-window experiment and MAE evaluation."""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from models import calcular_pma, choquet_cardinalidade

NOMES_ESTILOS = {0: "Choquet", 1: "CC-min", 2: "CT", 3: "CF-AVG", 4: "CF-nonAVG", 5: "CF1F2", 6: "dCF", 7: "dCC", 8: "dXC"}


def executar_experimento(valores, tamanhos_janela=range(5, 71, 5), valores_q=None):
    """Evaluate original rounded one-step predictions for every window, q and style."""
    if valores_q is None:
        valores_q = [round(i / 10, 1) for i in range(1, 11)]
    resultados = []
    for janela in tamanhos_janela:
        if janela <= 0 or janela >= len(valores):
            raise ValueError(f"Janela {janela} inválida para série de {len(valores)} registros")
        print(f"Processando janela: {janela}")
        alvo = valores[janela:]
        janelas_teste = [valores[i - janela:i] for i in range(janela, len(valores))]
        predicoes_pma = [round(calcular_pma(jt)) for jt in janelas_teste]
        resultados.append({"tamanho_janela": janela, "q": None, "metodo": "PMA", "mae": mean_absolute_error(alvo, predicoes_pma)})
        for q in valores_q:
            for style, nome in NOMES_ESTILOS.items():
                preds = [round(choquet_cardinalidade(jt, style, q)) for jt in janelas_teste]
                resultados.append({"tamanho_janela": janela, "q": q, "metodo": f"Choquet_{nome}", "mae": mean_absolute_error(alvo, preds)})
    return pd.DataFrame(resultados)


def exportar_resultados(df_resultados, destino):
    """Export a wide CSV with Portuguese Excel separators."""
    df_choquet = df_resultados[df_resultados["metodo"] != "PMA"].pivot_table(index=["tamanho_janela", "q"], columns="metodo", values="mae").reset_index()
    df_choquet.columns.name = None
    df_pma = df_resultados[df_resultados["metodo"] == "PMA"][["tamanho_janela", "mae"]].rename(columns={"mae": "PMA"})
    tabela = df_choquet.merge(df_pma, on="tamanho_janela", how="left").sort_values(["tamanho_janela", "q"]).reset_index(drop=True)
    primeiras = ["tamanho_janela", "q", "PMA"]
    tabela = tabela[primeiras + [c for c in tabela.columns if c not in primeiras]]
    tabela.to_csv(destino, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    return tabela
