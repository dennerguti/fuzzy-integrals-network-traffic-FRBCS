"""Visualizations and numerical comparison of experiment results."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def plotar_metodos(
    df_resultados,
    metodos_escolhidos=None,
    valores_q=None,
    salvar_pdf=None,
):
    # Todos os métodos como padrão
    if metodos_escolhidos is None:
        metodos_escolhidos = [
            "PMA",
            "Choquet_Choquet",
            "Choquet_CC-min",
            "Choquet_CT",
            "Choquet_CF-AVG",
            "Choquet_CF-nonAVG",
            "Choquet_CF1F2",
            "Choquet_dCF",
            "Choquet_dCC",
            "Choquet_dXC",
        ]

    # Todos os valores de q como padrão
    if valores_q is None:
        valores_q = [
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ]

    # Dados do PMA
    df_pma = df_resultados[
        (df_resultados["metodo"] == "PMA")
        & (df_resultados["metodo"].isin(metodos_escolhidos))
    ]

    # Dados dos métodos baseados em Choquet
    df_choquet = df_resultados[
        (df_resultados["metodo"].isin(metodos_escolhidos))
        & (df_resultados["metodo"] != "PMA")
        & (df_resultados["q"].isin(valores_q))
    ]

    # Gerar um gráfico para cada valor de q
    for q in valores_q:

        dados_q = df_choquet[
            df_choquet["q"] == q
        ].copy()

        # Renomear apenas para exibição no gráfico
        dados_q["metodo"] = dados_q["metodo"].replace({
            "Choquet_Choquet": "Choquet"
        })

        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 7))

        # Métodos Choquet
        if not dados_q.empty:
            sns.lineplot(
                data=dados_q,
                x="tamanho_janela",
                y="mae",
                hue="metodo",
                marker="o",
                markersize=8,
                linewidth=2.5,
                ax=ax,
            )

        # PMA
        if not df_pma.empty:
            ax.plot(
                df_pma["tamanho_janela"],
                df_pma["mae"],
                color="black",
                linestyle="--",
                marker="s",
                markersize=8,
                linewidth=2.5,
                label="PMA",
            )

        # Título em inglês
        ax.set_title(
            f"Comparison of Methods — q = {q}",
            fontsize=22,
            pad=15,
        )

        # Eixos em inglês e com fonte maior
        ax.set_xlabel(
            "Sliding Window Size",
            fontsize=19,
            labelpad=10,
        )

        ax.set_ylabel(
            "Mean Absolute Error (MAE)",
            fontsize=19,
            labelpad=10,
        )

        # Valores do eixo X
        ax.set_xticks(
            sorted(
                df_resultados[
                    "tamanho_janela"
                ].unique()
            )
        )

        # Aumentar fonte dos valores dos eixos
        ax.tick_params(
            axis="both",
            labelsize=16,
        )

        # Grade
        ax.grid(alpha=0.25)

        # Legenda em inglês e com fonte maior
        ax.legend(
            title="Method",
            title_fontsize=18,
            fontsize=16,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        # Ajustar layout
        fig.tight_layout()

        # Salvar em PDF
        if salvar_pdf is not None:

            # Se houver apenas um valor de q,
            # utiliza exatamente o nome informado
            if len(valores_q) == 1:
                nome_arquivo = salvar_pdf

            # Se houver vários q,
            # cria um arquivo para cada valor
            else:
                nome_arquivo = salvar_pdf.replace(
                    ".pdf",
                    f"_q{q}.pdf"
                )

            fig.savefig(
                nome_arquivo,
                format="pdf",
                bbox_inches="tight",
            )

            print(
                f"PDF salvo: {nome_arquivo}"
            )

        # Mostrar gráfico
        plt.show()

        # Fechar figura
        plt.close(fig)

def plotar_funcao_por_q(
    df_resultados,
    metodo_escolhido,
    incluir_pma=True,
):
    valores_q = [round(i / 10, 1) for i in range(1, 11)]

    dados_metodo = df_resultados[
        (df_resultados["metodo"] == metodo_escolhido)
        & (df_resultados["q"].isin(valores_q))
    ].copy()

    if dados_metodo.empty:
        print(f'Método "{metodo_escolhido}" não encontrado.')
        print("\nMétodos disponíveis:")
        print(df_resultados["metodo"].unique())
        return

    # Garante a ordem correta na legenda
    dados_metodo["q_legenda"] = pd.Categorical(
        dados_metodo["q"].map(lambda valor: f"{valor:.1f}"),
        categories=[f"{q:.1f}" for q in valores_q],
        ordered=True,
    )

    dados_metodo = dados_metodo.sort_values(
        ["q", "tamanho_janela"]
    )

    plt.figure(figsize=(13, 8))

    sns.lineplot(
        data=dados_metodo,
        x="tamanho_janela",
        y="mae",
        hue="q_legenda",
        palette="viridis",
        marker="o",
        linewidth=2,
    )

    if incluir_pma:
        dados_pma = (
            df_resultados[df_resultados["metodo"] == "PMA"]
            .sort_values("tamanho_janela")
        )

        plt.plot(
            dados_pma["tamanho_janela"],
            dados_pma["mae"],
            color="black",
            linestyle="--",
            marker="s",
            linewidth=3,
            label="PMA",
        )

    plt.title(
        f"Comportamento de {metodo_escolhido} por valor de q"
    )
    plt.xlabel("Tamanho da janela deslizante")
    plt.ylabel("MAE")

    plt.xticks(
        sorted(dados_metodo["tamanho_janela"].unique())
    )

    plt.grid(alpha=0.25)

    plt.legend(
        title="Valor de q",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()

def comparar_metodos_q(
    df_resultados,
    metodo_1,
    q_1,
    metodo_2,
    q_2,
    incluir_pma=True,
    salvar_pdf=None,
):
    # Filtrar os dados do primeiro método
    dados_1 = df_resultados[
        (df_resultados["metodo"] == metodo_1)
        & (df_resultados["q"].round(1) == round(q_1, 1))
    ].copy()

    # Filtrar os dados do segundo método
    dados_2 = df_resultados[
        (df_resultados["metodo"] == metodo_2)
        & (df_resultados["q"].round(1) == round(q_2, 1))
    ].copy()

    # Verificar se as combinações existem
    if dados_1.empty or dados_2.empty:
        print("Uma das combinações não foi encontrada.")

        print("\nMétodos disponíveis:")
        print(df_resultados["metodo"].unique())

        print("\nValores de q disponíveis:")
        print(sorted(df_resultados["q"].dropna().unique()))
        return

    # Nomes mais limpos para exibição no gráfico
    nomes_exibicao = {
        "Choquet_Choquet": "Choquet",
        "Choquet_CC-min": "CC-min",
        "Choquet_CT": "CT",
        "Choquet_CF-AVG": "CF-AVG",
        "Choquet_CF-nonAVG": "CF-nonAVG",
        "Choquet_CF1F2": "CF1F2",
        "Choquet_dCF": "dCF",
        "Choquet_dCC": "dCC",
        "Choquet_dXC": "dXC",
    }

    nome_1 = nomes_exibicao.get(metodo_1, metodo_1)
    nome_2 = nomes_exibicao.get(metodo_2, metodo_2)

    # Criar rótulos para a legenda
    dados_1["comparacao"] = f"{nome_1} — q={q_1:.1f}"
    dados_2["comparacao"] = f"{nome_2} — q={q_2:.1f}"

    # Juntar os dois conjuntos
    dados_comparacao = pd.concat(
        [dados_1, dados_2],
        ignore_index=True
    ).sort_values(
        ["comparacao", "tamanho_janela"]
    )

    # Criar figura
    fig, ax = plt.subplots(figsize=(13, 8))

    # Plotar os dois métodos
    sns.lineplot(
        data=dados_comparacao,
        x="tamanho_janela",
        y="mae",
        hue="comparacao",
        style="comparacao",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=9,
        ax=ax,
    )

    # Adicionar PMA
    if incluir_pma:
        dados_pma = (
            df_resultados[
                df_resultados["metodo"] == "PMA"
            ]
            .sort_values("tamanho_janela")
        )

        ax.plot(
            dados_pma["tamanho_janela"],
            dados_pma["mae"],
            color="black",
            linestyle="--",
            marker="s",
            markersize=8,
            linewidth=2.5,
            label="PMA",
        )

    # Título em inglês
    ax.set_title(
        f"Comparison: {nome_1} (q={q_1:.1f}) "
        f"vs. {nome_2} (q={q_2:.1f})",
        fontsize=22,
        pad=15,
    )

    # Eixos em inglês
    ax.set_xlabel(
        "Sliding Window Size",
        fontsize=19,
    )

    ax.set_ylabel(
        "Mean Absolute Error (MAE)",
        fontsize=19,
    )

    # Valores do eixo X
    ax.set_xticks(
        sorted(
            dados_comparacao[
                "tamanho_janela"
            ].unique()
        )
    )

    # Aumentar fonte dos valores dos eixos
    ax.tick_params(
        axis="both",
        labelsize=16,
    )

    # Grade
    ax.grid(alpha=0.25)

    # Legenda maior e em inglês
    ax.legend(
        title="Method",
        title_fontsize=18,
        fontsize=17,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    # Ajustar layout
    fig.tight_layout()

    # Salvar PDF ANTES de mostrar o gráfico
    if salvar_pdf is not None:
        fig.savefig(
            salvar_pdf,
            format="pdf",
            bbox_inches="tight",
        )

        print(f"PDF salvo: {salvar_pdf}")

    # Mostrar gráfico
    plt.show()

    # Fechar figura
    plt.close(fig)

    # -----------------------------------------
    # Comparação numérica por janela
    # -----------------------------------------

    tabela = dados_1[
        ["tamanho_janela", "mae"]
    ].merge(
        dados_2[
            ["tamanho_janela", "mae"]
        ],
        on="tamanho_janela",
        suffixes=("_1", "_2")
    )

    # Determinar o melhor método
    tabela["melhor_metodo"] = tabela.apply(
        lambda linha: (
            f"{nome_1} (q={q_1:.1f})"
            if linha["mae_1"] < linha["mae_2"]
            else (
                f"{nome_2} (q={q_2:.1f})"
                if linha["mae_2"] < linha["mae_1"]
                else "Empate"
            )
        ),
        axis=1
    )

    # Diferença absoluta de MAE
    tabela["diferenca_mae"] = (
        tabela["mae_1"]
        - tabela["mae_2"]
    ).abs()

    return tabela

