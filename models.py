"""PMA and Power Measure-based fuzzy integrals (original formulas)."""
from math import exp, factorial, sqrt

# ---------------------------------------------------------------
# PMA - Média Móvel Ponderada baseada na distribuição de Poisson
# ---------------------------------------------------------------

def pesos_pma(tamanho_janela):
    """
    Calcula os pesos da Média Móvel Ponderada baseada
    na distribuição de Poisson.

    Parâmetros:
        tamanho_janela (int): quantidade de valores da janela.

    Retorna:
        list[float]: vetor com os pesos calculados.
    """
    pesos = []

    for indice in range(1, tamanho_janela + 1):
        peso_poisson = (
            tamanho_janela**indice
            * exp(-tamanho_janela)
            / factorial(indice)
        )

        pesos.append(peso_poisson * 2)

    return pesos


def calcular_pma(valores):
    """
    Calcula a Média Móvel Ponderada (PMA).

    Parâmetros:
        valores (list): valores da série temporal.

    Retorna:
        float: resultado da soma ponderada.
    """

    pesos = pesos_pma(len(valores))

    resultado = sum(
        valor * peso
        for valor, peso in zip(valores, pesos)
    )

    return resultado

def choquet_cardinalidade(v, style, q=1.0):
    """
    Integral usando a Power Measure (PM) sobre a cardinalidade:

        medida_fuzzy = (|A| / n) ** q

    Com q = 1.0 a medida se reduz à medida de cardinalidade (uniforme)
    original: |A| / n.

    style:
        0 = Choquet padrão
        1 = CC-min
        2 = CT-integral
        3 = CF-integral-AVG
        4 = CF-integral-nonAVG
        5 = CF1F2-integral
        6 = dCF-integral
        7 = dCC-integral
        8 = dXC-integral

    q: expoente da Power Measure (0 < q <= 1, conforme o artigo,
       testado tipicamente em 0.1, 0.2, ..., 1.0)
    """

    vetor = sorted(v)
    n = len(vetor)

    if n == 0:
        return 0.0

    soma = 0.0

    # styles em que o primeiro elemento (i == 0) deve entrar "puro",
    # sem passar pela fórmula F/δ (Definições 6, 8 e 9 do artigo)
    styles_com_primeiro_termo_puro = (5, 6, 7, 8)

    # styles cujo resultado final deve ser limitado a 1
    # (Definições 5 e 6 do artigo)
    styles_com_cap_em_1 = (3, 4, 5)

    for i in range(n):

        valor_atual = vetor[i]

        if i == 0:
            valor_anterior = 0.0
        else:
            valor_anterior = vetor[i - 1]

        diferenca_valores = (
            valor_atual - valor_anterior
        )

        # A_i contém do elemento i até o último
        cardinalidade = n - i

        # Power Measure: (|A| / n) ** q
        medida_fuzzy = (cardinalidade / n) ** q

        # --- tratamento do primeiro termo (x_(1) puro) ---
        if i == 0 and style in styles_com_primeiro_termo_puro:
            soma += valor_atual
            continue

        if style == 0:
            # Choquet padrão
            parcela = (
                diferenca_valores
                * medida_fuzzy
            )

        elif style == 1:
            # CC-min
            parcela = (
                min(valor_atual, medida_fuzzy)
                - min(valor_anterior, medida_fuzzy)
            )

        elif style == 2:
            # CT-integral
            denominador = (
                diferenca_valores
                + medida_fuzzy
                - (
                    diferenca_valores
                    * medida_fuzzy
                )
            )

            if denominador == 0:
                parcela = 0.0
            else:
                parcela = (
                    diferenca_valores
                    * medida_fuzzy
                ) / denominador

        elif style == 3:
            # CF-integral-AVG
            if diferenca_valores < medida_fuzzy:
                parcela = diferenca_valores
            else:
                parcela = min(
                    diferenca_valores / 2,
                    medida_fuzzy,
                )

        elif style == 4:
            # CF-integral-nonAVG
            if diferenca_valores == 0:
                parcela = 0.0

            elif diferenca_valores <= medida_fuzzy:
                parcela = (
                    diferenca_valores
                    + medida_fuzzy
                ) / 2

            else:
                parcela = min(
                    diferenca_valores / 2,
                    medida_fuzzy,
                )

        elif style == 5:
            # CF1F2-integral (só entra aqui para i >= 1)
            parcela = (
                sqrt(
                    valor_atual
                    * medida_fuzzy
                )
                - (
                    valor_anterior
                    * medida_fuzzy**2
                )
            )

        elif style == 6:
            # dCF-integral (só entra aqui para i >= 1)
            diferenca_raizes = (
                sqrt(valor_atual)
                - sqrt(valor_anterior)
            )

            parcela = (
                sqrt(
                    max(
                        0.0,
                        diferenca_raizes,
                    )
                )
                * medida_fuzzy
            )

        elif style == 7:
            # dCC-integral (só entra aqui para i >= 1)
            parte_atual = (
                min(
                    valor_atual,
                    medida_fuzzy,
                )
                * max(
                    valor_atual**2,
                    medida_fuzzy**2,
                )
            )

            parte_anterior = (
                min(
                    valor_anterior,
                    medida_fuzzy,
                )
                * max(
                    valor_anterior**2,
                    medida_fuzzy**2,
                )
            )

            diferenca_raizes = (
                sqrt(parte_atual)
                - sqrt(parte_anterior)
            )

            parcela = sqrt(
                max(
                    0.0,
                    diferenca_raizes,
                )
            )

        elif style == 8:
            # dXC-integral (só entra aqui para i >= 1)
            diferenca_raizes = (
                sqrt(
                    valor_atual
                    * medida_fuzzy
                )
                - sqrt(
                    valor_anterior
                    * medida_fuzzy
                )
            )

            parcela = sqrt(
                max(
                    0.0,
                    diferenca_raizes,
                )
            )

        else:
            raise ValueError(
                "O style deve estar entre 0 e 8."
            )

        soma += parcela

    if style in styles_com_cap_em_1:
        soma = min(1.0, soma)

    return soma
