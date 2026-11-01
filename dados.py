"""Leitura do arquivo de dados do projeto.

O CSV usa vírgula como separador decimal ("0,030472"), então cada valor
é convertido trocando a vírgula por ponto antes de virar número.
"""

import csv

import numpy as np


def carregar_dados(caminho):
    """Lê o CSV e devolve quatro vetores: t, u, x, y.

    t: tempo (s), de 0 a 30 com passo de 0,02 s
    u: sinal de entrada (onda quadrada de +1 / -1)
    x: primeira saída medida, com ruído
    y: segunda saída medida, com ruído
    """
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    def coluna(nome):
        return np.array([float(linha[nome].replace(",", ".")) for linha in linhas])

    return coluna("t"), coluna("u"), coluna("x_ruido"), coluna("y_ruido")
