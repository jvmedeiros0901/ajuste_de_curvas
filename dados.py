"""Leitura dos arquivos de dados do projeto.

O CSV do professor usa vírgula como separador decimal ("0,030472"), então
cada valor é convertido trocando a vírgula por ponto antes de virar número.
O CSV da CPU já usa ponto, e a troca não muda nada nele.
"""

import csv

import numpy as np


def carregar_dados(caminho, colunas_saida):
    """Lê o CSV e devolve o tempo t, a entrada u e um dicionário com as saídas.

    colunas_saida diz qual coluna do arquivo vira cada saída, por exemplo
    {"x": "x_ruido", "y": "y_ruido"}.
    """
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    def coluna(nome):
        return np.array([float(linha[nome].replace(",", ".")) for linha in linhas])

    sinais = {nome: coluna(col) for nome, col in colunas_saida.items()}
    return coluna("t"), coluna("u"), sinais
