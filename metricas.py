"""Métricas para medir a qualidade de um ajuste.

Todas são calculadas "na mão", com as fórmulas vistas em aula, em vez de
usar funções prontas de bibliotecas de estatística.

Convenção: `real` são os dados medidos e `ajustado` é o que o modelo devolveu.
O erro (ou resíduo) de cada ponto é  e = real - ajustado.
"""

import numpy as np


def emq(real, ajustado):
    """Erro Médio Quadrático (EMQ, ou MSE em inglês): média de e²."""
    erro = real - ajustado
    return np.mean(erro**2)


def rmse(real, ajustado):
    """Raiz do EMQ. Fica na mesma unidade dos dados, por isso é mais fácil de ler."""
    return np.sqrt(emq(real, ajustado))


def mae(real, ajustado):
    """Erro Médio Absoluto: média de |e|. Pesa menos os erros grandes que o EMQ."""
    return np.mean(np.abs(real - ajustado))


def pearson(a, b):
    """Coeficiente de correlação de Pearson (r), entre -1 e 1.

    r = Σ (a - ā)(b - b̄) / sqrt( Σ (a - ā)² · Σ (b - b̄)² )

    r = 1: os pontos caem exatamente numa reta que sobe.
    r = 0: não há relação linear nenhuma.
    """
    da = a - np.mean(a)
    db = b - np.mean(b)
    return np.sum(da * db) / np.sqrt(np.sum(da**2) * np.sum(db**2))


def regressao_linear(a, b):
    """Reta b ≈ inclinacao · a + intercepto pelo método dos mínimos quadrados.

    São as fórmulas que saem de derivar o erro quadrático e igualar a zero
    (Exemplo 6.1 da apostila), escritas com as médias de a e b.
    """
    da = a - np.mean(a)
    inclinacao = np.sum(da * (b - np.mean(b))) / np.sum(da**2)
    intercepto = np.mean(b) - inclinacao * np.mean(a)
    return inclinacao, intercepto


def resumo(real, ajustado):
    """Junta as métricas num dicionário, para imprimir e plotar."""
    return {
        "EMQ": emq(real, ajustado),
        "RMSE": rmse(real, ajustado),
        "MAE": mae(real, ajustado),
        "r": pearson(ajustado, real),
    }
