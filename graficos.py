"""Funções que desenham e salvam os gráficos do projeto.

`rotulos` é um dicionário com o nome de cada saída para os eixos,
por exemplo {"x": "x(t)", "y": "y(t)"} ou {"temperatura": "temperatura (°C)"}.
"""

import matplotlib

matplotlib.use("Agg")  # só salva arquivos, não abre janela
import matplotlib.pyplot as plt
import numpy as np

from metricas import regressao_linear

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

COR_DADOS = "tab:blue"
COR_AJUSTE = "tab:red"


def _caixa(ax, texto):
    """Caixinha de texto no canto superior esquerdo (para EMQ e r)."""
    ax.text(0.01, 0.97, texto, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6"))


def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def dados(t, u, sinais, rotulos, caminho):
    """Figura 1: entrada u e as saídas medidas."""
    fig, axs = plt.subplots(len(sinais) + 1, 1, figsize=(11, 7), sharex=True)
    axs[0].step(t, u, where="post", color="k", lw=1)
    axs[0].set_ylabel("entrada u(t)")
    axs[0].set_title("Dados do sistema a ser identificado")
    for ax, (nome, s) in zip(axs[1:], sinais.items()):
        ax.scatter(t, s, s=6, alpha=0.4, color=COR_DADOS, label="medido (com ruído)")
        ax.set_ylabel(rotulos[nome])
        ax.legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("tempo (s)")
    _salvar(fig, caminho)


def como_funciona_rbf(t, s, Phi, w, centros, rotulo, janela, caminho):
    """Figura didática: cada gaussiana × seu peso, e a soma de todas."""
    sel = (t >= janela[0]) & (t <= janela[1])
    perto = [j for j, c in enumerate(centros) if janela[0] - 1 <= c <= janela[1] + 1]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for j in perto:
        axs[0].plot(t[sel], Phi[sel, j + 1], lw=1)
    axs[0].set_ylabel("φ_j(t)")
    axs[0].set_title("As funções de base: um 'sino' gaussiano em cada centro")
    axs[1].scatter(t[sel], s[sel], s=8, alpha=0.4, color=COR_DADOS, label="dados com ruído")
    for j in perto:
        axs[1].plot(t[sel], w[j + 1] * Phi[sel, j + 1], lw=0.8, alpha=0.7)
    axs[1].plot(t[sel], Phi[sel] @ w, color=COR_AJUSTE, lw=2.2, label="soma = curva ajustada")
    axs[1].set_ylabel(rotulo)
    axs[1].set_xlabel("tempo (s)")
    axs[1].set_title("Cada sino multiplicado pelo seu peso w_j, e a soma de todos")
    axs[1].legend(loc="upper right", fontsize=8)
    _salvar(fig, caminho)


def escolha_m(lista_m, erros, m_escolhido, rotulos, caminho):
    """Erro de treino e de validação para cada número de centros M."""
    fig, axs = plt.subplots(1, len(erros), figsize=(11, 4))
    for ax, nome in zip(axs, erros):
        ax.plot(lista_m, erros[nome]["treino"], "o-", label="treino (pontos usados no ajuste)")
        ax.plot(lista_m, erros[nome]["validacao"], "s-", label="validação (pontos separados)")
        ax.axvline(m_escolhido, color="k", ls="--", lw=1, label=f"M escolhido = {m_escolhido}")
        ax.set_xlabel("número de centros M")
        ax.set_ylabel("EMQ")
        ax.set_title(rotulos[nome])
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    _salvar(fig, caminho)


def ajuste(t, sinais, ajustes, metricas, rotulos, titulo, caminho, rotulo_ajuste):
    """Etapas 01/02/04: dados com ruído, curva do modelo e o EMQ."""
    fig, axs = plt.subplots(len(sinais), 1, figsize=(11, 6.5), sharex=True)
    for ax, nome in zip(axs, sinais):
        ax.scatter(t, sinais[nome], s=6, alpha=0.35, color=COR_DADOS, label="dados com ruído")
        ax.plot(t, ajustes[nome], color=COR_AJUSTE, lw=1.4, ls="--", label=rotulo_ajuste)
        ax.set_ylabel(rotulos[nome])
        ax.legend(loc="upper right", fontsize=8)
        _caixa(ax, f"EMQ = {metricas[nome]['EMQ']:.4f}")
    axs[0].set_title(titulo)
    axs[-1].set_xlabel("tempo (s)")
    _salvar(fig, caminho)


def regressao(sinais, ajustes, metricas, rotulos, titulo, caminho):
    """Etapa 03: medido × ajustado, reta de regressão, faixa ±√EMQ e r."""
    fig, axs = plt.subplots(len(sinais), 1, figsize=(11, 7))
    for ax, nome, cor in zip(axs, sinais, ["tab:red", "tab:orange"]):
        a, b = ajustes[nome], sinais[nome]
        inclinacao, intercepto = regressao_linear(a, b)
        grade = np.linspace(a.min(), a.max(), 100)
        reta = inclinacao * grade + intercepto
        faixa = np.sqrt(metricas[nome]["EMQ"])
        ax.scatter(a, b, s=6, alpha=0.3, color=COR_DADOS)
        ax.plot(grade, reta, "k--", lw=1.4,
                label=f"regressão: medido = {inclinacao:.3f}·ajustado {intercepto:+.4f}")
        ax.fill_between(grade, reta - faixa, reta + faixa, color=cor, alpha=0.15, label="±√EMQ")
        ax.set_xlabel(f"{rotulos[nome]} ajustado pelo modelo")
        ax.set_ylabel(f"{rotulos[nome]} medido")
        ax.legend(loc="lower right", fontsize=8)
        _caixa(ax, f"EMQ = {metricas[nome]['EMQ']:.4f}\nr = {metricas[nome]['r']:.4f}")
    axs[0].set_title(titulo)
    _salvar(fig, caminho)


def simulacao(t, sinais, simulados, metricas, rotulos, t_corte, caminho):
    """Simulação livre do modelo MMQ, com a divisão treino/teste marcada."""
    fig, axs = plt.subplots(len(sinais), 1, figsize=(11, 6.5), sharex=True)
    for ax, nome in zip(axs, sinais):
        ax.scatter(t, sinais[nome], s=6, alpha=0.35, color=COR_DADOS, label="dados com ruído")
        ax.plot(t, simulados[nome], color="tab:green", lw=1.4, label="simulação livre (MMQ)")
        ax.axvline(t_corte, color="k", ls=":", lw=1.2)
        ax.set_ylabel(rotulos[nome])
        ax.legend(loc="upper right", fontsize=8)
        _caixa(ax, f"EMQ = {metricas[nome]['EMQ']:.4f}\nr = {metricas[nome]['r']:.4f}")
    axs[0].set_title("Modelo MMQ andando sozinho (só recebe a entrada u)  |  à esquerda da linha: treino, à direita: teste")
    axs[-1].set_xlabel("tempo (s)")
    _salvar(fig, caminho)


def custo_gd(custos, emq_minimo, rotulos, caminho):
    """Curva do custo ao longo das épocas do gradiente descendente."""
    fig, axs = plt.subplots(1, len(custos), figsize=(11, 4))
    for ax, nome in zip(axs, custos):
        ax.plot(np.arange(1, len(custos[nome]) + 1), custos[nome], lw=1.5)
        ax.axhline(emq_minimo[nome], color="k", ls="--", lw=1, label="mínimo (equação normal)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("época")
        ax.set_ylabel("custo J(w) = EMQ")
        ax.set_title(f"Gradiente descendente: {rotulos[nome]}")
        ax.legend(fontsize=8)
    _salvar(fig, caminho)
