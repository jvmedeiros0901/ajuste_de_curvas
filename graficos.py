"""Funções que desenham e salvam os gráficos do projeto (temperatura da CPU)."""

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
ROTULO = "temperatura (°C)"


def _caixa(ax, texto):
    """Caixinha de texto no canto superior esquerdo (para EMQ e r)."""
    ax.text(0.01, 0.97, texto, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6"))


def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def dados(t, u, temp, caminho):
    """Figura 1: entrada u e a temperatura medida."""
    fig, (ax_u, ax_t) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_u.step(t, u, where="post", color="k", lw=1)
    ax_u.set_ylabel("entrada u(t)")
    ax_u.set_title("Dados do sistema a ser identificado")
    ax_t.scatter(t, temp, s=6, alpha=0.4, color=COR_DADOS, label="medido (com ruído)")
    ax_t.set_ylabel(ROTULO)
    ax_t.set_xlabel("tempo (s)")
    ax_t.legend(loc="upper right", fontsize=8)
    _salvar(fig, caminho)


def ajuste(t, temp, ajustado, metricas, titulo, rotulo_ajuste, caminho, mostrar_emq=True):
    """Etapas 01/02/04: dados com ruído, curva do modelo e (opcional) o EMQ."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(t, temp, s=6, alpha=0.35, color=COR_DADOS, label="dados com ruído")
    ax.plot(t, ajustado, color=COR_AJUSTE, lw=1.4, ls="--", label=rotulo_ajuste)
    if mostrar_emq:
        _caixa(ax, f"EMQ = {metricas['EMQ']:.4f}")
    ax.set_title(titulo)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel(ROTULO)
    ax.legend(loc="upper right", fontsize=8)
    _salvar(fig, caminho)


def regressao(temp, ajustado, metricas, titulo, caminho):
    """Etapa 03: medido × ajustado, reta de regressão, faixa ±√EMQ e r."""
    inclinacao, intercepto = regressao_linear(ajustado, temp)
    grade = np.linspace(ajustado.min(), ajustado.max(), 100)
    reta = inclinacao * grade + intercepto
    faixa = np.sqrt(metricas["EMQ"])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(ajustado, temp, s=6, alpha=0.3, color=COR_DADOS)
    ax.plot(grade, reta, "k--", lw=1.4,
            label=f"regressão: medido = {inclinacao:.3f}·ajustado {intercepto:+.4f}")
    ax.fill_between(grade, reta - faixa, reta + faixa, color=COR_AJUSTE, alpha=0.15, label="±√EMQ")
    _caixa(ax, f"EMQ = {metricas['EMQ']:.4f}\nr = {metricas['r']:.4f}")
    ax.set_title(titulo)
    ax.set_xlabel(f"{ROTULO} ajustado pelo modelo")
    ax.set_ylabel(f"{ROTULO} medido")
    ax.legend(loc="lower right", fontsize=8)
    _salvar(fig, caminho)


# ---------------------------------------------------------------------------
# Figuras extras (não pedidas no enunciado, ficam fora do relatório)
# ---------------------------------------------------------------------------


def escolha_m(lista_m, erros_treino, erros_validacao, m_escolhido, caminho):
    """Erro de treino e de validação para cada número de centros M."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(lista_m, erros_treino, "o-", label="treino (pontos usados no ajuste)")
    ax.plot(lista_m, erros_validacao, "s-", label="validação (pontos separados)")
    ax.axvline(m_escolhido, color="k", ls="--", lw=1, label=f"M escolhido = {m_escolhido}")
    ax.set_title(ROTULO)
    ax.set_xlabel("número de centros M")
    ax.set_ylabel("EMQ")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    _salvar(fig, caminho)


def como_funciona_rbf(t, temp, Phi, w, centros, janela, caminho):
    """Figura didática: cada gaussiana × seu peso, e a soma de todas."""
    sel = (t >= janela[0]) & (t <= janela[1])
    perto = [j for j, c in enumerate(centros) if janela[0] - 1 <= c <= janela[1] + 1]
    fig, (ax_base, ax_soma) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for j in perto:
        ax_base.plot(t[sel], Phi[sel, j + 1], lw=1)
    ax_base.set_ylabel("φ_j(t)")
    ax_base.set_title("As funções de base: um 'sino' gaussiano em cada centro")
    ax_soma.scatter(t[sel], temp[sel], s=8, alpha=0.4, color=COR_DADOS, label="dados com ruído")
    for j in perto:
        ax_soma.plot(t[sel], w[j + 1] * Phi[sel, j + 1], lw=0.8, alpha=0.7)
    ax_soma.plot(t[sel], Phi[sel] @ w, color=COR_AJUSTE, lw=2.2, label="soma = curva ajustada")
    ax_soma.set_ylabel(ROTULO)
    ax_soma.set_xlabel("tempo (s)")
    ax_soma.set_title("Cada sino multiplicado pelo seu peso w_j, e a soma de todos")
    ax_soma.legend(loc="upper right", fontsize=8)
    _salvar(fig, caminho)


def simulacao(t, temp, simulado, metricas, t_corte, caminho):
    """Simulação livre do modelo MMQ, com a divisão treino/teste marcada."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(t, temp, s=6, alpha=0.35, color=COR_DADOS, label="dados com ruído")
    ax.plot(t, simulado, color="tab:green", lw=1.4, label="simulação livre (MMQ)")
    ax.axvline(t_corte, color="k", ls=":", lw=1.2)
    _caixa(ax, f"EMQ = {metricas['EMQ']:.4f}\nr = {metricas['r']:.4f}")
    ax.set_title("Modelo MMQ andando sozinho (só recebe a entrada u)  |  à esquerda da linha: treino, à direita: teste")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel(ROTULO)
    ax.legend(loc="upper right", fontsize=8)
    _salvar(fig, caminho)


def custo_gd(custos, emq_minimo, caminho):
    """Curva do custo ao longo das épocas do gradiente descendente."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(1, len(custos) + 1), custos, lw=1.5)
    ax.axhline(emq_minimo, color="k", ls="--", lw=1, label="mínimo (equação normal)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("época")
    ax.set_ylabel("custo J(w) = EMQ")
    ax.set_title(f"Gradiente descendente: {ROTULO}")
    ax.legend(fontsize=8)
    _salvar(fig, caminho)
