"""Projeto Unidade II: identificação de um sistema dinâmico por ajuste de curvas.

Roda tudo em ordem, imprime as tabelas no terminal e salva os gráficos e
os números em resultados/.

    python main.py
"""

from pathlib import Path

import numpy as np

import graficos
import modelos
from dados import carregar_dados
from metricas import emq, resumo

PASTA = Path(__file__).parent
ARQUIVO_DADOS = PASTA / "dados" / "dados_projeto_02.csv"
PASTA_RESULTADOS = PASTA / "resultados"

M_TESTADOS = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
M_ESCOLHIDO = 50          # número de gaussianas (justificado pela figura 3)
FRACAO_VALIDACAO = 0.20   # pontos separados para escolher M
FRACAO_TREINO_MMQ = 0.70  # MMQ aprende nos primeiros 70% do tempo
SEMENTE = 42              # deixa o sorteio sempre igual (resultados reprodutíveis)

linhas_relatorio = []


def mostrar(texto=""):
    """Imprime no terminal e guarda a linha para o arquivo resultados.txt."""
    print(texto)
    linhas_relatorio.append(texto)


def tabela_metricas(titulo, metricas):
    mostrar(f"\n{titulo}")
    mostrar(f"  {'sinal':<6}{'EMQ':>10}{'RMSE':>10}{'MAE':>10}{'r':>10}")
    for nome, m in metricas.items():
        mostrar(f"  {nome:<6}{m['EMQ']:>10.4f}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}{m['r']:>10.4f}")


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    t, u, x, y = carregar_dados(ARQUIVO_DADOS)
    sinais = {"x": x, "y": y}
    N = len(t)

    mostrar("=" * 64)
    mostrar("DADOS")
    mostrar("=" * 64)
    mostrar(f"  {N} amostras, de t = {t[0]:.0f} s a t = {t[-1]:.0f} s, passo de {t[1] - t[0]:.2f} s")
    mostrar(f"  entrada u troca de sinal {np.count_nonzero(np.diff(u))} vezes")
    graficos.dados(t, u, x, y, PASTA_RESULTADOS / "fig01_dados.png")

    # -------------------------------------------------------------------
    # Escolha do número de gaussianas M
    # -------------------------------------------------------------------
    # Sorteia 20% dos pontos e esconde do ajuste. Se o modelo for bom de
    # verdade, também acerta esses pontos que não viu (validação).
    mostrar("\n" + "=" * 64)
    mostrar("RBF: ESCOLHA DO NÚMERO DE CENTROS M")
    mostrar("=" * 64)
    rng = np.random.default_rng(SEMENTE)
    ordem = rng.permutation(N)
    n_val = int(FRACAO_VALIDACAO * N)
    idx_val, idx_tr = ordem[:n_val], ordem[n_val:]

    erros_m = {nome: {"treino": [], "validacao": []} for nome in sinais}
    mostrar(f"  {'M':>5}{'σ (s)':>9}{'EMQ treino x':>15}{'EMQ valid. x':>15}{'EMQ treino y':>15}{'EMQ valid. y':>15}")
    for M in M_TESTADOS:
        centros, sigma = modelos.centros_uniformes(t, M)
        Phi = modelos.matriz_rbf(t, centros, sigma)
        linha = f"  {M:>5}{sigma:>9.3f}"
        for nome, s in sinais.items():
            w = modelos.resolver_equacao_normal(Phi[idx_tr], s[idx_tr])
            e_tr, e_val = emq(s[idx_tr], Phi[idx_tr] @ w), emq(s[idx_val], Phi[idx_val] @ w)
            erros_m[nome]["treino"].append(e_tr)
            erros_m[nome]["validacao"].append(e_val)
            linha += f"{e_tr:>15.4f}{e_val:>15.4f}"
        mostrar(linha)
    graficos.escolha_m(M_TESTADOS, erros_m, M_ESCOLHIDO, PASTA_RESULTADOS / "fig03_escolha_M.png")

    # -------------------------------------------------------------------
    # Etapas 01 a 03: ajuste via RBF, EMQ, regressão linear e Pearson
    # -------------------------------------------------------------------
    centros, sigma = modelos.centros_uniformes(t, M_ESCOLHIDO)
    Phi_rbf = modelos.matriz_rbf(t, centros, sigma)
    ajuste_rbf, met_rbf = {}, {}
    for nome, s in sinais.items():
        w = modelos.resolver_equacao_normal(Phi_rbf, s)
        ajuste_rbf[nome] = Phi_rbf @ w
        met_rbf[nome] = resumo(s, ajuste_rbf[nome])
        if nome == "y":
            graficos.como_funciona_rbf(t, s, Phi_rbf, w, centros, PASTA_RESULTADOS / "fig02_como_funciona_rbf.png")

    mostrar(f"\n  M = {M_ESCOLHIDO} gaussianas, σ = {sigma:.3f} s, {M_ESCOLHIDO + 1} pesos por sinal")
    tabela_metricas("ETAPAS 01-03: AJUSTE VIA RBF (todos os pontos)", met_rbf)
    graficos.ajuste(t, sinais, ajuste_rbf, met_rbf, "Etapas 01 e 02: ajuste via RBF e EMQ",
                    PASTA_RESULTADOS / "fig04_rbf_ajuste.png", "ajuste via RBF")
    graficos.regressao(sinais, ajuste_rbf, met_rbf, "Etapa 03: regressão linear medido × ajustado (RBF)",
                       PASTA_RESULTADOS / "fig05_rbf_regressao.png")

    # -------------------------------------------------------------------
    # Etapa 04: MMQ com o modelo y[n] = f(y[n-1], y[n-2], u[n], u[n-1])
    # -------------------------------------------------------------------
    mostrar("\n" + "=" * 64)
    mostrar("ETAPA 04: MMQ COM O MODELO DINÂMICO")
    mostrar("=" * 64)
    n_treino = int(FRACAO_TREINO_MMQ * N)
    t_corte = t[n_treino]
    mostrar(f"  treino: t < {t_corte:.1f} s  |  teste: t >= {t_corte:.1f} s")

    pesos_mmq, previsto, simulado = {}, {}, {}
    met_mmq, met_mmq_teste, met_sim, met_sim_teste = {}, {}, {}, {}
    for nome, s in sinais.items():
        Phi, alvo = modelos.matriz_regressores(s, u)
        linhas_treino = n_treino - 2  # Φ começa em n = 2
        w = modelos.resolver_equacao_normal(Phi[:linhas_treino], alvo[:linhas_treino])
        pesos_mmq[nome] = w
        previsto[nome] = modelos.prever_um_passo(w, s, u)
        simulado[nome] = modelos.simular(w, s, u)
        # métricas a partir de n = 2 (os dois primeiros pontos são copiados dos dados)
        met_mmq[nome] = resumo(s[2:], previsto[nome][2:])
        met_mmq_teste[nome] = resumo(s[n_treino:], previsto[nome][n_treino:])
        met_sim[nome] = resumo(s[2:], simulado[nome][2:])
        met_sim_teste[nome] = resumo(s[n_treino:], simulado[nome][n_treino:])

    mostrar("\n  Parâmetros livres identificados (equação normal):")
    mostrar(f"  {'peso':<16}{'x':>12}{'y':>12}")
    for i, rotulo in enumerate(modelos.NOMES_PESOS_MMQ):
        mostrar(f"  {rotulo:<16}{pesos_mmq['x'][i]:>12.5f}{pesos_mmq['y'][i]:>12.5f}")

    tabela_metricas("Previsão de um passo (todos os pontos)", met_mmq)
    tabela_metricas("Previsão de um passo (só o trecho de teste)", met_mmq_teste)
    tabela_metricas("Simulação livre (todos os pontos)", met_sim)
    tabela_metricas("Simulação livre (só o trecho de teste)", met_sim_teste)

    graficos.ajuste(t, sinais, previsto, met_mmq, "Etapa 04: ajuste via MMQ (previsão de um passo) e EMQ",
                    PASTA_RESULTADOS / "fig06_mmq_ajuste.png", "ajuste via MMQ")
    graficos.regressao({k: v[2:] for k, v in sinais.items()}, {k: v[2:] for k, v in previsto.items()}, met_mmq,
                       "Etapa 04: regressão linear medido × ajustado (MMQ)",
                       PASTA_RESULTADOS / "fig07_mmq_regressao.png")
    graficos.simulacao(t, sinais, simulado, met_sim, t_corte, PASTA_RESULTADOS / "fig08_mmq_simulacao.png")

    # -------------------------------------------------------------------
    # Conferência: gradiente descendente chega nos mesmos pesos?
    # -------------------------------------------------------------------
    mostrar("\n" + "=" * 64)
    mostrar("CONFERÊNCIA: GRADIENTE DESCENDENTE × EQUAÇÃO NORMAL (MMQ)")
    mostrar("=" * 64)
    custos, emq_minimo = {}, {}
    for nome, s in sinais.items():
        Phi, alvo = modelos.matriz_regressores(s, u)
        Phi_tr, alvo_tr = Phi[: n_treino - 2], alvo[: n_treino - 2]
        media, desvio = Phi_tr[:, 1:].mean(axis=0), Phi_tr[:, 1:].std(axis=0)
        Z = modelos.padronizar(Phi_tr, media, desvio)
        eta, condicionamento = modelos.taxa_aprendizado_segura(Z)
        w_z, custos[nome], epocas = modelos.gradiente_descendente(Z, alvo_tr, eta)
        w_gd = modelos.despadronizar_pesos(w_z, media, desvio)
        emq_minimo[nome] = emq(alvo_tr, Phi_tr @ pesos_mmq[nome])
        diferenca = np.max(np.abs(w_gd - pesos_mmq[nome]))
        mostrar(f"  {nome}: η = {eta:.3f}, condicionamento = {condicionamento:.1f}, "
                f"{epocas} épocas, maior diferença nos pesos = {diferenca:.1e}")
    graficos.custo_gd(custos, emq_minimo, PASTA_RESULTADOS / "fig09_gd_custo.png")

    # -------------------------------------------------------------------
    # Comparação final
    # -------------------------------------------------------------------
    mostrar("\n" + "=" * 64)
    mostrar("COMPARAÇÃO FINAL (todos os pontos)")
    mostrar("=" * 64)
    mostrar(f"  {'modelo':<28}{'EMQ x':>9}{'r x':>9}{'EMQ y':>9}{'r y':>9}")
    for rotulo, met in [("RBF (filtragem no tempo)", met_rbf),
                        ("MMQ, previsão de um passo", met_mmq),
                        ("MMQ, simulação livre", met_sim)]:
        mostrar(f"  {rotulo:<28}{met['x']['EMQ']:>9.4f}{met['x']['r']:>9.4f}"
                f"{met['y']['EMQ']:>9.4f}{met['y']['r']:>9.4f}")

    (PASTA_RESULTADOS / "resultados.txt").write_text("\n".join(linhas_relatorio) + "\n", encoding="utf-8")
    print(f"\nGráficos e números salvos em {PASTA_RESULTADOS}/")


if __name__ == "__main__":
    main()
