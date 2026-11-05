"""Projeto Unidade II: identificação da temperatura da CPU por ajuste de curvas.

Roda todas as etapas do enunciado em ordem, imprime as tabelas no terminal e
salva os gráficos e os números na pasta resultados/.

    python main.py

Os dados (dados/dados_cpu.csv) foram coletados com coletar_cpu.py:
t = tempo (s), u = carga na CPU (0 ou 1), temperatura = saída medida (°C).
"""

from pathlib import Path

import numpy as np

import graficos
import modelos
from metricas import emq, resumo

PASTA = Path(__file__).parent
ARQUIVO_DADOS = PASTA / "dados" / "dados_cpu.csv"
PASTA_RESULTADOS = PASTA / "resultados"

M_TESTADOS = [25, 50, 100, 150, 200, 250, 300, 400, 500, 600]
M = 300                   # número de gaussianas da RBF (justificado por extra_escolha_M.png)
JANELA_RBF = (20, 70)     # trecho mostrado em extra_como_funciona_rbf.png (s)
FRACAO_VALIDACAO = 0.20   # pontos separados para escolher M
FRACAO_TREINO_MMQ = 0.70  # MMQ aprende nos primeiros 70% do tempo
SEMENTE = 42              # deixa o sorteio sempre igual (resultados reprodutíveis)

linhas_relatorio = []


def mostrar(texto=""):
    """Imprime no terminal e guarda a linha para o arquivo resultados.txt."""
    print(texto)
    linhas_relatorio.append(texto)


def titulo(texto):
    mostrar("\n" + "=" * 64)
    mostrar(texto)
    mostrar("=" * 64)


def cabecalho_metricas():
    mostrar(f"  {'':<32}{'EMQ':>10}{'RMSE':>10}{'MAE':>10}{'r':>10}")


def linha_metricas(nome, m):
    mostrar(f"  {nome:<32}{m['EMQ']:>10.4f}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}{m['r']:>10.4f}")


def main():
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    dados = np.genfromtxt(ARQUIVO_DADOS, delimiter=",", names=True)
    t, u, temp = dados["t"], dados["u"], dados["temperatura"]
    N = len(t)

    titulo("DADOS: TEMPERATURA DA CPU")
    mostrar(f"  {N} amostras, de t = {t[0]:.0f} s a t = {t[-1]:.0f} s, passo de {t[1] - t[0]:.2f} s")
    mostrar(f"  entrada u troca de valor {np.count_nonzero(np.diff(u))} vezes")
    graficos.dados(t, u, temp, PASTA_RESULTADOS / "fig01_dados.png")

    # -------------------------------------------------------------------
    # Escolha do número de gaussianas M
    # -------------------------------------------------------------------
    # Sorteia 20% dos pontos e esconde do ajuste. Se o modelo for bom de
    # verdade, também acerta esses pontos que não viu (validação).
    titulo("RBF: ESCOLHA DO NÚMERO DE CENTROS M")
    ordem = np.random.default_rng(SEMENTE).permutation(N)
    n_val = int(FRACAO_VALIDACAO * N)
    idx_val, idx_tr = ordem[:n_val], ordem[n_val:]

    erros_treino, erros_validacao = [], []
    mostrar(f"  {'M':>5}{'σ (s)':>9}{'EMQ treino':>14}{'EMQ validação':>16}")
    for m in M_TESTADOS:
        centros, sigma = modelos.centros_uniformes(t, m)
        Phi = modelos.matriz_rbf(t, centros, sigma)
        w = modelos.resolver_equacao_normal(Phi[idx_tr], temp[idx_tr])
        erros_treino.append(emq(temp[idx_tr], Phi[idx_tr] @ w))
        erros_validacao.append(emq(temp[idx_val], Phi[idx_val] @ w))
        mostrar(f"  {m:>5}{sigma:>9.3f}{erros_treino[-1]:>14.4f}{erros_validacao[-1]:>16.4f}")
    graficos.escolha_m(M_TESTADOS, erros_treino, erros_validacao, M,
                       PASTA_RESULTADOS / "extra_escolha_M.png")

    # -------------------------------------------------------------------
    # Etapas 01 a 03: ajuste via RBF, EMQ, regressão linear e Pearson
    # -------------------------------------------------------------------
    centros, sigma = modelos.centros_uniformes(t, M)
    Phi_rbf = modelos.matriz_rbf(t, centros, sigma)
    w_rbf = modelos.resolver_equacao_normal(Phi_rbf, temp)
    ajuste_rbf = Phi_rbf @ w_rbf
    met_rbf = resumo(temp, ajuste_rbf)

    titulo("ETAPAS 01-03: AJUSTE VIA RBF (todos os pontos)")
    mostrar(f"  M = {M} gaussianas, σ = {sigma:.3f} s, {M + 1} pesos")
    cabecalho_metricas()
    linha_metricas("RBF", met_rbf)

    graficos.ajuste(t, temp, ajuste_rbf, met_rbf, "Etapa 01: ajuste via RBF", "ajuste via RBF",
                    PASTA_RESULTADOS / "fig02_rbf_ajuste.png", mostrar_emq=False)
    graficos.ajuste(t, temp, ajuste_rbf, met_rbf, "Etapa 02: EMQ do ajuste via RBF", "ajuste via RBF",
                    PASTA_RESULTADOS / "fig03_rbf_emq.png")
    graficos.regressao(temp, ajuste_rbf, met_rbf, "Etapa 03: regressão linear medido × ajustado (RBF)",
                       PASTA_RESULTADOS / "fig04_rbf_regressao.png")
    graficos.como_funciona_rbf(t, temp, Phi_rbf, w_rbf, centros, JANELA_RBF,
                               PASTA_RESULTADOS / "extra_como_funciona_rbf.png")

    # -------------------------------------------------------------------
    # Etapa 04: MMQ com o modelo y[n] = f(y[n-1], y[n-2], u[n], u[n-1])
    # -------------------------------------------------------------------
    titulo("ETAPA 04: MMQ COM O MODELO DINÂMICO")
    n_treino = int(FRACAO_TREINO_MMQ * N)
    t_corte = t[n_treino]
    mostrar(f"  treino: t < {t_corte:.1f} s  |  teste: t >= {t_corte:.1f} s")

    Phi, alvo = modelos.matriz_regressores(temp, u)
    Phi_tr, alvo_tr = Phi[: n_treino - 2], alvo[: n_treino - 2]   # Φ começa em n = 2
    w_mmq = modelos.resolver_equacao_normal(Phi_tr, alvo_tr)
    previsto = modelos.prever_um_passo(w_mmq, temp, u)
    simulado = modelos.simular(w_mmq, temp, u)

    mostrar("\n  Parâmetros livres identificados (equação normal):")
    for rotulo, peso in zip(modelos.NOMES_PESOS_MMQ, w_mmq):
        mostrar(f"  {rotulo:<16}{peso:>12.5f}")

    # métricas a partir de n = 2 (os dois primeiros pontos são copiados dos dados)
    met_mmq = resumo(temp[2:], previsto[2:])
    met_sim = resumo(temp[2:], simulado[2:])
    mostrar("")
    cabecalho_metricas()
    linha_metricas("Previsão de um passo (todos)", met_mmq)
    linha_metricas("Previsão de um passo (só teste)", resumo(temp[n_treino:], previsto[n_treino:]))
    linha_metricas("Simulação livre (todos)", met_sim)
    linha_metricas("Simulação livre (só teste)", resumo(temp[n_treino:], simulado[n_treino:]))

    graficos.ajuste(t, temp, previsto, met_mmq, "Etapa 04: ajuste via MMQ (previsão de um passo) e EMQ",
                    "ajuste via MMQ", PASTA_RESULTADOS / "fig05_mmq_ajuste.png")
    graficos.regressao(temp[2:], previsto[2:], met_mmq, "Etapa 04: regressão linear medido × ajustado (MMQ)",
                       PASTA_RESULTADOS / "fig06_mmq_regressao.png")
    graficos.simulacao(t, temp, simulado, met_sim, t_corte, PASTA_RESULTADOS / "extra_mmq_simulacao.png")

    # -------------------------------------------------------------------
    # Conferência: gradiente descendente chega nos mesmos pesos?
    # -------------------------------------------------------------------
    titulo("CONFERÊNCIA: GRADIENTE DESCENDENTE × EQUAÇÃO NORMAL (MMQ)")
    media, desvio = Phi_tr[:, 1:].mean(axis=0), Phi_tr[:, 1:].std(axis=0)
    Z = modelos.padronizar(Phi_tr, media, desvio)
    eta, condicionamento = modelos.taxa_aprendizado_segura(Z)
    w_z, custos = modelos.gradiente_descendente(Z, alvo_tr, eta)
    w_gd = modelos.despadronizar_pesos(w_z, media, desvio)
    diferenca = np.max(np.abs(w_gd - w_mmq))
    mostrar(f"  η = {eta:.3f}, condicionamento = {condicionamento:.1f}, {len(custos)} épocas, "
            f"maior diferença nos pesos = {diferenca:.1e}")
    graficos.custo_gd(custos, emq(alvo_tr, Phi_tr @ w_mmq), PASTA_RESULTADOS / "extra_gd_custo.png")

    # -------------------------------------------------------------------
    # Comparação final
    # -------------------------------------------------------------------
    titulo("COMPARAÇÃO FINAL (todos os pontos)")
    cabecalho_metricas()
    linha_metricas("RBF (filtragem no tempo)", met_rbf)
    linha_metricas("MMQ, previsão de um passo", met_mmq)
    linha_metricas("MMQ, simulação livre", met_sim)

    (PASTA_RESULTADOS / "resultados.txt").write_text("\n".join(linhas_relatorio) + "\n", encoding="utf-8")
    print(f"\nGráficos e números salvos em {PASTA_RESULTADOS}/")


if __name__ == "__main__":
    main()
