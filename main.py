"""Projeto Unidade II: identificação de um sistema dinâmico por ajuste de curvas.

Roda tudo em ordem, imprime as tabelas no terminal e salva os gráficos e
os números na pasta de resultados do conjunto escolhido.

    python main.py          # dados genéricos do professor
    python main.py cpu      # temperatura da CPU (coletada com coletar_cpu.py)
"""

import sys
from pathlib import Path

import numpy as np

import graficos
import modelos
from dados import carregar_dados
from metricas import emq, resumo

PASTA = Path(__file__).parent

# Cada conjunto de dados diz de onde vêm os dados, onde salvar os resultados
# e quantas gaussianas testar/usar na RBF.
CONJUNTOS = {
    "generico": {
        "arquivo": "dados_projeto_02.csv",
        "saidas": {"x": "x_ruido", "y": "y_ruido"},
        "rotulos": {"x": "x(t)", "y": "y(t)"},
        "resultados": "resultados",
        "M_testados": [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300],
        "M_escolhido": 50,           # justificado pela figura 3
        "janela_rbf": (0, 6),        # trecho mostrado na figura 2 (s)
    },
    "cpu": {
        "arquivo": "dados_cpu.csv",
        "saidas": {"temperatura": "temperatura", "ventoinha": "ventoinha"},
        "rotulos": {"temperatura": "temperatura (°C)", "ventoinha": "ventoinha (rpm)"},
        "resultados": "resultados_cpu",
        "M_testados": [25, 50, 100, 150, 200, 250, 300, 400, 500, 600],
        "M_escolhido": 300,
        "janela_rbf": (0, 90),
    },
}

FRACAO_VALIDACAO = 0.20   # pontos separados para escolher M
FRACAO_TREINO_MMQ = 0.70  # MMQ aprende nos primeiros 70% do tempo
SEMENTE = 42              # deixa o sorteio sempre igual (resultados reprodutíveis)

linhas_relatorio = []


def mostrar(texto=""):
    """Imprime no terminal e guarda a linha para o arquivo resultados.txt."""
    print(texto)
    linhas_relatorio.append(texto)


def tabela_metricas(titulo, metricas):
    largura = max(6, max(len(nome) for nome in metricas) + 1)
    mostrar(f"\n{titulo}")
    mostrar(f"  {'sinal':<{largura}}{'EMQ':>10}{'RMSE':>10}{'MAE':>10}{'r':>10}")
    for nome, m in metricas.items():
        mostrar(f"  {nome:<{largura}}{m['EMQ']:>10.4f}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}{m['r']:>10.4f}")


def main():
    nome_conjunto = sys.argv[1] if len(sys.argv) > 1 else "generico"
    if nome_conjunto not in CONJUNTOS:
        sys.exit(f"Conjunto desconhecido: {nome_conjunto}. Use um destes: {', '.join(CONJUNTOS)}")
    cfg = CONJUNTOS[nome_conjunto]
    rotulos = cfg["rotulos"]
    pasta_resultados = PASTA / cfg["resultados"]
    pasta_resultados.mkdir(exist_ok=True)

    t, u, sinais = carregar_dados(PASTA / "dados" / cfg["arquivo"], cfg["saidas"])
    N = len(t)
    nomes = list(sinais)

    mostrar("=" * 64)
    mostrar(f"DADOS ({nome_conjunto})")
    mostrar("=" * 64)
    mostrar(f"  {N} amostras, de t = {t[0]:.0f} s a t = {t[-1]:.0f} s, passo de {t[1] - t[0]:.2f} s")
    mostrar(f"  entrada u troca de valor {np.count_nonzero(np.diff(u))} vezes")
    graficos.dados(t, u, sinais, rotulos, pasta_resultados / "fig01_dados.png")

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
    cabecalho = f"  {'M':>5}{'σ (s)':>9}"
    for nome in nomes:
        largura = max(15, len(nome) + 12)
        cabecalho += f"{'EMQ treino ' + nome:>{largura}}{'EMQ valid. ' + nome:>{largura}}"
    mostrar(cabecalho)
    for M in cfg["M_testados"]:
        centros, sigma = modelos.centros_uniformes(t, M)
        Phi = modelos.matriz_rbf(t, centros, sigma)
        linha = f"  {M:>5}{sigma:>9.3f}"
        for nome, s in sinais.items():
            largura = max(15, len(nome) + 12)
            w = modelos.resolver_equacao_normal(Phi[idx_tr], s[idx_tr])
            e_tr, e_val = emq(s[idx_tr], Phi[idx_tr] @ w), emq(s[idx_val], Phi[idx_val] @ w)
            erros_m[nome]["treino"].append(e_tr)
            erros_m[nome]["validacao"].append(e_val)
            linha += f"{e_tr:>{largura}.4f}{e_val:>{largura}.4f}"
        mostrar(linha)
    graficos.escolha_m(cfg["M_testados"], erros_m, cfg["M_escolhido"], rotulos,
                       pasta_resultados / "fig03_escolha_M.png")

    # -------------------------------------------------------------------
    # Etapas 01 a 03: ajuste via RBF, EMQ, regressão linear e Pearson
    # -------------------------------------------------------------------
    M = cfg["M_escolhido"]
    centros, sigma = modelos.centros_uniformes(t, M)
    Phi_rbf = modelos.matriz_rbf(t, centros, sigma)
    ajuste_rbf, met_rbf = {}, {}
    for nome, s in sinais.items():
        w = modelos.resolver_equacao_normal(Phi_rbf, s)
        ajuste_rbf[nome] = Phi_rbf @ w
        met_rbf[nome] = resumo(s, ajuste_rbf[nome])
        if nome == nomes[-1]:
            graficos.como_funciona_rbf(t, s, Phi_rbf, w, centros, rotulos[nome], cfg["janela_rbf"],
                                       pasta_resultados / "fig02_como_funciona_rbf.png")

    mostrar(f"\n  M = {M} gaussianas, σ = {sigma:.3f} s, {M + 1} pesos por sinal")
    tabela_metricas("ETAPAS 01-03: AJUSTE VIA RBF (todos os pontos)", met_rbf)
    graficos.ajuste(t, sinais, ajuste_rbf, met_rbf, rotulos, "Etapas 01 e 02: ajuste via RBF e EMQ",
                    pasta_resultados / "fig04_rbf_ajuste.png", "ajuste via RBF")
    graficos.regressao(sinais, ajuste_rbf, met_rbf, rotulos,
                       "Etapa 03: regressão linear medido × ajustado (RBF)",
                       pasta_resultados / "fig05_rbf_regressao.png")

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
    largura = max(12, max(len(n) for n in nomes) + 1)
    mostrar(f"  {'peso':<16}" + "".join(f"{nome:>{largura}}" for nome in nomes))
    for i, rotulo in enumerate(modelos.NOMES_PESOS_MMQ):
        mostrar(f"  {rotulo:<16}" + "".join(f"{pesos_mmq[nome][i]:>{largura}.5f}" for nome in nomes))

    tabela_metricas("Previsão de um passo (todos os pontos)", met_mmq)
    tabela_metricas("Previsão de um passo (só o trecho de teste)", met_mmq_teste)
    tabela_metricas("Simulação livre (todos os pontos)", met_sim)
    tabela_metricas("Simulação livre (só o trecho de teste)", met_sim_teste)

    graficos.ajuste(t, sinais, previsto, met_mmq, rotulos,
                    "Etapa 04: ajuste via MMQ (previsão de um passo) e EMQ",
                    pasta_resultados / "fig06_mmq_ajuste.png", "ajuste via MMQ")
    graficos.regressao({k: v[2:] for k, v in sinais.items()}, {k: v[2:] for k, v in previsto.items()},
                       met_mmq, rotulos, "Etapa 04: regressão linear medido × ajustado (MMQ)",
                       pasta_resultados / "fig07_mmq_regressao.png")
    graficos.simulacao(t, sinais, simulado, met_sim, rotulos, t_corte,
                       pasta_resultados / "fig08_mmq_simulacao.png")

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
    graficos.custo_gd(custos, emq_minimo, rotulos, pasta_resultados / "fig09_gd_custo.png")

    # -------------------------------------------------------------------
    # Comparação final
    # -------------------------------------------------------------------
    mostrar("\n" + "=" * 64)
    mostrar("COMPARAÇÃO FINAL (todos os pontos)")
    mostrar("=" * 64)
    cabecalho = f"  {'modelo':<28}"
    for nome in nomes:
        largura = max(9, len(nome) + 5)
        cabecalho += f"{'EMQ ' + nome:>{largura}}{'r ' + nome:>{largura}}"
    mostrar(cabecalho)
    for rotulo, met in [("RBF (filtragem no tempo)", met_rbf),
                        ("MMQ, previsão de um passo", met_mmq),
                        ("MMQ, simulação livre", met_sim)]:
        linha = f"  {rotulo:<28}"
        for nome in nomes:
            largura = max(9, len(nome) + 5)
            linha += f"{met[nome]['EMQ']:>{largura}.4f}{met[nome]['r']:>{largura}.4f}"
        mostrar(linha)

    (pasta_resultados / "resultados.txt").write_text("\n".join(linhas_relatorio) + "\n", encoding="utf-8")
    print(f"\nGráficos e números salvos em {pasta_resultados}/")


if __name__ == "__main__":
    main()
