"""Os dois modelos do projeto: RBF e MMQ.

Os dois são "lineares nos pesos": a saída é sempre  ŷ = Φ · w , em que
- Φ (Phi) é a matriz de projeto: cada linha é um instante e cada coluna é
  uma função de base calculada naquele instante;
- w é o vetor de pesos, que é o que o computador precisa descobrir.

O que muda de um modelo para o outro é QUAIS funções de base vão nas colunas.
Os pesos sempre saem da equação normal  ΦᵀΦ w = Φᵀy , resolvida com a
pseudo-inversa (np.linalg.pinv), que é mais estável que inverter ΦᵀΦ.
"""

import numpy as np


def resolver_equacao_normal(Phi, alvo):
    """Pesos que minimizam o erro quadrático: w = pinv(Φ) · y."""
    return np.linalg.pinv(Phi) @ alvo


# ---------------------------------------------------------------------------
# Modelo 1: RBF (Funções de Base Radial gaussianas) ao longo do tempo
# ---------------------------------------------------------------------------
#
#   ŷ(t) = w0 + Σ_j  w_j · exp( -(t - c_j)² / (2σ²) )
#
# Cada gaussiana é um "sino" centrado em c_j, com largura σ. Somando vários
# sinos, cada um multiplicado pelo seu peso, dá para desenhar uma curva suave
# que passa pelo meio da nuvem de pontos com ruído: isso é a filtragem.


def centros_uniformes(t, M):
    """M centros espalhados igualmente entre o primeiro e o último instante.

    A largura σ é o espaçamento entre centros vizinhos (regra prática do
    mini-projeto): assim os sinos se sobrepõem sem ficarem iguais.
    """
    centros = np.linspace(t[0], t[-1], M)
    sigma = centros[1] - centros[0]
    return centros, sigma


def matriz_rbf(t, centros, sigma):
    """Monta Φ: uma coluna de 1s (para w0) e uma coluna por gaussiana."""
    colunas = [np.ones(len(t))]
    for c in centros:
        colunas.append(np.exp(-((t - c) ** 2) / (2 * sigma**2)))
    return np.column_stack(colunas)


# ---------------------------------------------------------------------------
# Modelo 2: MMQ com o modelo dinâmico do enunciado
# ---------------------------------------------------------------------------
#
#   ŷ[n] = w0 + w1·y[n-1] + w2·y[n-2] + w3·u[n] + w4·u[n-1]
#
# É a forma linear de  y[n] = f(y[n-2], y[n-1], u[n], u[n-1]) . A saída de
# agora é prevista a partir das duas saídas anteriores (a "memória" do
# movimento) e da entrada de agora e da anterior.

NOMES_PESOS_MMQ = ["w0 (constante)", "w1 · y[n-1]", "w2 · y[n-2]", "w3 · u[n]", "w4 · u[n-1]"]


def matriz_regressores(s, u):
    """Monta Φ do modelo dinâmico e o vetor alvo.

    Começa em n = 2, porque o primeiro instante que tem dois valores
    anteriores é o terceiro. Por isso Φ tem N - 2 linhas.
    """
    Phi = np.column_stack([
        np.ones(len(s) - 2),  # 1       → w0
        s[1:-1],              # y[n-1]  → w1
        s[:-2],               # y[n-2]  → w2
        u[2:],                # u[n]    → w3
        u[1:-1],              # u[n-1]  → w4
    ])
    alvo = s[2:]              # y[n]
    return Phi, alvo


def prever_um_passo(w, s, u):
    """Previsão de um passo: usa os valores MEDIDOS de y[n-1] e y[n-2].

    O modelo sempre parte do dado real, então nunca acumula erro. Os dois
    primeiros instantes não têm previsão e ficam iguais ao dado medido.
    """
    Phi, _ = matriz_regressores(s, u)
    return np.concatenate([s[:2], Phi @ w])


def simular(w, s, u):
    """Simulação livre: o modelo usa as SUAS PRÓPRIAS previsões anteriores.

    Só os dois primeiros valores vêm dos dados. Daí em diante o modelo anda
    sozinho, recebendo apenas a entrada u. Isso testa se ele aprendeu a
    dinâmica do sistema ou se só estava copiando o valor anterior.
    """
    saida = np.zeros(len(s))
    saida[:2] = s[:2]
    for n in range(2, len(s)):
        saida[n] = w[0] + w[1] * saida[n - 1] + w[2] * saida[n - 2] + w[3] * u[n] + w[4] * u[n - 1]
    return saida


# ---------------------------------------------------------------------------
# Gradiente descendente: outro caminho para chegar nos mesmos pesos
# ---------------------------------------------------------------------------


def gradiente_descendente(Phi, alvo, eta, max_epocas=20_000, tol=1e-6):
    """Encontra w dando pequenos passos "morro abaixo" na função custo.

    Mesma estrutura da função vista em aula (notebook do robô):
    Custo:      J(w) = (1/N) · ||Φw - y||²           (o próprio EMQ)
    Gradiente:  ∇J  = (2/N) · Φᵀ (Φw - y)            (para onde o custo sobe)
    Passo:      w ← w - η · ∇J                       (anda no sentido contrário)

    η (eta) é a taxa de aprendizado e cada passo é uma época. O laço para
    quando o gradiente fica praticamente zero, ou seja, chegamos no fundo.
    Devolve os pesos e o histórico do custo (uma entrada por época).
    """
    N = len(alvo)
    w = np.zeros(Phi.shape[1])
    custos = []
    for epoca in range(max_epocas):
        erro = Phi @ w - alvo
        custos.append(np.mean(erro**2))               # J(w)
        gradiente = (2 / N) * Phi.T @ erro            # ∇J
        if np.linalg.norm(gradiente) < tol:           # chegou ao mínimo
            break
        w = w - eta * gradiente                       # passo
    return w, custos


def taxa_aprendizado_segura(Phi):
    """η = 1/λmax, em que λmax é o maior autovalor da Hessiana H = (2/N)ΦᵀΦ.

    O gradiente descendente só converge se η < 2/λmax. Usar metade desse
    limite é uma escolha segura (mesma regra do mini-projeto).
    """
    N = Phi.shape[0]
    autovalores = np.linalg.eigvalsh((2 / N) * Phi.T @ Phi)
    return 1 / autovalores[-1], autovalores[-1] / autovalores[0]


def padronizar(Phi, media, desvio):
    """Z-score em todas as colunas, menos a de 1s: z = (x - média) / desvio.

    Deixa todas as colunas na mesma escala, o que faz o gradiente
    descendente convergir bem mais rápido.
    """
    Z = Phi.copy()
    Z[:, 1:] = (Phi[:, 1:] - media) / desvio
    return Z


def despadronizar_pesos(w_z, media, desvio):
    """Converte pesos achados com dados padronizados para as unidades originais."""
    w = np.empty_like(w_z)
    w[1:] = w_z[1:] / desvio
    w[0] = w_z[0] - np.sum(w_z[1:] * media / desvio)
    return w
