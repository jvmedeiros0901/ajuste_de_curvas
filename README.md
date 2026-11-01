# Identificação de um sistema dinâmico: RBF × MMQ

Projeto da Unidade II de **ECT3401 Computação Numérica** (UFRN), prof. Joilson B. A. Rego.

A partir de medições com ruído de um sistema físico (uma entrada `u` e duas saídas `x` e `y`),
o projeto ajusta dois modelos matemáticos e compara os resultados:

1. **RBF**: soma de funções gaussianas ao longo do tempo, que filtra o ruído do sinal.
2. **MMQ**: o modelo dinâmico `y[n] = w0 + w1·y[n-1] + w2·y[n-2] + w3·u[n] + w4·u[n-1]`,
   com os pesos obtidos pela equação normal (pseudo-inversa) e conferidos por gradiente descendente.

Para cada modelo são calculados o Erro Médio Quadrático (EMQ), RMSE, MAE, a regressão linear
entre os dados medidos e os ajustados e o coeficiente de correlação de Pearson.

![Ajuste via RBF](resultados/fig04_rbf_ajuste.png)

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

O programa imprime as tabelas no terminal e salva os gráficos e os números em `resultados/`.

## Resultados (todos os pontos)

| Modelo | EMQ x | r x | EMQ y | r y |
|---|---|---|---|---|
| RBF (M = 50 gaussianas) | 0,0097 | 0,727 | 0,0053 | 0,968 |
| MMQ, previsão de um passo | 0,0123 | 0,635 | 0,0075 | 0,954 |
| MMQ, simulação livre | 0,0150 | 0,520 | 0,0195 | 0,878 |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `main.py` | Roda todas as etapas em ordem |
| `dados.py` | Lê o CSV (que usa vírgula decimal) |
| `modelos.py` | Matrizes de projeto da RBF e do MMQ, equação normal, simulação e gradiente descendente |
| `metricas.py` | EMQ, RMSE, MAE, Pearson e regressão linear, implementados com as fórmulas |
| `graficos.py` | Gera as figuras |
| `dados/dados_projeto_02.csv` | Dados do projeto: `t`, `u`, `x_ruido`, `y_ruido` |
| `resultados/` | Figuras e `resultados.txt` gerados pelo `main.py` |
