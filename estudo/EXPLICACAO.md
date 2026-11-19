# O projeto, do começo ao fim

Este texto conta o projeto na ordem em que as coisas acontecem. Cada palavra nova é explicada na hora em que aparece.

---

## 1. O que o professor pediu

O professor pediu três coisas:

1. medir um sistema de verdade;
2. achar uma fórmula que imite o que foi medido, de dois jeitos diferentes;
3. dizer qual dos dois jeitos errou menos.

O documento dele divide isso em etapas (Etapa 01, 02, 03 e 04). Elas vão aparecer ao longo do texto.

---

## 2. O sistema que medimos: o seu notebook

O "sistema" é o processador do seu notebook.

Um script (o arquivo `coletar_cpu.py`) fez o processador alternar entre dois estados, a cada 30 segundos:

- **trabalhando:** fazendo conta sem parar, como uma pessoa correndo;
- **descansando:** parado, como uma pessoa sentada.

Chamamos isso de **carga**, com a letra `u`. Ela só tem dois valores: `u = 1` quando está trabalhando e `u = 0` quando está descansando.

Enquanto isso, o script lia a **temperatura** do processador a cada meio segundo. Todo processador tem um termômetro embutido no próprio chip, e o Linux mostra o valor dele num arquivo de texto. O script só lia esse arquivo.

No fim, tudo foi salvo numa tabela (`dados/dados_cpu.csv`), com uma linha a cada meio segundo:

| tempo (s) | carga u | temperatura (°C) |
|---|---|---|
| 0,0 | 0 | 57 |
| 0,5 | 0 | 57 |
| 1,0 | 0 | 57 |

São 1501 linhas, cobrindo 750 segundos (12 minutos e meio).

---

## 3. O ruído

O termômetro do processador só mostra **números inteiros**: 66, 67, 68. Ele nunca mostra 66,5.

Veja este trecho da tabela, em que a temperatura de verdade ficou parada:

| tempo (s) | temperatura medida |
|---|---|
| 103,5 | 67 |
| 104,0 | 66 |
| 104,5 | 66 |
| 105,0 | 66 |
| 105,5 | 67 |

A temperatura de verdade estava perto de 66,5. Como o termômetro precisa escolher um inteiro, a medida fica pulando entre 66 e 67.

Esse pula-pula se chama **ruído**: a diferença entre o valor de verdade e o valor que o termômetro mostra. Por isso a medida é **quase** o valor real, mas não exatamente.

---

## 4. Gráfico dos dados

Primeira figura do projeto: [fig01_dados.png](../resultados/fig01_dados.png)

- **Em cima:** a carga. Sobe para 1 (trabalhando) e desce para 0 (descansando).
- **Embaixo:** a temperatura. Cada pontinho azul é uma linha da tabela.

Dá para ver o padrão que se repete: quando o processador começa a trabalhar, a temperatura dispara até uns 85 °C, depois cai para uns 67 °C e fica ali. Quando ele descansa, ela desce devagar até uns 56 °C. E repete.

**No código** ([main.py](../main.py), linhas 57 e 58), a tabela é lida e separada em três listas: `t` (tempo), `u` (carga) e `temp` (temperatura). A linha 64 desenha a figura.

---

## 5. O que é "ajustar"

Os pontinhos azuis têm ruído. O que queremos é uma **linha lisa passando pelo meio deles**, mostrando como a temperatura se comportou de verdade, sem o pula-pula.

Encontrar essa linha se chama **ajustar uma curva**.

A linha é um **palpite** do valor de verdade, já que o valor de verdade ninguém conhece. A partir daqui existem dois números para cada instante:

- **medido:** o que o termômetro mostrou;
- **calculado:** o que a nossa linha diz.

---

## 6. Como saber se um palpite é bom: erro e MMQ

Para escolher a linha, é preciso um jeito de dizer se um palpite é bom ou ruim. Vamos ver isso com 5 medidas: **67, 66, 66, 66, 67**.

**Erro** é quanto o palpite errou em uma medida:

```
erro = medido − calculado
```

Se o termômetro mostrou 67 e o palpite foi 66, o erro é 67 − 66 = 1.

O erro pode ser negativo: medido 66 e palpite 67 dá 66 − 67 = −1. Se eu somasse 1 com −1, daria 0, como se eu não tivesse errado. Para isso não acontecer, cada erro é **elevado ao quadrado**, porque qualquer número ao quadrado fica positivo: 1² = 1 e (−1)² = 1.

Depois eu **somo** os erros ao quadrado de todas as medidas. Isso dá um número só, que diz o quanto o palpite errou no total.

Testando três palpites com as mesmas 5 medidas:

**Palpite 66**

```
67 − 66 =  1   →  1
66 − 66 =  0   →  0
66 − 66 =  0   →  0
66 − 66 =  0   →  0
67 − 66 =  1   →  1
                soma = 2
```

**Palpite 67**

```
67 − 67 =  0   →  0
66 − 67 = −1   →  1
66 − 67 = −1   →  1
66 − 67 = −1   →  1
67 − 67 =  0   →  0
                soma = 3
```

**Palpite 66,4** (a média das 5 medidas)

```
67 − 66,4 =  0,6   →  0,36
66 − 66,4 = −0,4   →  0,16
66 − 66,4 = −0,4   →  0,16
66 − 66,4 = −0,4   →  0,16
67 − 66,4 =  0,6   →  0,36
                    soma = 1,2
```

O melhor palpite é o de **menor soma**: 66,4. Nenhum outro número dá menos que 1,2.

Esse jeito de escolher, ficando com o palpite de menor soma dos erros ao quadrado, se chama **Método dos Mínimos Quadrados (MMQ)**. "Mínimos" porque queremos a menor soma. "Quadrados" porque os erros são elevados ao quadrado.

---

## 7. Um número só não basta

No exemplo das 5 medidas, a temperatura estava parada. Mas nos dados completos ela vai de 53 °C a 88 °C, subindo e descendo o tempo todo.

Se eu usasse um palpite só para os 750 segundos, o melhor seria a média de tudo, uns 64,6 °C. Ele erraria uns 23 graus quando a temperatura está em 88.

Precisamos de um palpite que **mude com o tempo**: alto quando a temperatura está alta, baixo quando está baixa. A partir daqui o projeto mostra dois jeitos de fazer isso.

---

## 8. Etapa 01: o primeiro jeito, a RBF

A RBF monta a linha somando vários "montes". Na aula, o monte se chama **função gaussiana**, e RBF quer dizer **Função de Base Radial**. As gaussianas são as peças da RBF.

### Um monte sozinho

Desenho: [um_morrinho_explicacao.png](um_morrinho_explicacao.png)

Um monte tem um **centro** (o instante em que ele é mais alto) e uma **altura**. No desenho, o centro está em 100 s e a altura é 10:

- em 100 s, bem no topo, ele vale 10;
- em 102,5 s, descendo, vale 6,1;
- em 105 s, quase no pé, vale 1,4;
- em 110 s, fora do monte, vale 0.

A fórmula do monte, igual à do notebook do professor (`Exemplo_RBF.ipynb`), é:

```
φ(t) = exp( −(t − c)² / (2σ²) )
```

`c` é o centro e `σ` (sigma) é a largura. Não precisa decorar: a ideia é que ela vale 1 no centro e vai caindo até zero longe dele.

### Vários montes somados

Desenho: [tres_morrinhos_explicacao.png](tres_morrinhos_explicacao.png)

São três montes (azul, verde, laranja), e a linha vermelha é a **soma** deles. Em cada instante, somo quanto cada monte vale ali. Em 100 s: 0,68 + 10 + 0,41 = 11,1.

A soma tem um formato que nenhum monte tem sozinho. Mudando a altura de cada monte, a soma pode ter o formato que eu quiser.

### Aplicado à temperatura

A RBF coloca **300 montes**, um a cada 2,5 segundos, de 0 a 750 s. A altura de cada um se chama **peso**. Somando os 300 montes, cada um com seu peso, sai a linha.

**Quem escolhe os pesos é o MMQ**, a mesma regra do 66,4: os pesos que deixam a menor soma dos erros ao quadrado entre a linha e as medidas. Só que agora são 300 números de uma vez (na verdade 301, porque tem um valor de base extra), e o computador não testa um por um. Ele usa uma fórmula que dá a resposta direto, a **equação normal**, que também está no notebook do professor.

Então a RBF e o MMQ trabalham juntos:

- a RBF diz **quais são as peças** (os montes);
- o MMQ diz **quanto usar de cada peça** (os pesos).

O resultado é a Etapa 01: [fig02_rbf_ajuste.png](../resultados/fig02_rbf_ajuste.png). Os pontinhos azuis são as medidas, e a linha vermelha tracejada é a soma dos 300 montes.

**No código** ([main.py](../main.py), linhas 91 a 94):

```python
centros, sigma = modelos.centros_uniformes(t, M)        # onde fica cada monte e a largura
Phi_rbf = modelos.matriz_rbf(t, centros, sigma)          # o valor de cada monte em cada instante
w_rbf = modelos.resolver_equacao_normal(Phi_rbf, temp)   # os pesos, pelo MMQ (equação normal)
ajuste_rbf = Phi_rbf @ w_rbf                             # soma dos montes com os pesos = a linha
```

---

## 9. Etapa 02: quanto a linha erra (EMQ)

Agora que a linha existe, medimos o quanto ela erra. É a mesma conta da seção 6, com um passo a mais:

1. erro em cada instante: medido − calculado;
2. eleva ao quadrado;
3. soma tudo;
4. **divide pela quantidade de medidas**.

O resultado é o **EMQ, Erro Médio Quadrático**: a média dos erros ao quadrado. Dividir serve para o número não depender de quantas medidas existem.

Para a RBF, com as 1501 medidas: **EMQ = 3,63**.

Como os erros foram elevados ao quadrado, esse número está em "graus ao quadrado", o que não diz muita coisa. Tirando a raiz, volta para graus: √3,63 ≈ **1,9 grau**. Ou seja, a linha erra em média uns 2 graus.

Figura da Etapa 02: [fig03_rbf_emq.png](../resultados/fig03_rbf_emq.png). É a mesma linha da Etapa 01, com o EMQ escrito no canto, igual ao slide do professor.

**No código** ([metricas.py](../metricas.py), linhas 13 a 16): `erro = real - ajustado`, e depois `np.mean(erro**2)`, que eleva ao quadrado e tira a média.

---

## 10. Etapa 03: regressão linear e Pearson

### O gráfico medido × calculado

Figura: [fig04_rbf_regressao.png](../resultados/fig04_rbf_regressao.png)

Esse gráfico não tem tempo. Para cada instante, ele marca um ponto:

- posição para o lado: o **calculado**;
- posição para cima: o **medido**.

Se a linha acertasse tudo, medido seria igual a calculado em todos os instantes, e os pontos ficariam numa reta diagonal perfeita. Quanto mais perto da diagonal, melhor o ajuste.

### A reta (regressão linear)

A linha preta tracejada é a reta que passa mais perto de todos os pontos, escolhida também pelo MMQ. A fórmula dela deu:

```
medido = 1,000 × calculado + 0,0000
```

Isso é a diagonal perfeita: a RBF não erra puxando sempre para cima nem sempre para baixo.

### O Pearson (r)

O **r de Pearson** é um número entre 0 e 1 (pode ser negativo, mas aqui não importa) que diz **o quanto os pontos formam uma reta**:

- perto de 1: os pontos estão colados numa reta;
- perto de 0: os pontos formam uma nuvem sem formato.

Desenho com exemplos: [pearson_explicacao.png](pearson_explicacao.png)

Para a RBF: **r = 0,973**, bem perto de 1.

### A faixa rosa

A faixa rosa em volta da reta tem largura de 1,9 grau (a raiz do EMQ) para cima e para baixo. A maioria dos pontos fica dentro dela. O professor usa a mesma faixa no slide dele.

**No código:** a reta está em `regressao_linear` e o Pearson em `pearson`, os dois em [metricas.py](../metricas.py), escritos com as fórmulas da aula.

---

## 11. Etapa 04: o segundo jeito

### A ideia

A RBF desenha a linha usando só o tempo. Ela não sabe nada sobre a carga. O segundo jeito usa outra ideia:

> A temperatura de agora depende da temperatura de pouco antes e da carga.

Pensa numa panela no fogão. A água não muda de temperatura de repente: agora ela está quase igual a meio segundo atrás. Com o fogo ligado, vai esquentando. Com o fogo desligado, vai esfriando. No processador, o fogo é a carga.

O professor escreve isso assim:

```
y[n] = f( y[n−2], y[n−1], u[n], u[n−1] )
```

- `y` é a temperatura e `u` é a carga;
- `n` é "agora", `n−1` é meio segundo antes e `n−2` é um segundo antes.

### A fórmula com pesos

Cada pedaço ganha um peso:

```
temperatura agora = w0
                  + w1 × (temperatura de meio segundo antes)
                  + w2 × (temperatura de um segundo antes)
                  + w3 × (carga agora)
                  + w4 × (carga de meio segundo antes)
```

São só 5 pesos. O professor chama esses pesos de **parâmetros livres**, e o enunciado pede para identificá-los. Eles são escolhidos pelo **mesmo MMQ**, com a mesma equação normal da RBF.

Funciona como uma receita:

- os **pesos** são as quantidades, que o MMQ escolhe;
- as **temperaturas de antes e a carga** são os ingredientes, que vêm da tabela;
- o **calculado** é o resultado da receita em cada instante.

Os pesos encontrados:

| peso | valor |
|---|---|
| w0 | 6,51 |
| w1 | 0,927 |
| w2 | −0,039 |
| w3 | 10,38 |
| w4 | −8,92 |

### Exemplo em um instante: 305 segundos

Da tabela: temperatura meio segundo antes = 61, um segundo antes = 62, carga agora = 0, carga meio segundo antes = 0.

```
6,51
+ 0,927 × 61    = 56,55
+ (−0,039) × 62 = −2,42
+ 10,38 × 0     = 0
+ (−8,92) × 0   = 0
─────────────────────
= 60,64
```

O modelo calculou 60,64. O termômetro mediu 61.

### Treino e teste

Para achar os pesos, o código usa só os primeiros 70% do tempo (de 0 a 525 s). É o **treino**. Os últimos 30% (de 525 a 750 s) ficam guardados para o **teste**.

É como uma prova: se cair a mesma questão da lista, tirar nota alta não prova nada. O teste usa dados que o modelo nunca viu, e o EMQ no teste deu 4,16, parecido com o total. Então o modelo funciona em dados novos.

### As figuras da Etapa 04

- [fig05_mmq_ajuste.png](../resultados/fig05_mmq_ajuste.png): medidas e linha calculada pela fórmula. **EMQ = 5,38** (raiz ≈ 2,3 graus). A linha tem degraus porque cada conta parte da medida anterior, que já vem arredondada.
- [fig06_mmq_regressao.png](../resultados/fig06_mmq_regressao.png): a mesma análise da Etapa 03. A reta deu quase a diagonal (medido = 1,002 × calculado − 0,09), e **r = 0,960**.

**No código** ([main.py](../main.py), linhas 119 a 122): `matriz_regressores` monta as colunas (temperaturas de antes e carga), `resolver_equacao_normal` acha os 5 pesos (a mesma função da RBF) e `prever_um_passo` calcula a linha.

---

## 12. Comparação e conclusão

| | EMQ | raiz do EMQ | r |
|---|---|---|---|
| RBF (Etapa 01) | 3,63 | 1,9 grau | 0,973 |
| Fórmula de 5 pesos (Etapa 04) | 5,38 | 2,3 graus | 0,960 |

**A RBF errou menos.** Ela usa 301 pesos e desenha a linha em cima do que foi medido, então acompanha cada detalhe. Mas ela só usa o tempo: não sabe nada da carga e não diz o que acontece depois dos 750 segundos. Ela serve para **limpar o ruído** das medidas.

**A fórmula da Etapa 04 errou um pouco mais**, mas tem só 5 pesos e cada um tem significado: o w1 = 0,927 diz que a temperatura "lembra" muito do valor anterior, e o w3 = 10,38 mostra que ligar a carga empurra a temperatura para cima. Ela descreve **como o processador reage à carga** e funcionou em dados que não viu (o teste).

O ponto fraco da fórmula é o pico de 85 °C logo que a carga liga. O processador começa no máximo e logo se limita sozinho, e uma soma simples com pesos não consegue imitar isso direito.

Frase para o professor:

> "A RBF teve o menor erro, EMQ de 3,63 e r de 0,973, porque desenha a curva sobre os dados e filtra o ruído. O modelo do enunciado, com 5 parâmetros achados por mínimos quadrados, errou um pouco mais (EMQ de 5,38 e r de 0,960), mas explica como a temperatura responde à carga e funciona em dados que não foram usados para achar os pesos."

---

## 13. Palavras do professor e o que elas querem dizer

| o professor fala | quer dizer |
|---|---|
| ajuste de curva | achar a linha que passa pelo meio das medidas |
| valor ajustado ou estimado | o calculado pela linha |
| função gaussiana, função de base radial | o "monte" |
| pesos, parâmetros livres | as alturas dos montes, ou os 5 números da fórmula da Etapa 04 |
| MMQ, mínimos quadrados | escolher os pesos com a menor soma dos erros ao quadrado |
| equação normal | a fórmula que acha esses pesos de uma vez |
| EMQ | média dos erros ao quadrado |
| regressão linear | a reta que passa mais perto dos pontos medido × calculado |
| coeficiente de Pearson (r) | o quanto esses pontos formam uma reta, de 0 a 1 |
