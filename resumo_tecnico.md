# Wave Order Picking Problem (SBPO 2025 / Desafio Mercado Livre)
## Resumo técnico para defesa oral (10 minutos)

Repositório oficial: `mercadolibre/challenge-sbpo-2025`
Código, soluções e esta tabela/relatório foram gerados e validados com o `checker.py` oficial do repositório.

---

## 1. O problema e a função objetivo

Dado um conjunto de **O pedidos**, **I itens** e **A corredores**:

- cada pedido `o` exige `d[o][i]` unidades do item `i`;
- cada corredor `a` estoca `s[a][i]` unidades do item `i`;
- deve-se escolher um subconjunto de pedidos `X` e um subconjunto de corredores `Y` que **maximizem**

```
            Σ_{o ∈ X} unidades(o)
  f(X,Y) = ------------------------
                    |Y|
```

sujeito a:

1. **Capacidade da onda**: `LB ≤ Σ_{o∈X} unidades(o) ≤ UB` (limites lidos da instância);
2. **Cobertura por item**: para todo item `i`, `Σ_{o∈X} d[o][i] ≤ Σ_{a∈Y} s[a][i]` (o total pedido não pode exceder o total estocado nos corredores visitados, somado).

Este é um **programa fracionário 0‑1**: maximizar uma razão (unidades pickadas / corredores visitados) sujeita a restrições de cobertura tipo *set‑cover* e de capacidade tipo *knapsack*. É NP‑difícil (generaliza o problema da mochila e o *set cover* ponderado).

### Modelo matemático (MILP equivalente a cada iteração)

Variáveis: `x_o ∈ {0,1}` (pedido `o` selecionado), `y_a ∈ {0,1}` (corredor `a` visitado).

```
maximizar   Σ_o unidades(o)·x_o  −  q · Σ_a y_a        (objetivo linearizado, ver §3)

s.a.        LB ≤ Σ_o unidades(o)·x_o ≤ UB
            Σ_o d[o][i]·x_o ≤ Σ_a s[a][i]·y_a    ∀ item i
            x_o, y_a ∈ {0,1}
```

---

## 2. Estratégia escolhida e justificativa

A razão no objetivo (1) impede o uso direto de um solver MILP padrão (objetivo não‑linear). Usamos o **algoritmo de Dinkelbach** para programação fracionária 0‑1:

1. Dado um peso `q` (estimativa da razão ótima), resolve‑se o **problema linearizado**
   `max Σ unidades(o)·x_o − q·Σ y_a` sujeito às mesmas restrições.
2. Se a solução ótima `(X*,Y*)` desse problema linear satisfaz `unidades(X*) − q·|Y*| ≤ 0`, então `q` já é a razão ótima.
3. Caso contrário, atualiza‑se `q ← unidades(X*)/|Y*|` e repete‑se.

Esse método converge (em geral poucas iterações) **quando o passo 1 é resolvido de forma exata**. Dado o tamanho das instâncias (de 5 a mais de 68 mil pedidos, até ~54 mil itens), resolver o passo 1 de forma exata a cada iteração só é viável para instâncias pequenas. Adotamos uma estratégia **híbrida**:

- **PLI exato (OR‑Tools `linear_solver`, backend SCIP/CBC) + Dinkelbach** para as 7 instâncias com `O ≤ 200` pedidos (`a/0001`, `a/0002`, `a/0003`, `a/0004`, `a/0009`, `a/0012`, `a/0020`). Cada iteração resolve um MILP com *warm start*. O peso `q` só é atualizado quando o MILP termina com status ótimo; uma solução interrompida pelo tempo entra no melhor resultado, mas não vira o próximo `q`.
- **Busca por conjuntos pequenos de corredores, seguida de GRASP**, para as outras 43. Metade do orçamento escolhe poucos corredores e empacota os pedidos que cabem neles. A outra metade roda a construção gulosa com Dinkelbach heurístico. Fica a melhor razão factível das duas.

### Busca por corredores

As melhores soluções conhecidas visitam poucos corredores (em geral de 1 a 14, mesmo quando a instância oferece cerca de 480). Com o conjunto de corredores fixo, maximizar a razão é maximizar as unidades dentro de `[LB, UB]` sem estourar o estoque. A busca ranqueia corredores que cobrem muitos pedidos sozinhos ou que guardam estoque demandado, avalia subconjuntos pequenos e depois troca ou remove corredores que diluem a razão.

### Construção heurística ("density-first")

Esta etapa usa o tempo que sobra depois da busca por corredores.

A ideia central, inspirada na estrutura das soluções ótimas conhecidas (poucos corredores muito "densos" atendendo muitos pedidos pequenos), é:

1. **Preenchimento gratuito ("free-fit")**: dado o conjunto atual de corredores selecionados, adicionar — vetorizado com NumPy/SciPy sobre a matriz esparsa pedido×item — todo pedido que já é 100% coberto pela folga de estoque atual (não exige novo corredor). Isso nunca piora a razão (aumenta o numerador sem tocar o denominador).
2. **Expansão** marginal: quando não há mais pedidos "grátis", escolhe‑se, dentre uma amostra de candidatos, o pedido cujo ganho marginal `unidades(o) − q·(novos corredores necessários)` é maior (um *set cover* guloso ponderado decide quais corredores novos cobririam o pedido). Só se expande se o ganho for positivo — a menos que o volume mínimo da onda (`LB`) ainda não tenha sido atingido, caso em que a expansão é obrigatória (restrição rígida) mas ainda escolhida para minimizar o número de novos corredores.
3. Repete‑se (1)–(2) até atingir `UB`, esgotar os candidatos ou o tempo acabar.
4. **Busca local / GRASP**: movimentos aleatorizados de adição/remoção de pedidos avaliados pelo mesmo critério linearizado (`unidades − q·corredores`), mais uma varredura determinística (`trim_pass`) que tenta remover pedidos pequenos cuja saída libera corredores inteiros, e remoção de corredores redundantes após cada mudança.
5. O laço de Dinkelbach repete os passos 1–4 atualizando `q`, com reinícios (diversificação GRASP) quando `q` estabiliza.

### Orçamento de tempo

Cada instância recebe um orçamento de tempo proporcional ao número de pedidos (20 s a 80 s), bem abaixo do limite de 600 s do desafio oficial, para manter o tempo total de execução do lote gerenciável; em uma submissão real usaríamos o limite completo.

---

## 3. Explicação dos blocos essenciais do código

| Arquivo | Papel |
|---|---|
| `wop.py` | Classe `Instance`: lê o arquivo de entrada, monta matrizes esparsas `order_mat` (O×I) e `aisle_mat` (A×I) em CSR, pré‑computa `order_units`, `total_supply` e o mapa reverso `item_to_aisles`. Também escreve o arquivo de solução no formato oficial. |
| `exact.py` | `solve_inner_milp`: monta o MILP linearizado (variáveis `x_o`, `y_a`, restrições de onda e cobertura, objetivo `Σunid·x − q·Σy`) via OR‑Tools `pywraplp` (SCIP/CBC) e resolve com *hint* da iteração anterior. `solve_exact`: laço de Dinkelbach em torno dele. |
| `heuristic.py` | `State` mantém demanda e estoque de forma incremental. `solve_aisle_search` escolhe conjuntos pequenos de corredores e empacota os pedidos. `construct2`, `local_search` e `trim_pass` fazem a construção gulosa e o refinamento. `solve_heuristic` fica com a melhor das duas. |
| `solve_one.py` | CLI: lê uma instância, decide o método (`exact` se `O≤200`, senão `heuristic`), roda com o orçamento de tempo dado e grava a solução. |
| `run_all.py` | Roda `solve_one.py` em lote para um dataset (com *timeout* de segurança via `subprocess`), atribuindo o orçamento de tempo por tamanho da instância. |
| `build_report_table.py` | Valida **todas** as soluções geradas com a própria lógica do `checker.py` oficial (importado diretamente), calcula o gap percentual contra `best_solutions/best_objectives.csv` e gera `results_table.csv`. |

---

## 4. Resultados: função objetivo por instância vs. melhor conhecida

Todas as 50 instâncias oficiais (`datasets/a`, `datasets/b`, `datasets/x`) foram resolvidas e **100% das soluções passaram no `checker.py` oficial como viáveis** (nenhuma instância zerada por infactibilidade). A tabela completa está em `results_table.csv` / `table.md`; resumo por dataset:

| Dataset | Nº instâncias | Gap médio | Gap máximo | Nº ótimos (gap = 0) |
|---|---:|---:|---:|---:|
| a | 20 | 1.98% | 22.11% | 15 |
| b | 15 | 1.57% | 8.85% | 6 |
| x | 15 | 1.23% | 5.13% | 4 |
| **Geral** | **50** | **1.63%** | **22.11%** | **25** |

A soma das funções objetivo é 17.667,84, contra 17.781,59 da referência (99,36% do total). As 7 instâncias resolvidas por PLI exato estão todas em gap 0. As outras 18 com gap 0 foram obtidas pela busca de corredores. O maior gap restante é `a/0010` (22,11%). No dataset X, o maior é `x/0001` (5,13%).

Para contexto adicional, `results/final_phase.pdf` (resultados oficiais da fase final do desafio, cobrindo os datasets B e X) mostra que os valores em `best_objectives.csv` correspondem aos melhores objetivos obtidos pelas equipes competidoras ao longo de toda a competição — ou seja, o gap reportado aqui é contra o **estado da arte do desafio**, não contra um limite teórico fácil de alcançar.

---

## 5. Limitações da solução atual

1. **Construção gulosa por amostragem**: a escolha do "melhor pedido marginal" a cada expansão é feita sobre uma amostra aleatória (até 800 candidatos) e não sobre o conjunto completo de pedidos restantes, por razão de desempenho. Isso pode deixar passar combinações de pedidos/corredores melhores, especialmente em instâncias com muitos milhares de pedidos.
2. **O passo de "set cover" (`dry_run_new_aisles`) é guloso e local** — não garante o número mínimo de corredores novos para cobrir o déficit de um pedido; é NP‑difícil em geral e aqui é apenas aproximado.
3. **Dinkelbach heurístico não tem garantia de monotonicidade**: como o subproblema não é resolvido de forma exata para instâncias grandes, `q` pode oscilar (observado empiricamente) em vez de convergir suavemente; mitigamos mantendo sempre a melhor razão real já vista, mas isso não é uma prova de otimalidade nem de proximidade do ótimo.
4. **Orçamento de tempo reduzido**: usamos de 20 a 80 segundos por instância (o desafio oficial permitia 600 s). Com esse orçamento, `x/0014` ficou a 0,04% da referência. O que ainda pesa é um punhado de instâncias em que o conjunto de corredores candidatos não inclui a combinação da referência, em especial `a/0010`.
5. **Sem portfólio de métodos**: não há comparação interna entre, por exemplo, Simulated Annealing com *annealing schedule* explícito ou um PLI com *cutting planes* dedicados (desigualdades válidas de cobertura) — poderia melhorar a qualidade do MILP exato e estender o limiar de 200 pedidos.
6. **O limiar `O ≤ 200` para o método exato é fixo e arbitrário** — não há um critério adaptativo (ex.: baseado em `O × A` ou na densidade da matriz) que decida dinamicamente quando vale a pena tentar o MILP exato.

## 6. O que seria melhorado com mais tempo

- Usar o orçamento completo de 600 s por instância (dentro do limite oficial), e paralelizar execuções independentes entre instâncias.
- Resolver a relaxação linear do MILP a cada iteração de Dinkelbach para obter um **limite superior (upper bound)** e assim reportar um gap de otimalidade certificado (não apenas o gap contra o melhor conhecido pela competição).
- Substituir o *set cover* guloso por uma formulação de fluxo/cobertura mais forte, ou por um MILP auxiliar pequeno (apenas sobre os corredores candidatos a um item) para decidir corredores extras de forma mais precisa.
- Implementar *Simulated Annealing* propriamente dito (com temperatura decrescente e aceitação probabilística de pioras) no lugar do *hill-climbing* com reinícios GRASP atual, para escapar de ótimos locais em instâncias de médio porte.
- Paralelizar a avaliação de candidatos na fase de expansão (atualmente sequencial em Python puro) usando vetorização adicional ou multiprocessing, permitindo aumentar o tamanho da amostra sem aumentar o tempo de parede.
- Aumentar o limiar do método exato (ex.: via uma formulação MILP mais compacta, agregando itens raramente restritivos) para cobrir mais instâncias do dataset A com garantia de otimalidade.

## Apêndice: tabela completa (50 instâncias)

| Dataset | Instância | Viável | FO obtida | Melhor conhecida | Gap (%) |
|---|---|---|---:|---:|---:|
| a | 0001 | Sim | 15.000 | 15.000 | 0.00 |
| a | 0002 | Sim | 2.000 | 2.000 | 0.00 |
| a | 0003 | Sim | 12.000 | 12.000 | 0.00 |
| a | 0004 | Sim | 3.500 | 3.500 | 0.00 |
| a | 0005 | Sim | 177.875 | 177.875 | 0.00 |
| a | 0006 | Sim | 691.000 | 691.000 | 0.00 |
| a | 0007 | Sim | 392.250 | 392.250 | 0.00 |
| a | 0008 | Sim | 159.429 | 162.941 | 2.16 |
| a | 0009 | Sim | 4.417 | 4.417 | 0.00 |
| a | 0010 | Sim | 13.328 | 17.112 | 22.11 |
| a | 0011 | Sim | 15.273 | 16.850 | 9.36 |
| a | 0012 | Sim | 11.250 | 11.250 | 0.00 |
| a | 0013 | Sim | 110.786 | 117.385 | 5.62 |
| a | 0014 | Sim | 181.000 | 181.636 | 0.35 |
| a | 0015 | Sim | 149.333 | 149.333 | 0.00 |
| a | 0016 | Sim | 85.000 | 85.000 | 0.00 |
| a | 0017 | Sim | 36.500 | 36.500 | 0.00 |
| a | 0018 | Sim | 117.200 | 117.200 | 0.00 |
| a | 0019 | Sim | 202.000 | 202.000 | 0.00 |
| a | 0020 | Sim | 5.000 | 5.000 | 0.00 |
| b | 0001 | Sim | 631.500 | 631.500 | 0.00 |
| b | 0002 | Sim | 180.636 | 180.636 | 0.00 |
| b | 0003 | Sim | 149.600 | 149.600 | 0.00 |
| b | 0004 | Sim | 152.143 | 155.941 | 2.44 |
| b | 0005 | Sim | 368.000 | 368.000 | 0.00 |
| b | 0006 | Sim | 79.071 | 84.028 | 5.90 |
| b | 0007 | Sim | 108.000 | 108.000 | 0.00 |
| b | 0008 | Sim | 225.200 | 227.100 | 0.84 |
| b | 0009 | Sim | 80.357 | 82.500 | 2.60 |
| b | 0010 | Sim | 444.000 | 444.000 | 0.00 |
| b | 0011 | Sim | 896.000 | 898.800 | 0.31 |
| b | 0012 | Sim | 554.182 | 567.636 | 2.37 |
| b | 0013 | Sim | 756.444 | 757.000 | 0.07 |
| b | 0014 | Sim | 883.333 | 885.167 | 0.21 |
| b | 0015 | Sim | 151.438 | 166.143 | 8.85 |
| x | 0001 | Sim | 67.214 | 70.848 | 5.13 |
| x | 0002 | Sim | 217.667 | 217.667 | 0.00 |
| x | 0003 | Sim | 667.333 | 667.667 | 0.05 |
| x | 0004 | Sim | 200.500 | 202.750 | 1.11 |
| x | 0005 | Sim | 628.667 | 628.667 | 0.00 |
| x | 0006 | Sim | 191.571 | 191.857 | 0.15 |
| x | 0007 | Sim | 111.786 | 115.154 | 2.92 |
| x | 0008 | Sim | 418.100 | 438.222 | 4.59 |
| x | 0009 | Sim | 814.625 | 819.875 | 0.64 |
| x | 0010 | Sim | 962.375 | 964.625 | 0.23 |
| x | 0011 | Sim | 1060.000 | 1060.000 | 0.00 |
| x | 0012 | Sim | 1339.667 | 1339.667 | 0.00 |
| x | 0013 | Sim | 1055.286 | 1060.714 | 0.51 |
| x | 0014 | Sim | 1633.000 | 1633.667 | 0.04 |
| x | 0015 | Sim | 256.000 | 263.909 | 3.00 |