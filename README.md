# Wave Order Picking (SBPO 2025)

Solver em Python para o Problema de Separação de Pedidos por Ondas do desafio Mercado Livre / SBPO 2025. O objetivo é maximizar as unidades coletadas divididas pelo número de corredores visitados, com a onda entre `LB` e `UB` e a demanda coberta pelo estoque dos corredores escolhidos.

O enunciado do trabalho pede o código-fonte, os arquivos de solução e a apresentação. O roteiro da defesa está em [`resumo_tecnico.md`](resumo_tecnico.md).

Repositório oficial do desafio: [mercadolibre/challenge-sbpo-2025](https://github.com/mercadolibre/challenge-sbpo-2025).

## Resultado

As 50 instâncias de `datasets/a`, `datasets/b` e `datasets/x` têm solução factível em `output/`, conferida com o `checker.py` oficial. A soma das funções objetivo é 17.667,84, frente a 17.781,59 da referência (99,36%). O gap médio é 1,63% e 25 instâncias estão no valor de `best_solutions/best_objectives.csv`. A tabela por instância está em `results_table.csv`.

## Método

Instâncias com até 200 pedidos vão para um MILP linearizado pelo algoritmo de Dinkelbach (OR-Tools, SCIP com fallback CBC). Nas demais, metade do tempo busca um conjunto pequeno de corredores e empacota os pedidos que cabem nele. O tempo restante roda uma construção gulosa com busca local. Fica a melhor razão factível.

## Dependências

Python 3, com `numpy`, `scipy` e `ortools`. O checker usa só `numpy`. O caminho exato precisa do OR-Tools.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy ortools
```

## Resolver uma instância

```bash
.venv/bin/python solve_one.py datasets/a/instance_0020.txt /tmp/sol_0020.txt --budget 30
.venv/bin/python checker.py datasets/a/instance_0020.txt /tmp/sol_0020.txt
```

`a/0020` é o exemplo do enunciado. O ótimo é 5 (pedidos 0, 1, 2 e 4, corredores 1 e 3).

`--method exact` ou `--method heuristic` força o caminho. Sem a opção, `O ≤ 200` usa o exato e o resto usa a heurística. `--seed` controla a parte aleatória da heurística. `--budget` é o tempo em segundos.

O arquivo de saída tem quatro linhas: quantidade de pedidos, índices dos pedidos, quantidade de corredores, índices dos corredores. Índices começam em 0.

## Lote e tabela

`run_all.py` grava em `output/<dataset>/` e acrescenta uma linha em `run_log.tsv` (esse log fica fora do Git). O orçamento vai de 30 s (`O ≤ 200`) a 80 s, conforme o número de pedidos.

```bash
.venv/bin/python run_all.py a --start 1 --end 20
.venv/bin/python run_all.py b --start 1 --end 15
.venv/bin/python run_all.py x --start 1 --end 15
.venv/bin/python build_report_table.py
```

`build_report_table.py` revalida `output/` com o checker e reescreve `results_table.csv`.

## Layout

| Caminho | Conteúdo |
|---|---|
| `wop.py` | Leitura da instância e escrita da solução |
| `exact.py` | Dinkelbach com MILP |
| `heuristic.py` | Busca por corredores e GRASP |
| `solve_one.py` | Uma instância |
| `run_all.py` | Um dataset |
| `checker.py` | Validador oficial |
| `build_report_table.py` | Gap contra a referência |
| `datasets/` | Instâncias A, B e X |
| `output/` | Soluções entregues |
| `best_solutions/best_objectives.csv` | Referência oficial |
| `resumo_tecnico.md` | Roteiro da apresentação |
