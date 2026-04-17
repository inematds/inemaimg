# Comparativo de qualidade e steps dos modelos

> Guia prático sobre qualidade de imagem, quando usar cada modelo, e
> como os inference steps afetam o resultado. Baseado na documentação
> oficial de cada modelo e nos testes reais no GB10 (NGC 26.03, PyTorch 2.11).
> Última atualização: 2026-04-17.

## Ranking de qualidade

| # | Modelo | Qualidade | Velocidade no GB10 | Licença |
|---|---|---|---|---|
| 1 | **flux2-dev** | Topo absoluto — 32B params, o mais detalhado | ~2–3 min (28 steps, 1024²) | Non-commercial |
| 2 | **qwen-edit-2511** | Muito boa pra edição — preserva identidade, contexto | ~3–4 min (40 steps, 1024²) | Apache 2.0 |
| 3 | **flux2-klein** | Surpreendente pro custo — 4 steps com qualidade ~85% do dev | segundos–1 min | Non-commercial |
| 4 | **ernie** | Competente mas inferior — 8B params, menos detalhe | ~2–3 min (50 steps, 1024²) | Aberta |

## Onde cada um brilha

### flux2-dev — qualidade máxima

- Melhor renderização de materiais (metal, vidro, tecido)
- Composição mais cinematográfica
- Detalhes finos (texturas de pele, reflexos, micro-detalhes)
- **Quando usar:** hero images, peças finais de campanha, quando qualidade > tudo

### qwen-edit-2511 — o editor

- **Único que realmente edita** com imagens de referência (1–3)
- Preserva identidade (rosto, produto, marca)
- Melhor pra "coloque este produto neste cenário"
- **Quando usar:** product placement, variações de ângulo, face swap, qualquer tarefa com imagem de referência
- **Único com licença comercial pra uso sério** (Apache 2.0)

### flux2-klein — velocidade com qualidade

- 4 steps com qualidade ~85% do dev
- Perfeito pra iterar prompts rápido
- Aceita referência mas com menos controle que o qwen
- Iluminação é o ponto forte (responde muito bem a descrições de luz)
- **Quando usar:** iteração de prompt, previews, volumes grandes

### ernie — o fallback simples

- T2I puro, não aceita referência nenhuma
- Menor dos 4 (8B), menos detalhe fino
- Útil como fallback ou pra volumes grandes sem exigência de qualidade
- **Quando usar:** geração em massa sem referência, licença totalmente aberta

## Como os steps afetam a qualidade

Mais steps = o sampler converge mais pra imagem "ideal" do modelo. Mas
todo modelo tem um **ponto de saturação** — depois dele você gasta tempo
sem ganho visível. É como assar um bolo: 30 min assa, 40 min assa
melhor, 2 horas queima.

### flux2-klein — NÃO aumente além de 4

O checkpoint é **step-distilled pra exatamente 4 steps**. Não é um
"mínimo", é o ponto ótimo. Aumentar **piora** a qualidade.

| Steps | Resultado |
|---|---|
| **4** | **Sweet-spot** — é pra isso que ele foi treinado |
| 8+ | Imagem diverge, artefatos, qualidade **cai** |

### qwen-edit-2511 — ganho real até ~40, depois platô

| Steps | Resultado | Tempo (GB10, 1024²) |
|---|---|---|
| 10–15 | Rascunho — composição ok, detalhes borrados | ~1 min |
| 20–25 | Bom — maioria dos detalhes resolve | ~2 min |
| **40** | **Recomendado** — convergência completa (model card) | ~3–4 min |
| 50+ | Marginal — 25% mais lento, ganho quase imperceptível | ~5 min |

### ernie — platô em ~50

| Steps | Resultado | Tempo (GB10, 1024²) |
|---|---|---|
| 20 | Rascunho utilizável | ~1 min |
| 35 | Bom custo/benefício | ~1.5 min |
| **50** | **Recomendado** (model card) | ~2–3 min |
| 80+ | Desperdiça tempo | ~4+ min |

### flux2-dev — ganho real até ~28

| Steps | Resultado | Tempo (GB10, 1024²) |
|---|---|---|
| 10 | Preview rápido, aceitável | ~1 min |
| 20 | Bom | ~1.5 min |
| **28** | **Recomendado** pela BFL | ~2–3 min |
| 50 | Marginal, só se quiser espremer o último % | ~5 min |

## Fluxo recomendado pro timesmkt3

```
1. Iteração de prompt  →  flux2-klein (fast, 512², 4 steps)   →  segundos
2. Validação visual    →  flux2-klein (high, 1024², 4 steps)  →  ~1 min
3. Render final        →  flux2-dev (1024², 28 steps)         →  ~3 min
4. Edição com produto  →  qwen-edit-2511 + imagem ref         →  ~4 min
5. Produção comercial  →  qwen-edit-2511 ou ernie             →  licença OK
```

## Resumo visual das diferenças

| Aspecto | flux2-dev | qwen-edit | flux2-klein | ernie |
|---|---|---|---|---|
| Detalhe fino | excelente | bom (foco em fidelidade) | bom | mediano |
| Composição | cinematográfica | guiada pela referência | boa | genérica |
| Materiais | fotorrealista | bom | bom | ok |
| Texto na imagem | ruim | ruim | ruim | ruim |
| Coerência de cena | alta | muito alta (com ref) | alta | média |
| Imagem de referência | sim (até 4) | sim (até 3, melhor controle) | sim (até 4) | não |
| Velocidade | lento | lento | **rápido** | médio |
| Licença comercial | não | **sim** | não | **sim** |
