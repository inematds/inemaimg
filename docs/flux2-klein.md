# FLUX.2-klein-9B — Guia de uso e prompting

> Referência prática baseada na documentação oficial da
> [Black Forest Labs](https://docs.bfl.ml/guides/prompting_guide_flux2_klein),
> no [model card do HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
> e na [API do Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/flux2).
> Última atualização: 2026-04-15.

## Visão geral

| Detalhe | Valor |
|---|---|
| Parâmetros | 9B (transformer) + Qwen3 8B (text encoder) |
| Tipo | Rectified-flow transformer, step-distilled |
| Steps recomendado | **4** (sweet-spot da destilação) |
| Licença | FLUX Non-Commercial (uso comercial exige acordo com BFL) |
| VRAM | ~29 GB nativos; ~35 GB+ com text encoder |
| Resoluções | 128–2048 px em incrementos de **16 px** |
| Imagens de referência | até **4** via `image=` no pipeline |
| Text encoder | Qwen3 8B (`Qwen3ForCausalLM`) |
| Variantes | `-fp8`, `-kv`, `-kv-fp8`, `-nvfp4`, `-base-9B` (undistilled) |

## Prompting

### Regra de ouro

**Escreva prosa, não keywords.** O modelo foi treinado com captions
descritivas. A BFL diz textualmente:

> "Describe like a novelist, not a search engine."

### Estrutura recomendada

**Sujeito → Cenário → Detalhes → Iluminação → Atmosfera**

Exemplo:

> A ceramic coffee mug on a sunlit marble countertop, morning light
> streaming through sheer curtains, soft diffused shadows, warm color
> temperature, shallow depth of field with bokeh in the background

### Tamanho do prompt

| Tipo | Palavras | Quando usar |
|---|---|---|
| Curto | 10–30 | exploração rápida de conceitos |
| Médio | 30–80 | maioria dos casos de produção |
| Longo | 80–300 | editorial complexo, produto detalhado |

**Não exceda ~300 palavras** — prompts muito longos geram conflitos
internos entre as instruções.

### Iluminação — o fator de maior impacto

Seja específico sobre:
- **Tipo de fonte:** luz natural, ring light, neon, velas
- **Qualidade:** suave, dura, difusa, contrastada
- **Direção:** lateral, de cima, contraluz, 45 graus
- **Temperatura de cor:** quente (3000K), neutra (5500K), fria (7000K)
- **Interação com superfícies:** reflexos especulares, sombras projetadas

Exemplo bom: *"warm golden-hour sidelight from the left, long shadows
on concrete, highlights catching the chrome bumper"*

Exemplo ruim: *"good lighting"*

### O que NÃO funciona

| Prática | Por quê |
|---|---|
| `masterpiece, 8k, ultra detailed, best quality` | quality boosters são ignorados — o modelo não foi treinado com eles |
| Keywords empilhadas com vírgula estilo SDXL | degrada a aderência ao prompt; o encoder espera prosa |
| `guidance_scale > 1.0` | **ignorado silenciosamente** pelo checkpoint distilado — não faz nada |
| `negative_prompt` | FLUX.2 **não suporta** negative prompt na arquitetura do pipeline |
| Prompts > 300 palavras | gera conflito interno entre as instruções |
| Renderizar texto legível na imagem | limitação conhecida — texto sai distorcido frequentemente |
| `num_inference_steps > 4` | o checkpoint é **step-distilled pra 4** — aumentar steps não melhora e pode **piorar** (a imagem diverge, surgem artefatos). 4 não é um "mínimo", é o ponto ótimo. |

### Caption upsampling

O pipeline suporta `caption_upsample_temperature=0.15` que melhora
prompts curtos automaticamente via o text encoder. Nosso loader já
envia esse parâmetro por padrão. Prompts longos e detalhados não são
afetados.

## Parâmetros do pipeline

| Parâmetro | Valor recomendado | Notas |
|---|---|---|
| `num_inference_steps` | **4** | destilado pra 4; aceita 1–50 mas degrada fora de 4 |
| `guidance_scale` | **1.0** | ignorado internamente; manter em 1.0 pra evitar overhead |
| `caption_upsample_temperature` | **0.15** | melhora prompts curtos; inofensivo em prompts longos |
| `max_sequence_length` | 512 | limite de tokens do prompt |
| `text_encoder_out_layers` | `(9, 18, 27)` | default do klein (diferente do dev que usa `(10, 20, 30)`) |
| `torch_dtype` | `torch.bfloat16` | recomendado pela BFL |
| `width`, `height` | 128–2048, incrementos de 16 | baseline: 1024×1024 |

### Resoluções testadas

| Ratio | Resolução | Notas |
|---|---|---|
| 1:1 | 1024 × 1024 | baseline, melhor qualidade geral |
| 16:9 | 1536 × 864 | paisagem widescreen |
| 9:16 | 864 × 1536 | portrait / Stories |
| 4:3 | 1216 × 896 | semi-widescreen |
| 3:4 | 896 × 1216 | retrato clássico |

Resoluções muito fora do quadrado com poucos pixels totais tendem a
gerar artefatos de stretching.

## Imagens de referência (editing)

O pipeline aceita **até 4 imagens de referência** via `image=`:

```python
pipe(
    prompt="mesma cena, mas ao pôr do sol com céu alaranjado",
    image=[ref1, ref2],   # list[PIL.Image.Image]
    num_inference_steps=4,
)
```

O modelo unifica T2I, single-reference e multi-reference numa mesma
arquitetura — a presença ou ausência de `image=` determina o modo.

**Limitações do pipeline aberto:**
- Edição avançada (masking, role por imagem, inpainting por prompt) só
  está disponível na API hospedada da BFL, não no checkpoint aberto.
- O pipeline aberto faz reference conditioning — a imagem influencia a
  geração, mas você não controla *como* ela influencia (não dá pra dizer
  "use esta imagem como fundo e esta como personagem").

## Código de exemplo

### Text-to-image básico

```python
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="A Brazilian street market at dawn, colorful tarps over "
           "wooden stalls, warm golden light filtering through morning "
           "haze, vendors arranging tropical fruits, condensation on "
           "cold juice bottles, shallow depth of field",
    width=1024,
    height=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    caption_upsample_temperature=0.15,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
```

### Edição com referência

```python
from PIL import Image

ref = Image.open("produto.png")
image = pipe(
    prompt="o mesmo produto numa cozinha moderna, bancada de mármore, "
           "luz natural suave da janela, reflexos sutis no aço inox",
    image=ref,
    width=1024,
    height=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    caption_upsample_temperature=0.15,
).images[0]
```

### Via API do inemaimg

```bash
curl -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "flux2-klein",
    "prompt": "A ceramic coffee mug on a sunlit marble countertop...",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 42
  }'
```

## Fontes

- [Model card — HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [Prompting Guide — BFL docs](https://docs.bfl.ml/guides/prompting_guide_flux2_klein)
- [Diffusers API — Flux2KleinPipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux2)
- [Technical docs — Runware](https://runware.ai/docs/models/bfl-flux-2-klein-9b)
- [Tutorial — DataCamp](https://www.datacamp.com/tutorial/flux-2-klein-tutorial)
