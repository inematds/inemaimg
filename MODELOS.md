# Modelos de Imagem — O Que É Cada Um e Comparação

Referência dos modelos avaliados para o projeto timesmkt3. Documento vivo — atualizar conforme novos modelos forem testados.

---

## Visão Geral Rápida

| Modelo | Tipo | Parâmetros | Tarefa | VRAM (prático) | Licença |
|---|---|---|---|---|---|
| ERNIE-Image | DiT | 8B | Text-to-image | 24 GB | Aberta |
| Qwen-Image-Edit-2511 | DiT | ~20B | Edição + multi-imagem | 24 GB (com offload) | Apache 2.0 |
| FLUX.2-klein-9B | Rectified Flow | 9B | T2I + edição, 4 steps | 24 GB (com offload) | FLUX Non-Commercial |
| FLUX.2-dev | Rectified Flow | 32B | T2I + edição premium | 24 GB (4-bit) | FLUX Non-Commercial |
| FLUX.2-small-decoder | VAE Decoder | ~28M | Acelera qualquer FLUX.2 | — | Apache 2.0 |

---

## 1. ERNIE-Image (baidu/ERNIE-Image)

**O que é:** Diffusion Transformer single-stream de 8B parâmetros, da Baidu. Focado em text-to-image puro.

**Tarefas suportadas:**
- Text-to-image

**NÃO suporta:**
- Image-to-image
- Referência de imagem
- ControlNet
- Edição

**Configuração recomendada:**
- dtype: `torch.bfloat16`
- Resoluções: 1024x1024, 848x1264, 768x1376, 896x1200 (e inversas)
- `guidance_scale`: 4.0
- `num_inference_steps`: 50 (padrão) ou 8 (variante Turbo)

**Código base:**
```python
from diffusers import ErnieImagePipeline
import torch

pipe = ErnieImagePipeline.from_pretrained(
    "baidu/ERNIE-Image", torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="...",
    height=1264, width=848,
    num_inference_steps=50,
    guidance_scale=4.0,
    use_pe=True,
).images[0]
```

**Quando usar:** geração criativa de imagens a partir de prompt puro, quando não há referência visual envolvida. Modelo compacto, roda confortavelmente em GPU consumer.

---

## 2. Qwen-Image-Edit-2511 (Qwen/Qwen-Image-Edit-2511)

**O que é:** Modelo de edição baseado em Qwen-Image (~20B parâmetros). Aceita 1 a 3 imagens de referência e prompt textual. Sucessor direto do 2509.

**Tarefas suportadas:**
- Image-to-image editing
- Multi-image editing (pessoa + cenário, pessoa + produto, etc.)
- ControlNet nativo (pose, depth, canny, sketch)
- Edição de texto na imagem (fonte, cor, material)
- Preservação de identidade
- Restauração de fotos antigas

**Configuração recomendada:**
- dtype: `torch.bfloat16`
- `true_cfg_scale`: 4.0
- `guidance_scale`: 1.0
- `num_inference_steps`: 40
- `negative_prompt`: `" "` (espaço simples)
- Offload para GPU de 24 GB: `pipe.enable_model_cpu_offload()`

**Código base:**
```python
from diffusers import QwenImageEditPlusPipeline
import torch
from PIL import Image

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16
).to("cuda")
pipe.enable_model_cpu_offload()

output = pipe(
    image=[Image.open("ref1.png"), Image.open("ref2.png")],
    prompt="...",
    negative_prompt=" ",
    true_cfg_scale=4.0,
    guidance_scale=1.0,
    num_inference_steps=40,
).images[0]
```

**LoRAs compatíveis relevantes:**
- **Multiple-Angles-LoRA** (AdversaLLC) — 96 ângulos de câmera para produtos. Apache 2.0.
- **BFS Best Face Swap** (Alissonerdx) — face/head swap com preservação de iluminação. MIT.

**Quando usar:** qualquer tarefa que envolva imagem de referência — inserir pessoa em cenário, mostrar produto em contexto, variações de ângulo, troca de rosto, edição instruída por texto.

---

## 3. FLUX.2-klein-9B (black-forest-labs/FLUX.2-klein-9B)

**O que é:** Rectified flow transformer de 9B parâmetros da Black Forest Labs. Step-distilled: gera imagens de alta qualidade em apenas **4 passos de inferência**. Arquitetura unificada (text-to-image + edição no mesmo modelo).

**Tarefas suportadas:**
- Text-to-image
- Image-to-image editing
- Single e multi-reference editing

**Configuração recomendada:**
- dtype: `torch.bfloat16`
- `guidance_scale`: 1.0 (fixo, guidance incorporado no treinamento)
- `num_inference_steps`: 4
- Resolução: 1024x1024
- Tempo por imagem: **sub-segundo em RTX 4090**

**Código base:**
```python
from diffusers import Flux2KleinPipeline
import torch

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="...",
    height=1024, width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
).images[0]
```

**Quando usar:** quando velocidade importa — previews em tempo real, muitas variações, feedback imediato ao usuário. Throughput ~37x maior que Qwen-Edit.

**Atenção:** licença **non-commercial**. Inviável para produto comercial sem acordo separado com BFL.

---

## 4. FLUX.2-dev (black-forest-labs/FLUX.2-dev)

**O que é:** Modelo premium da família FLUX.2, 32B parâmetros. Qualidade estado-da-arte absoluto. Suporta text-to-image, image-to-image e multi-referência sem fine-tuning.

**Tarefas suportadas:**
- Text-to-image premium
- Image-to-image
- Single e multi-reference editing

**Configuração recomendada:**
- Versão quantizada para consumer: `diffusers/FLUX.2-dev-bnb-4bit`
- dtype: `torch.bfloat16`
- `num_inference_steps`: 28–50
- `guidance_scale`: 4.0
- Text encoder remoto recomendado para caber em 24 GB

**Código base (4-bit):**
```python
from diffusers import Flux2Pipeline
import torch

pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    text_encoder=None,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="...",
    num_inference_steps=28,
    guidance_scale=4.0,
).images[0]
```

**Quando usar:** hero images, campanhas principais, artes finais onde qualidade supera tempo. Não usar para geração em massa.

**Atenção:** licença **non-commercial**, igual ao klein.

---

## 5. FLUX.2-small-decoder (black-forest-labs/FLUX.2-small-decoder)

**O que é:** Não é um modelo gerador. É um **VAE decoder distilado** (~28M params) que substitui o decoder padrão de qualquer FLUX.2. **Apache 2.0.**

**Ganhos:**
- ~1.4x mais rápido na decodificação
- ~1.4x menos VRAM no momento de decode
- Perda de qualidade imperceptível
- Compatível com FLUX.2-klein (4B e 9B) e FLUX.2-dev

**Uso:**
```python
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2
import torch

vae = AutoencoderKLFlux2.from_pretrained(
    "black-forest-labs/FLUX.2-small-decoder", torch_dtype=torch.bfloat16
)
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", vae=vae, torch_dtype=torch.bfloat16
)
```

**Quando usar:** sempre que usar qualquer FLUX.2. Zero downside.

---

## Comparações-Chave

### Carga de hardware (do mais leve ao mais pesado)

1. **FLUX.2-klein-9B** — 9B params, 4 steps, <1s/imagem, ~29 GB nativo
2. **ERNIE-Image** — 8B params, 50 steps, ~20s/imagem, 24 GB nativo
3. **FLUX.2-dev (4-bit)** — 32B params quantizado, 28-50 steps, ~15-30s/imagem
4. **Qwen-Image-Edit-2511** — ~20B params, 40 steps, ~20-40s/imagem, ~40 GB nativo

### Throughput por GPU (RTX 4090, estimativa)

| Modelo | Imagens/hora |
|---|---|
| FLUX.2-klein-9B | ~4.500 |
| ERNIE-Image | ~180 |
| FLUX.2-dev | ~120-240 |
| Qwen-Image-Edit-2511 | ~90-180 |

### Qwen-Image-Edit-2511 vs FLUX.2-klein-9B (head-to-head)

| Métrica | Qwen-Edit-2511 | FLUX.2-klein-9B |
|---|---|---|
| Parâmetros | ~20B | 9B |
| Steps | 40 | 4 |
| Velocidade | 20-40s | <1s |
| VRAM | 40 GB nativo | 29 GB nativo |
| Referência de imagem | Sim (até 3) | Sim |
| ControlNet | Nativo | Via extensão |
| Licença | Apache 2.0 | Non-commercial |
| Comercial OK | Sim | Não |

### FLUX.2-dev vs FLUX.2-klein-9B

| Métrica | FLUX.2-dev | FLUX.2-klein-9B |
|---|---|---|
| Parâmetros | 32B | 9B |
| Steps | 28-50 | 4 |
| Velocidade (4090) | 15-30s | <1s |
| `guidance_scale` | 4.0 ajustável | 1.0 fixo |
| Qualidade | Estado-da-arte | Próximo, ~5x eficiência |
| Uso ideal | Render final premium | Preview/volume |

---

## Matriz de Decisão

| Caso de uso | Modelo recomendado |
|---|---|
| Texto → imagem criativa, sem referência | ERNIE-Image ou FLUX.2-klein |
| Pessoa + cenário, preservando identidade | Qwen-Image-Edit-2511 |
| Produto em múltiplos ângulos | Qwen-Edit-2511 + Multiple-Angles-LoRA |
| Troca de rosto | Qwen-Edit-2511 + BFS LoRA |
| Preview rápido durante iteração | FLUX.2-klein-9B |
| Render final de campanha premium | FLUX.2-dev |
| Gerar 1000+ imagens/hora | FLUX.2-klein-9B |
| Produto comercial (SaaS pago) | Qwen-Edit-2511 + ERNIE (licenças abertas) |

---

## Notas de Licença

- **Apache 2.0 / Aberta** — ERNIE-Image, Qwen-Image-Edit-2511, FLUX.2-small-decoder → uso comercial livre.
- **FLUX Non-Commercial** — FLUX.2-dev, FLUX.2-klein-9B → apenas uso pessoal, pesquisa, ou acordo comercial com BFL.
- **MIT** — BFS LoRA → livre, mas cuidado com uso ético de face swap.

Para produto comercial, priorizar os modelos Apache/MIT. Usar FLUX.2 apenas em prototipagem interna ou com licenciamento explícito.
