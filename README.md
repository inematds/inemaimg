# inemaimg

> Servidor local multi-modelo de **geração e edição de imagens** com
> hot-swap entre Qwen-Edit, FLUX.2 e ERNIE, exposto como API HTTP e
> playground web. Roda em Docker no DGX Spark (GB10), consumido pelo
> [timesmkt3](https://github.com/inematds/timesmkt3) como drop-in de
> provider de imagens.

![status](https://img.shields.io/badge/status-MVP-ff9f43) ![licença](https://img.shields.io/badge/licen%C3%A7a-projeto_interno-lightgrey) ![pytorch](https://img.shields.io/badge/pytorch-2.11%20(NGC%2026.03)-ee4c2c) ![gpu](https://img.shields.io/badge/gpu-Blackwell%20GB10-76b900)

## Sobre o projeto

O **inemaimg** nasceu de uma necessidade concreta do timesmkt3: tínhamos
que gerar centenas de imagens por semana pra campanhas de marketing,
pagando por chamadas em APIs como Kie.ai, Pollinations e OpenAI —
cada uma com licenças, preços e limites de rate diferentes. Com a
chegada do **DGX Spark (GB10 Grace-Blackwell Superchip)** com 119 GB
de memória unificada, ficou inviável *não* rodar tudo local.

O servidor resolve três problemas de uma vez:

1. **Multi-modelo sob demanda** — uma mesma API expõe 4 modelos de
   imagem com perfis complementares (velocidade, qualidade, licença
   comercial) e troca entre eles em memória conforme a requisição.
2. **Licenciamento claro** — Qwen-Edit-2511 (Apache 2.0) e ERNIE
   (aberta) pra uso comercial; FLUX.2-klein e FLUX.2-dev (non-commercial)
   pra iteração rápida e render premium durante o dev.
3. **Drop-in pra o timesmkt3** — um `generateImage(outputPath, prompt,
   model, aspect)` que substitui qualquer outro provider sem mexer
   nos call-sites.

**Stack:** Python 3.12 + FastAPI + diffusers (git main) + PyTorch 2.11
(NGC `26.03-py3` arm64-sbsa) + Docker Compose.

**Arquitetura resumida:**

```
   timesmkt3 / curl / browser
            │
            ▼
  ┌───────────────────┐
  │  FastAPI gateway  │  → /health /models /generate
  │  asyncio.Lock GPU │
  └─────────┬─────────┘
            │  (hot swap quando o modelo pedido != carregado)
            ▼
  ┌───────────────────┐
  │  Model Registry   │
  │  qwen-edit-2511   │  ← Apache 2.0, motor principal (prewarm)
  │  flux2-klein      │  ← non-commercial, 4 steps, o mais rápido
  │  ernie            │  ← aberta, T2I puro
  │  flux2-dev        │  ← non-commercial, 4-bit, premium
  └─────────┬─────────┘
            ▼
  ┌───────────────────┐
  │  NVIDIA GB10      │  119 GB memória unificada, sm_120
  │  1 modelo quente  │  (registry é trivialmente extensível
  │                   │   pra multi-hot no futuro)
  └───────────────────┘
```

Mais contexto:
- Visão, decisões de arquitetura, roadmap → [`PROJETO.md`](./PROJETO.md)
- Comparação técnica dos modelos, VRAM, licenças → [`MODELOS.md`](./MODELOS.md)
- Histórico de integração com o timesmkt3 → [`timesmkt3.md`](./timesmkt3.md)
- Guia de prompting + parâmetros do FLUX.2-klein → [`docs/flux2-klein.md`](./docs/flux2-klein.md)
- Comparativo de qualidade, steps e fluxo recomendado → [`docs/comparativo-modelos.md`](./docs/comparativo-modelos.md)

## Instalação

### 1. Requisitos de hardware e sistema

| Item | Alvo deste projeto | Mínimo pra rodar |
|---|---|---|
| GPU | NVIDIA GB10 (Blackwell, sm_120) | qualquer GPU com ≥ 24 GB VRAM |
| Arquitetura | arm64-sbsa (NGC 26.03 tem build oficial) | x86_64 também funciona, mas precisa swapar a base image pro build amd64 correspondente |
| RAM | 119 GB unificada (GB10) | 32 GB + 24 GB VRAM discreta |
| Disco | ~100 GB livres em `~/.cache/huggingface` | ~50 GB mínimo (só Qwen-Edit + FLUX.2-klein) |
| Driver NVIDIA | 580.x+ (CUDA 13) | 550.x+ (CUDA 12.4) |
| Docker | Compose v2 + NVIDIA Container Toolkit ou CDI | idem |
| SO | Linux (testado em Ubuntu 24.04 kernel 6.14 NVIDIA) | qualquer Linux com suporte ao nvidia runtime |

> **Nota sobre Blackwell / sm_120:** o PyTorch 2.7 do NGC `25.03-py3`
> tinha kernels ausentes pra várias operações em sm_120 e caía em
> fallbacks lentos (uma geração de 4 steps demorava ~290s). O bump
> pro NGC `26.03-py3` (PyTorch 2.11) resolveu isso — mesma geração
> agora roda em ~31s. Se você for reimplantar noutra base image,
> valide com um smoke test antes e leia `loaders/_blackwell_shims.py`
> pra entender o workaround do `nvrtc` que ainda está ativo.

### 2. Pré-requisitos no host

```bash
# Verificar GPU + driver
nvidia-smi

# Verificar Docker com acesso a GPU (NVIDIA Container Toolkit)
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.03-py3 nvidia-smi
# Se esse comando retornar um nvidia-smi válido, a stack do host está OK.

# Verificar espaço livre pro cache de modelos
df -h ~/.cache/huggingface || df -h $HOME
```

Se o segundo comando falhar com erro de runtime, instale o
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
antes de continuar.

### 3. Clonar e entrar no diretório

```bash
git clone git@github.com:inematds/inemaimg.git
cd inemaimg
```

Estrutura que você vai ver:

```
inemaimg/
├── server.py                  # FastAPI + registry + asyncio lock
├── loaders/
│   ├── _blackwell_shims.py    # workaround de nvrtc pra sm_120
│   ├── qwen_edit.py           # motor principal (Apache 2.0)
│   ├── ernie.py               # text-to-image aberto
│   ├── flux2_klein.py         # T2I + edit, 4 steps (non-commercial)
│   └── flux2_dev.py           # T2I premium 4-bit (non-commercial)
├── web/index.html             # playground servido em /
├── Dockerfile                 # base NGC 26.03-py3 arm64-sbsa
├── docker-compose.yml
├── requirements.txt
├── PROJETO.md  MODELOS.md  timesmkt3.md  README.md
```

### 4. Autenticar no HuggingFace (obrigatório pros modelos gated)

FLUX.2-klein e FLUX.2-dev são gated — exigem aprovação na página do
modelo **e** um token HF exportado pro container. Ver a seção
[Acesso a modelos gated (FLUX.2)](#acesso-a-modelos-gated-flux2) abaixo
pro passo-a-passo completo. Se você só vai usar `qwen-edit-2511` ou
`ernie`, **pule essa etapa**: eles são públicos.

Jeito rápido, assumindo que você já aprovou os gates e já rodou
`huggingface-cli login` no host:

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

O `docker-compose.yml` já faz `HF_TOKEN=${HF_TOKEN:-}`, então qualquer
valor exportado no shell é passado pro container. Se a variável não
estiver setada, só os modelos públicos carregam.

### 5. Primeiro boot (build + download dos pesos)

```bash
docker compose up -d --build
docker compose logs -f
```

O que acontece internamente:

1. **Build da imagem** (~10 min na primeira vez, 2–3 min nas seguintes).
   A base NGC `26.03-py3` arm64-sbsa tem ~20 GB — o pull ocupa mais
   tempo que o build em si.
2. **Prewarm assíncrono do Qwen-Edit-2511** (`INEMAIMG_PREWARM=qwen-edit-2511`
   no compose). Na primeira subida, baixa **~40 GB de pesos** via
   hf_transfer. Em uma conexão típica residencial isso leva **~10–20 min**;
   em fibra empresarial, ~5 min. Os modelos ficam cacheados em
   `~/.cache/huggingface` (volume montado), então boots seguintes não
   re-baixam nada — só carregam pra VRAM em ~4 min.
3. **`/health` responde imediatamente** mesmo durante o download — o
   prewarm roda em task separada. Você pode pollar o campo
   `prewarm_status` até ele virar `ready`:

```bash
# Em outro terminal enquanto o boot acontece:
watch -n 2 'curl -s localhost:8000/health | python3 -m json.tool'
```

Estados possíveis do `prewarm_status`:
- `running` → download ou load em andamento, esperar
- `ready` → primeira geração vai rodar na hora
- `failed` → olhar `prewarm_error` no JSON e `docker compose logs inemaimg`

### 6. Verificar que está tudo funcionando

```bash
# 1. Health check
curl -s localhost:8000/health | python3 -m json.tool
# Deve retornar prewarm_status: ready e gpu_memory_allocated_gb > 0

# 2. Listar modelos
curl -s localhost:8000/models | python3 -m json.tool

# 3. Playground no browser
#    Abra: http://localhost:8000/  (ou o IP do host na LAN)
```

Na UI você vai ver:
- Badge verde "pronto — qwen-edit-2511 · 53.79 GB" no canto superior direito
- Select com os 4 modelos disponíveis
- Controles de prompt, tamanho, steps, cfg, seed
- Histórico persistente das suas gerações

### 7. Smoke test real (opcional mas recomendado)

Dispara uma geração simples de 4 steps no Qwen-Edit pra confirmar que
a GPU tá processando de verdade:

```bash
python3 - <<'PY'
import base64, io, json, urllib.request
from PIL import Image, ImageDraw

img = Image.new("RGB", (384, 384), (40, 60, 130))
ImageDraw.Draw(img).rectangle([96, 96, 288, 288], fill=(220, 180, 40))
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

req = urllib.request.Request(
    "http://localhost:8000/generate",
    data=json.dumps({
        "model": "qwen-edit-2511",
        "prompt": "change the yellow square into a red circle",
        "images": [b64],
        "steps": 4,
        "seed": 42,
    }).encode(),
    headers={"content-type": "application/json"},
)
with urllib.request.urlopen(req, timeout=900) as r:
    j = json.loads(r.read())
    print(f"OK: {j['generation_time_s']}s on {j['model_used']}, vram {j['gpu_memory_allocated_gb']} GB")
    with open("/tmp/inemaimg_smoke.png", "wb") as f:
        f.write(base64.b64decode(j["image"]))
print("saved /tmp/inemaimg_smoke.png")
PY
```

Tempos esperados no GB10:
- **4 steps em 384²**: ~30s (primeira vez, ~31s; subsequentes, ~25s)
- **40 steps em 1024²** (recomendado do Qwen): ~3–4 min
- **4 steps em 512²** no FLUX.2-klein: alguns segundos (depois que
  terminar de baixar os ~35 GB do klein)

### 8. Parar, reiniciar, troubleshoot

```bash
# Parar
docker compose down

# Reiniciar preservando o cache HF
docker compose restart

# Rebuild da imagem (necessário só quando mexe em server.py, loaders/,
# Dockerfile ou requirements.txt — arquivos em web/ são volume mount
# e atualizam na hora)
docker compose up -d --build

# Logs ao vivo
docker compose logs -f

# Entrar no container pra debugar
docker compose exec inemaimg bash
```

Causas comuns de falha no boot:

| Sintoma | Causa provável | Fix |
|---|---|---|
| `prewarm_status: failed` | `INEMAIMG_PREWARM` aponta pra um id fora do registry | conferir `server.py:REGISTRY` e corrigir o compose |
| `403 Forbidden` no download | modelo gated sem `HF_TOKEN` | ver seção [Acesso a modelos gated](#acesso-a-modelos-gated-flux2) |
| `CUDA out of memory` | outro processo segurou a GPU | `nvidia-smi` → matar processo; ou ajustar `USE_CPU_OFFLOAD=True` no loader |
| `nvrtc: invalid value for --gpu-architecture` | base image sem kernels pra sm_120 | **não deveria acontecer no 26.03**; se acontecer, o shim em `loaders/_blackwell_shims.py` já cobre o caso conhecido — abrir issue descrevendo a op que falhou |
| `/health` trava durante geração | versão antiga rodando `pipe()` no event loop | o fix já está no `main` (executor), puxar `git pull && docker compose up -d --build` |

## Subir o serviço (quick reference)

```bash
# 1. (primeira vez) exportar token do HF pra repos gated
export HF_TOKEN=$(cat ~/.cache/huggingface/token)  # ou hf_xxx direto

# 2. subir
docker compose up -d --build

# 3. acompanhar logs enquanto o prewarm acontece
docker compose logs -f
```

O primeiro boot faz download dos pesos do modelo pré-aquecido (`qwen-edit-2511`,
~40 GB). Boots seguintes reusam o cache em `~/.cache/huggingface`.

- **UI de teste:** http://localhost:8000/
- **API:** http://localhost:8000/health, `/models`, `/generate`

## Endpoints

### `GET /health`
```json
{
  "status": "ok",
  "loaded_model": "qwen-edit-2511",
  "prewarm_status": "ready",
  "prewarm_error": null,
  "gpu_memory_allocated_gb": 53.79,
  "cuda_available": true
}
```
`prewarm_status` é um de `idle | running | ready | failed`. O prewarm roda
assíncrono — o `/health` responde imediatamente no boot e você pode pollar
essa chave até virar `ready`.

### `GET /models`
Lista modelos disponíveis no registry e qual está atualmente carregado.

### `POST /models/load`
Pré-carrega um modelo específico. Útil pra aquecer antes de uma rajada.
```json
{ "model": "flux2-klein" }
```

### `POST /generate`
Gera uma imagem. Troca o modelo em memória se necessário (custa o swap).
```json
{
  "model": "qwen-edit-2511",
  "prompt": "coloque o produto na cozinha, luz quente",
  "images": ["<base64 PNG/JPEG>", "..."],
  "steps": 40,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```
Retorna:
```json
{
  "image": "<base64 PNG>",
  "model_used": "qwen-edit-2511",
  "generation_time_s": 205.4,
  "gpu_memory_allocated_gb": 53.81
}
```

## Modelos registrados

| ID | Tipo | Licença | Notas |
|---|---|---|---|
| `qwen-edit-2511` | Edição + multi-imagem | Apache 2.0 | Motor principal. Exige 1–3 imagens de referência. |
| `ernie` | Text-to-image puro | Aberta | Rejeita `images`. |
| `flux2-klein` | T2I + edição, 4 steps | FLUX **Non-Commercial** | Até 4 ref images. `guidance_scale` ignorado (fixo 1.0). Sem `negative_prompt`. [Guia completo](./docs/flux2-klein.md). |
| `flux2-dev` | T2I premium, 4-bit | FLUX **Non-Commercial** | Usa `diffusers/FLUX.2-dev-bnb-4bit`. Gated. |

> **Aviso de licença:** `flux2-klein` e `flux2-dev` são FLUX Non-Commercial.
> Se o inemaimg for embutido num produto comercial sem acordo com a BFL,
> remova-os do `REGISTRY` em `server.py` antes do deploy.

## Qualidade e inference steps

Cada modelo tem um **ponto ótimo de steps** — além dele, você gasta
tempo sem ganho visível. Detalhes completos em
[`docs/comparativo-modelos.md`](./docs/comparativo-modelos.md).

| Modelo | Steps recomendado | Mais steps ajuda? | Qualidade relativa |
|---|---:|---|---|
| `flux2-dev` | **28** | Sim, até ~28 — marginal depois | Topo absoluto (32B) |
| `qwen-edit-2511` | **40** | Sim, até ~40 — platô depois | Muito boa (foco em edição/fidelidade) |
| `flux2-klein` | **4** | **NÃO** — piora acima de 4 (step-distilled) | Boa (~85% do dev) |
| `ernie` | **50** | Sim, até ~50 — platô depois | Competente (8B) |

**Fluxo recomendado:**

```
iteração de prompt  →  flux2-klein (512², 4 steps)         →  segundos
validação visual    →  flux2-klein (1024², 4 steps)        →  ~1 min
render final        →  flux2-dev (1024², 28 steps)         →  ~3 min
edição com produto  →  qwen-edit-2511 + ref images         →  ~4 min
produção comercial  →  qwen-edit-2511 ou ernie (licença OK)
```

## Consumindo de um cliente Node (timesmkt3)

O inemaimg foi desenhado pra ser consumido como uma API HTTP qualquer. No
timesmkt3 (`~/projetos/timesmkt3`) ele já está wireado como um "provider"
no mesmo padrão dos outros (kie, pollinations, piramyd), com drop-in zero
nos call-sites da pipeline.

### Arquivos tocados no timesmkt3

- `pipeline/generate-image-inemaimg.js` — provider novo. Exporta
  `generateImage(outputPath, prompt, model, aspectRatio)`, `AVAILABLE_MODELS`,
  `DEFAULT_MODEL`, `BASE_URL` e `getHealth`. Modo CLI incluso.
- `pipeline/worker.js` — adicionados o `require('./generate-image-inemaimg')`
  e a branch `if (p === 'inemaimg') return inemaimgProvider;` dentro de
  `getImageProvider()`.

Qualquer módulo da pipeline que use o dispatcher
(`worker.js`, `worker-video.js`, `worker-ad-creative.js`, `worker-video-pro.js`)
passa a falar com o inemaimg automaticamente quando a env abaixo estiver
setada — zero mudança nos call-sites.

### Variáveis de ambiente (no `.env` do timesmkt3)

```bash
IMAGE_PROVIDER=inemaimg                   # ativa o dispatcher
INEMAIMG_URL=http://localhost:8000        # ou http://<ip-do-servidor>:8000
INEMAIMG_MODEL=flux2-klein                # default; também: qwen-edit-2511, ernie, flux2-dev
INEMAIMG_QUALITY=fast                     # fast (padrão) | high
# INEMAIMG_TIMEOUT_S=1800                 # opcional; 30 min cobre cold starts
```

### Perfis `fast` vs `high`

| Modelo | `fast` | `high` |
|---|---|---|
| `flux2-klein` | 4 steps, 512² | 4 steps, 1024² |
| `qwen-edit-2511` | 15 steps, 512² | 40 steps, 1024² |
| `ernie` | 20 steps, 512² | 50 steps, 1024² |
| `flux2-dev` | 10 steps, 512² | 28 steps, 1024² |

`fast` é pra iterar rápido durante dev; `high` é pra renders finais. Trocar
o perfil é só mudar a env — não requer rebuild nem restart do timesmkt3.

### Smoke test direto pela CLI (antes de rodar a pipeline)

```bash
cd ~/projetos/timesmkt3
node pipeline/generate-image-inemaimg.js /tmp/teste.png "prompt aqui" flux2-klein 1:1
```

O CLI bate primeiro no `/health` pra te dar um status (modelo carregado,
`prewarm_status`, VRAM) e só então dispara `/generate`. Se o `flux2-klein`
ainda não terminou de baixar, passa `qwen-edit-2511` no lugar — ele é o
modelo pré-aquecido pelo `INEMAIMG_PREWARM` do `docker-compose.yml`.

### Modelos gated

`flux2-klein` e `flux2-dev` são gated no HF. O token tem que estar
disponível **no container do inemaimg** (não no timesmkt3 — o timesmkt3 só
fala HTTP com o inemaimg, não baixa pesos). Ver a seção abaixo sobre
`HF_TOKEN`.

### Integração programática (fora do CLI)

```js
const inemaimg = require('./pipeline/generate-image-inemaimg');

// Health check antes de um batch pesado
const health = await inemaimg.getHealth();
if (health.prewarm_status !== 'ready') {
  // modelo ainda carregando — esperar ou cair num fallback
}

// Uma geração
await inemaimg.generateImage(
  '/path/out.png',
  'a cyberpunk cat in a neon alley',
  'flux2-klein',
  '16:9',
);
```

## Acesso a modelos gated (FLUX.2)

Os modelos FLUX.2 exigem aprovação no HuggingFace antes de baixar:

1. Logue em https://huggingface.co/ e visite:
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
   - https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit
   - (ERNIE e Qwen-Edit são abertos, não precisam de aprovação)

2. Clique em **"Request access"** em cada página. Aprovação para uso não
   comercial costuma ser instantânea.

3. Gere um token *read-only* em
   https://huggingface.co/settings/tokens e exporte antes de subir:
   ```bash
   export HF_TOKEN=hf_xxx
   docker compose up -d --build
   ```
   O `docker-compose.yml` já repassa `HF_TOKEN` pro container. Se a variável
   não estiver no ambiente, só os modelos públicos carregam — os gated
   retornam HTTP 403 com instrução no `detail`.

## Estrutura do código

```
inemaimg/
├── server.py              # FastAPI + registry + lock serializado
├── loaders/
│   ├── __init__.py
│   ├── _blackwell_shims.py  # workaround sm_120 (ver abaixo)
│   ├── qwen_edit.py
│   ├── ernie.py
│   ├── flux2_klein.py
│   └── flux2_dev.py
├── web/index.html         # playground (vanilla JS, servido em /)
├── docker-compose.yml
├── Dockerfile             # base nvcr.io/nvidia/pytorch:26.03-py3 arm64-sbsa
└── requirements.txt
```

Cada loader expõe `load()` e `generate(pipe, req, images)`. Pra adicionar um
modelo novo, solte um arquivo em `loaders/` e registre a classe em
`REGISTRY` no `server.py`.

## Concorrência

Uma única `asyncio.Lock` serializa todo acesso à GPU. Isso garante que a
gente nunca roda duas gerações ao mesmo tempo (OOM seguro) e serializa
trocas de modelo. `pipe()` e `_load()` rodam em `run_in_executor` pra não
travar o event loop — `/health` continua respondendo durante uma geração
longa.

## Quirks da plataforma (Blackwell / GB10 / sm_120)

### Shim de `nvrtc` (`loaders/_blackwell_shims.py`)

Alguns kernels do PyTorch são JIT-compilados via `nvrtc` no primeiro uso.
No NGC 25.03 isso explodia com `"nvrtc: error: invalid value for
--gpu-architecture (-arch)"` pra `Tensor.prod()` e `Tensor.cumprod()` em
tensores CUDA específicos — a thrash clássica é o `image_grid_thw.prod(-1)`
dentro do encoder Qwen2.5-VL.

O shim em `_blackwell_shims.py` faz um monkey-patch minúsculo que roteia
prod/cumprod de tensores **int-CUDA** e **float-CUDA pequenos** (≤ 1024
elementos) via CPU. Grandes reduções float continuam na GPU.

Remover este arquivo quando a base image tiver kernels nativos pra todos
os caminhos (teste: tirar, rodar uma geração com imagem de referência; se
funcionar e não degradar perf, foi). Dependência já mínima — foi movida
prum arquivo separado justamente pra remover fácil.

### Base image bumpada

`Dockerfile` usa `nvcr.io/nvidia/pytorch:26.03-py3`. O `25.03` rodava mas
uma geração Qwen-Edit de 4 steps demorava **~290s steady-state** porque
vários paths de atenção caíam em implementações de referência. No `26.03`
(PyTorch 2.11) a mesma geração cai pra ~31s. Sem offload CPU (GB10 tem
memória unificada — offload é contraproducente aqui).

## Débito técnico conhecido

- **Porta exposta em `0.0.0.0` sem auth.** Temporário, pra permitir testes
  a partir de outras máquinas da LAN. Antes de produção: reverter pra
  `127.0.0.1:8000:8000` ou adicionar middleware de API key.
- **Perf ainda abaixo do ideal teórico.** 4 steps em ~31s é ~9x melhor que
  antes, mas o hardware provavelmente aguenta ~8s. Investigar com
  `torch.profiler` se o uso real pedir.

## Troubleshooting

### "Unexpected token 'I'" no UI
Significa que o backend devolveu um HTML "Internal Server Error" em vez
de JSON — alguma exception escapou dos handlers. Ver `docker compose logs
inemaimg` pra a traceback real.

### "model X is gated on HuggingFace"
Falta `HF_TOKEN` no ambiente ou o usuário do token não tem acesso ao repo.
Ver seção *Acesso a modelos gated* acima.

### `prewarm_status: "failed"`
Olhar `prewarm_error` no `/health` e `docker compose logs inemaimg`.
Causas comuns: sem espaço em disco pro cache HF, sem token pra um modelo
gated, `INEMAIMG_PREWARM` apontando pra um id fora do registry.

### Geração trava e GPU fica em 96% sem retornar
Aconteceu no NGC 25.03. O fix foi bumpar a base image — se reaparecer no
26.03, é regressão nova e provavelmente outro op caindo em fallback sm_120.
