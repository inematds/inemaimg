# inemaimg

Servidor local de geração/edição de imagens com hot-swap entre múltiplos
modelos, exposto via FastAPI. Roda em Docker com acesso à GPU e é consumido
pelo timesmkt3 como se fosse uma API externa de imagens.

Visão e decisões de arquitetura → [`PROJETO.md`](./PROJETO.md)
Comparação técnica dos modelos → [`MODELOS.md`](./MODELOS.md)

## Requisitos

- NVIDIA Container Toolkit + driver compatível com CUDA 12.8+
- GPU com suporte a Blackwell / sm_120 (alvo: DGX Spark / GB10, 119 GB unificados)
  ou qualquer GPU com ≥ 24 GB de VRAM discreta
- Docker Compose v2
- Cache de modelos do HuggingFace em `~/.cache/huggingface` (volume montado)

## Subir o serviço

```bash
# 1. (primeira vez) exportar token do HF pra repos gated
export HF_TOKEN=hf_xxx  # ver seção "Acesso a modelos gated" abaixo

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
| `flux2-klein` | T2I + edição, 4 steps | FLUX **Non-Commercial** | Gated no HF (aprovação quase instantânea). |
| `flux2-dev` | T2I premium, 4-bit | FLUX **Non-Commercial** | Usa `diffusers/FLUX.2-dev-bnb-4bit`. Gated. |

> **Aviso de licença:** `flux2-klein` e `flux2-dev` são FLUX Non-Commercial.
> Se o inemaimg for embutido num produto comercial sem acordo com a BFL,
> remova-os do `REGISTRY` em `server.py` antes do deploy.

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
