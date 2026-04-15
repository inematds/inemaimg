# Projeto — Servidor Multi-Modelo de Imagens

Servidor local que expõe múltiplos modelos de geração/edição de imagem via API HTTP, consumido pelo **timesmkt3** como se fosse um serviço de imagens externo. Cada modelo pode ser ativado/desativado sob demanda.

---

## Objetivo

Ter um único endpoint HTTP que o timesmkt3 consome, com capacidade de escolher qual modelo usar em cada requisição. O servidor roda local, isolado em Docker com acesso à GPU, e mantém apenas o modelo ativo carregado na VRAM.

---

## Modelos Disponíveis

Ver `MODELOS.md` para detalhes técnicos e comparações completas.

| ID interno | Modelo HuggingFace | Tipo | VRAM prática |
|---|---|---|---|
| `ernie` | `baidu/ERNIE-Image` | Text-to-image | 24 GB |
| `qwen-edit-2511` | `Qwen/Qwen-Image-Edit-2511` | Edição + multi-imagem | 24 GB (offload) |
| `flux2-klein` | `black-forest-labs/FLUX.2-klein-9B` | T2I + edição rápida | 24 GB (offload) |
| `flux2-dev` | `diffusers/FLUX.2-dev-bnb-4bit` | T2I premium 4-bit | 24 GB |

**LoRAs opcionais (aplicados sobre Qwen-Edit-2511):**
- `AdversaLLC/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` — 96 ângulos de câmera
- `Alissonerdx/BFS-Best-Face-Swap` — face/head swap

---

## Arquitetura

```
                    ┌──────────────────────┐
  timesmkt3  ───▶   │  FastAPI Gateway     │
                    │  POST /generate      │
                    │  body: { model, ... }│
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Model Registry     │
                    │   (hot swap)         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    GPU (1 modelo)    │
                    └──────────────────────┘
```

**Estratégia de carregamento:** hot swap com 1 modelo ativo por vez.

Quando uma requisição chega para um modelo diferente do atual:
1. Descarrega modelo atual (`del pipe` + `torch.cuda.empty_cache()`)
2. Carrega o novo modelo do cache local do HuggingFace
3. Processa a requisição

**Pré-aquecimento:** o modelo mais usado (a definir após métricas) é carregado no boot do container, para que a primeira requisição não pague o custo de cold start.

---

## Stack Técnica

- **Python 3.11+**
- **FastAPI** — servidor HTTP
- **Uvicorn** — ASGI runner
- **diffusers** (latest do GitHub) — pipelines de imagem
- **torch** com CUDA 12.4
- **Pillow** — manipulação de imagem
- **Docker + NVIDIA Container Toolkit** — isolamento com acesso à GPU

---

## Endpoints da API

### `GET /health`
Status do servidor e modelo atualmente carregado.

**Response:**
```json
{
  "status": "ok",
  "loaded_model": "qwen-edit-2511",
  "gpu_memory_used_gb": 22.4
}
```

### `GET /models`
Lista todos os modelos disponíveis e o atualmente ativo.

**Response:**
```json
{
  "available": ["ernie", "qwen-edit-2511", "flux2-klein", "flux2-dev"],
  "loaded": "qwen-edit-2511"
}
```

### `POST /models/load`
Pré-carrega um modelo específico (útil para aquecer antes de rajada).

**Request:**
```json
{ "model": "flux2-klein" }
```

### `POST /generate`
Gera imagem com o modelo especificado. Troca o modelo em memória se necessário.

**Request (text-to-image):**
```json
{
  "model": "ernie",
  "prompt": "...",
  "width": 1024,
  "height": 1024,
  "steps": 50,
  "guidance_scale": 4.0,
  "seed": 42
}
```

**Request (edição com referência):**
```json
{
  "model": "qwen-edit-2511",
  "prompt": "...",
  "images": ["<base64>", "<base64>"],
  "steps": 40,
  "true_cfg_scale": 4.0,
  "lora": "multiple-angles",
  "lora_weight": 0.9
}
```

**Response:**
```json
{
  "image": "<base64 PNG>",
  "model_used": "qwen-edit-2511",
  "generation_time_s": 28.4
}
```

---

## Hardware-alvo real (confirmado 2026-04-15)

O servidor roda num **NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)**:

- Arquitetura: **aarch64** (ARM64)
- GPU: **NVIDIA GB10** (Blackwell, compute capability sm_120)
- Memória: **119 GB unificada** entre CPU e GPU (LPDDR5X)
- CUDA: 13.0 | Driver: 580.95.05
- Container runtime: Docker + `nvidia` runtime (via NVIDIA Container Toolkit / CDI)

**Consequências no design:**

1. **Sem CPU offload.** Chamadas como `pipe.enable_model_cpu_offload()` são contraproducentes aqui — a GPU já acessa a RAM do sistema diretamente via memória unificada. Offload só adiciona cópias redundantes.
2. **Hot swap deixa de ser obrigatório.** Com 119 GB unificados dá pra manter 2–3 modelos grandes quentes ao mesmo tempo. Estratégia atual: manter apenas 1 modelo carregado no MVP, migrar pra LRU multi-modelo na Fase 2 se o padrão de tráfego justificar.
3. **Base image arm64.** `nvcr.io/nvidia/pytorch:25.03-py3` (NGC) tem builds arm64-sbsa com suporte oficial a Blackwell. Imagens `pytorch/pytorch` do Docker Hub não atendem (x86_64).
4. **Diffusers do git main.** As pipelines `QwenImageEditPlusPipeline`, `Flux2KleinPipeline`, `Flux2Pipeline` e `ErnieImagePipeline` são recentes — precisam de `diffusers` instalado de `git+https://github.com/huggingface/diffusers`, não do pip release.
5. **Quantização.** GB10 suporta FP4 nativamente. FLUX.2-dev em 4-bit (bnb) é o caminho natural quando entrar no registry.

## Docker Setup

### Requisitos no host
- NVIDIA Container Toolkit instalado (ou CDI equivalente)
- Driver NVIDIA compatível com CUDA 13.x
- GPU Blackwell / GB10 (arquitetura-alvo deste projeto) — ou qualquer GPU com 24 GB+ de VRAM discreta se reimplantado em outro host, desativando o assumption de unified memory nos loaders

### Estrutura do projeto
```
inemaimg/
├── MODELOS.md              # documentação dos modelos
├── PROJETO.md              # este arquivo
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── server.py               # FastAPI + registry + loaders
├── loaders/
│   ├── __init__.py
│   ├── ernie.py
│   ├── qwen_edit.py
│   ├── flux2_klein.py
│   └── flux2_dev.py
└── routers/
    └── generate.py         # roteamento por tipo de modelo
```

### Volumes importantes
- `~/.cache/huggingface` montado no container — evita re-download dos ~80 GB de modelos a cada rebuild
- Diretório de saída para imagens geradas (opcional, se não retornar só via base64)

### Comando de execução
```bash
docker compose up -d
```

O `docker-compose.yml` configura:
- `--gpus all` para acesso à GPU
- `restart: unless-stopped` para persistência
- Volume do cache HuggingFace
- Porta 8000 exposta

---

## Fluxo de Uso pelo timesmkt3

```
timesmkt3                        servidor-imagens
    │                                   │
    │  POST /generate                   │
    │  {model: "qwen-edit-2511", ...}   │
    ├──────────────────────────────────▶│
    │                                   │ (troca modelo se necessário)
    │                                   │ (gera imagem)
    │                                   │
    │        {image: "<base64>", ...}   │
    │◀──────────────────────────────────┤
    │                                   │
```

O timesmkt3 trata a resposta como trataria qualquer API externa de imagem — decodifica o base64 e salva/exibe.

---

## Regras Operacionais

1. **Fila serializada por GPU.** Nunca processar duas requisições simultâneas na mesma GPU (OOM garantido). Usar `asyncio.Lock` ou fila Celery/RQ.

2. **Cache persistente do HuggingFace.** Volume sempre montado. Perder o cache = rebaixar 80 GB.

3. **Pré-aquecer no boot.** Carregar o modelo dominante do tráfego durante a inicialização do container.

4. **Health check com info real.** `/health` retorna qual modelo está quente — timesmkt3 pode checar antes de mandar requisições pesadas.

5. **Logs por requisição.** Tempo de carregamento do modelo, tempo de inferência, VRAM usada. Base para otimização futura.

6. **Sem hot swap durante rajada.** Se possível, timesmkt3 deve agrupar requisições por modelo para evitar swap constante.

---

## Roadmap

### Fase 1 — MVP (atual)
- Docker com 1 GPU, hot swap entre os 4 modelos
- FastAPI com endpoints `/generate`, `/models`, `/health`
- Integração básica com timesmkt3

### Fase 2 — Após métricas reais
- Decidir entre:
  - Manter hot swap se um modelo domina o tráfego
  - Migrar para pool em RAM se o uso for balanceado (~128 GB RAM)
  - Multi-GPU se o throughput exigir (1 modelo por GPU)
- Fila robusta (Celery ou RQ) se houver concorrência real
- Métricas exportadas (Prometheus)

### Fase 3 — Produção
- Observabilidade completa
- Rate limiting
- Autenticação entre timesmkt3 e servidor
- Backup do cache de modelos

---

## Considerações de Licença

Ver `MODELOS.md` para matriz completa de licenças.

**Ponto crítico:** FLUX.2-dev e FLUX.2-klein-9B têm licença **non-commercial**. Se o timesmkt3 for produto comercial vendido a clientes:
- Priorizar **Qwen-Image-Edit-2511** (Apache 2.0) como motor principal
- Priorizar **ERNIE-Image** (aberta) para T2I puro
- Usar FLUX.2-* apenas em prototipagem interna ou com acordo comercial formal com Black Forest Labs

A arquitetura multi-modelo facilita essa política: os modelos não-comerciais podem ser desabilitados no `MODEL_REGISTRY` sem afetar o resto do sistema.

---

## Status Atual

- [x] Documentação de modelos (`MODELOS.md`)
- [x] Documentação do projeto (`PROJETO.md`)
- [x] Confirmação do hardware-alvo (GB10 / DGX Spark)
- [x] `Dockerfile` + `docker-compose.yml` (MVP)
- [x] `server.py` com registry + lock serializado por GPU
- [x] Loader `qwen-edit-2511` (motor principal comercial)
- [x] Loaders `ernie`, `flux2-klein`, `flux2-dev` (não testados end-to-end ainda)
- [x] Build do container + download do modelo Qwen-Edit-2511
- [x] Playground web em `/` (HTML estático)
- [x] Shim Blackwell/sm_120 pra `nvrtc` (`loaders/_blackwell_shims.py`)
- [x] `pipe()` rodando em `run_in_executor` (não trava event loop)
- [ ] Validar base image bump (25.03 → 26.03) resolveu perf de ~290s/4-step
- [ ] Integração com timesmkt3
- [ ] Teste de carga inicial
- [ ] Autenticação (reverter bind `0.0.0.0` temporário)
