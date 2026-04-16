# Integração timesmkt3 ↔ inemaimg

Histórico e estado atual da integração do inemaimg como provider default do
pipeline de imagens do timesmkt3 (Abr/2026).

## O que foi feito

### 1. Provider Node (`pipeline/generate-image-inemaimg.js` no timesmkt3)

Provider completo seguindo o mesmo padrão dos outros (`generate-image-kie.js`,
`generate-image-pollinations.js`, `generate-image-piramyd.js`).

- Lê as envs: `INEMAIMG_URL`, `INEMAIMG_MODEL`, `INEMAIMG_QUALITY`, `INEMAIMG_TIMEOUT_S`
- Expõe `generateImage(outputPath, prompt, model, aspectRatio)` + `DEFAULT_MODEL` + `AVAILABLE_MODELS` + `getHealth()`
- Tem modo CLI: `node pipeline/generate-image-inemaimg.js <out.png> "<prompt>" [model] [ratio]`
- FAST mode (default): 512² base, step counts mínimos por modelo
- HIGH mode: 1024² base, step counts dos docs oficiais de cada modelo

Modelos registrados (devem bater com `REGISTRY` em `server.py`):

| id                | notes                                                  |
|-------------------|--------------------------------------------------------|
| `flux2-klein`     | 4 steps distilled, mais rápido, non-commercial         |
| `qwen-edit-2511`  | 15-40 steps, edit-capable, Apache 2.0                  |
| `ernie`           | 20-50 steps, T2I puro, open                            |
| `flux2-dev`       | 10-28 steps, premium 4-bit, non-commercial             |

### 2. Worker plumbing (timesmkt3)

- `pipeline/worker.js` — `require('./generate-image-inemaimg')` + branch `if (p === 'inemaimg') return inemaimgProvider;` dentro de `getImageProvider()`
- `pipeline/worker-video-pro.js`, `worker-video.js`, `worker-ad-creative.js` — fallback de modelo agora usa `imageProvider.DEFAULT_MODEL` primeiro, evitando o bug de passar `z-image` pro inemaimg
- `config/env.js` — novo helper `getDefaultImageModel(provider)` centralizando defaults por provider (inemaimg→flux2-klein, kie→z-image, pollinations→flux, piramyd→flux)
- `telegram/campaign-utils.js`, `bot.js`, `bot-rerun.js`, `bot-text-pending.js` — tudo usa o helper; config table do bot lista `inemaimg / kie / pollinations / piramyd` como provider e `flux2-klein / qwen-edit-2511 / ernie / flux2-dev / z-image / flux` como modelos

### 3. `.env` do timesmkt3

```bash
IMAGE_PROVIDER=inemaimg
KIE_DEFAULT_MODEL=z-image
INEMAIMG_URL=http://localhost:8000
INEMAIMG_MODEL=flux2-klein
INEMAIMG_QUALITY=fast
```

A troca via bot no meio de uma campanha:

```
provider inemaimg          → flux2-klein (pega default do provider)
provider kie               → z-image
modelo qwen-edit-2511      → sobrescreve modelo da sessão
```

`INEMAIMG_QUALITY` só é lida no require do módulo, então pra trocar entre
`fast` / `high` precisa editar `.env` e dar `pm2 restart timesmkt3-worker`.
Se no futuro quiser trocar por campanha, é só refatorar `FAST` pra ser
computado dentro de `generateImage()` + aceitar `quality` como parâmetro.

## Pendente / pontos abertos

### 1. Reference images (imagens de referência)

O server inemaimg **já aceita** imagens de referência no endpoint `/generate`:

```python
# server.py GenerateRequest
images: list[str] | None = Field(
    default=None,
    description="Reference images as base64-encoded PNG/JPEG (edit models only).",
)
```

São passadas pra `loader_cls.generate(pipe, req, pil_images)`. Modelos
edit-capable (ex: `qwen-edit-2511`) usam as imagens como entrada; modelos T2I
puros ignoram.

**Mas o client Node `generate-image-inemaimg.js` não manda ainda.** O body atual é:

```js
const body = {
  model,
  prompt,
  width: size.width,
  height: size.height,
  seed: Math.floor(Math.random() * 9999999),
  ...defaults,  // steps, guidance_scale, true_cfg_scale, etc.
};
```

Pra suportar referência no client, o refactor seria:

```js
async function generateImage(outputPath, prompt, model = DEFAULT_MODEL, aspectRatio = '1:1', referenceImages = []) {
  const size = ASPECT_RATIO_SIZES[aspectRatio] || ASPECT_RATIO_SIZES['1:1'];
  const defaults = MODEL_DEFAULTS[model] || {};

  const images = referenceImages
    .filter(Boolean)
    .map((p) => fs.readFileSync(p).toString('base64'));

  const body = {
    model, prompt,
    width: size.width, height: size.height,
    seed: Math.floor(Math.random() * 9999999),
    ...(images.length > 0 ? { images } : {}),
    ...defaults,
  };
  // ...
}
```

E no worker (`worker-video-pro.js` / `worker-ad-creative.js`) precisaria
passar a imagem de referência — por exemplo, a foto do produto do brand, o
screenshot do site, ou uma imagem da pasta `prj/<projeto>/assets/`.

Casos de uso naturais:
- **Multiple angles** — qwen-edit com lora `multiple-angles`, gera outros ângulos do mesmo produto a partir de 1 foto
- **Face swap** — qwen-edit com lora `face-swap` pra manter consistência de personagem entre cenas
- **Product placement** — edit-mode pra colocar o produto em cenários diferentes

Quando for implementar isso: precisa decidir (a) onde a imagem de referência
mora no payload (`image_reference` no job.data?), (b) se é por cena (scene
plan define) ou global (campaign define), (c) qual modelo cai quando tem
referência (auto-switch pra qwen-edit-2511?).

### 2. Autostart do server inemaimg

Hoje o server não está no PM2 do timesmkt3 — só o daemon TTS, bot e worker. Se
o server inemaimg ficar morto, o pipeline inteiro falha quando tenta gerar
imagem via API.

Opções:
- Adicionar `inemaimg-server` no `ecosystem.config.cjs` do timesmkt3 (mesmo padrão do tts-daemon)
- Ou subir separado no PM2 do próprio inemaimg

O systemd unit `pm2-nmaldaner.service` já está enabled — qualquer process que
esteja no dump do PM2 volta sozinho no boot.

### 3. Fallback quando server fora

Hoje o worker não tem fallback — se `http://localhost:8000` não responder, a
geração falha e o stage 2 aborta. Pra robustez produção, `getImageProvider`
podia detectar server fora e cair pra `pollinations` ou `kie`. Mas por
enquanto só vale se o server realmente ficar instável.

## Git status no timesmkt3 (Abr/2026)

Não-commitado (aguardando validação fim-a-fim):

- `.env` — IMAGE_PROVIDER=inemaimg + envs novas
- `pipeline/generate-image-inemaimg.js` (novo)
- `pipeline/worker.js` — require + branch em getImageProvider
- `pipeline/worker-video-pro.js`, `worker-video.js`, `worker-ad-creative.js` — fallback provider-aware
- `config/env.js` — helper `getDefaultImageModel`
- `telegram/campaign-utils.js`, `bot.js`, `bot-rerun.js`, `bot-text-pending.js` — usar o helper
- (separado: `pipeline/worker-video-gatilhos.js` com fix de narração com pausas — nada a ver com inemaimg)
