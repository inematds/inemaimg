# Um prompt, vários destinos: como adaptar por provedor e modelo

> **O que este doc é:** a camada de tradução. O prompt de *intenção* é um só — quem
> o escreve é o domínio (no promoavatar3, a fase de texto, por segmento). O que muda
> por destino é o **dialeto**: tamanho útil, se aceita negative prompt, se guidance
> faz algo, como pedir texto legível, como passar referência.
>
> **O que este doc não é:** substituto de
> [`flux2-klein.md`](./flux2-klein.md) (o guia fundo do klein) nem de
> [`comparativo-modelos.md`](./comparativo-modelos.md) (qual modelo usar). Aqui é só
> a adaptação.
>
> Criado em 2026-08-09.

## A regra que evita retrabalho

**Escreva a intenção uma vez, no dialeto do FLUX** — prosa descritiva na ordem
sujeito → cenário → detalhes → iluminação → atmosfera. É o dialeto mais exigente e
o mais transferível: quase todo destino aceita prosa; o contrário (keywords estilo
SDXL viradas em prosa) não funciona.

Depois **corte ou acrescente** conforme a tabela do destino. Nunca reescreva do zero.

## Verificado nesta casa (GB10) × não verificado

Os quatro modelos locais estão documentados por medição sua. **Kie, fal e Agnes não
foram testados aqui** — não há chave nem rede neste ambiente para exercitá-los. As
linhas deles trazem o que a documentação pública de cada um diz e ficam marcadas
**(não verificado)** até alguém rodar de verdade. Não misture os dois níveis de
confiança: o valor deste doc é justamente saber qual é qual.

## Locais (inemaimg) — medido

| modelo | prosa ou keywords | tamanho útil | negative | guidance | steps | referência | armadilha |
|---|---|---|---|---|---|---|---|
| **flux2-klein** | prosa, sempre | 30–80 palavras (teto ~300) | **não suporta** | **ignorado** (fixo 1.0) | **4, e é o ótimo** — subir piora | até 4 imagens | texto legível sai distorcido; `masterpiece, 8k` é ignorado |
| **flux2-dev** | prosa | idem klein | não suporta | ativo | ~28 | sim | não sobe nesta máquina de propósito (falta `bitsandbytes`) — erro 500 esperado, não é bug |
| **qwen-edit-2511** | prosa, com a instrução de *edição* explícita | curto e direto | — | — | ~40 | **exige 1–3 imagens** | é editor: sem referência não faz o que se espera. Único com licença comercial (Apache 2.0) |
| **ernie** | prosa | média | — | — | ~50 | **rejeita `images`** | T2I puro; menos detalhe (8B) |

Três coisas que valem para todos os locais e que a doc do klein já mediu:

1. **Iluminação é o parâmetro de maior impacto.** Diga fonte, qualidade, direção,
   temperatura e como bate na superfície. `"warm golden-hour sidelight from the
   left, long shadows on concrete"` vale mais que qualquer adjetivo de qualidade.
2. **Quality boosters não fazem nada.** `masterpiece, 8k, ultra detailed` foi treino
   de outra família de modelos.
3. **Texto legível dentro da imagem é o limite dos locais.** Se a peça precisa de
   palavra correta (placa, cartaz, UI, embalagem), o local não entrega — é o único
   caso em que a skill `generate` já roteia para rota paga.

## Nuvem — **não verificado nesta casa**

Preencher cada bloco na primeira chamada real, com o corpo copiado da página do
provedor. É a mesma regra que a skill `generate` prega para as receitas de modelo:
dez minutos de preenchimento honesto evitam tentativa-e-erro **paga**.

### kie — (não verificado)

- **Receita:** `~/.claude/skills/generate/models/kling-kie.md` já existe, mas é de
  **vídeo** (Kling). Para imagem, criar `kie-imagem.md` no mesmo formato.
- **A preencher:** endpoint, auth (header e nome da variável), corpo exato, se a
  resposta é síncrona ou por polling, e onde o arquivo aparece (URL que expira × base64).
- **Dialeto:** depende do modelo escolhido dentro da Kie — a plataforma revende
  vários. **Registre o modelo junto com o prompt**, senão o doc não serve para nada.

### fal — (não verificado)

- **Receita:** `~/.claude/skills/generate/models/gpt-image-2-fal.md` existe e é a
  rota já escolhida para **texto legível na imagem** (~$0,05).
- **A preencher:** o bloco "Request / Response" da própria receita continua com o
  aviso de "copiar o corpo exato da página do modelo" — ou seja, ainda não foi
  exercitado.
- **Dialeto:** modelos da linha GPT Image respondem bem a instrução direta e
  literal para o texto ("the sign reads exactly: ABRIL"), diferente do FLUX, onde
  isso não adianta porque a limitação é de renderização.

### Agnes — **VERIFICADO em 2026-08-09**

Exercitada de verdade, com chamadas reais, ao implementar o adaptador do
`gen-imagem.py`. Custo **US$ 0**, ~10 s por imagem.

```
POST {AGNES_BASE_URL}/images/generations      # a base JÁ inclui /v1 nesta casa
Authorization: Bearer $AGNES_API_KEY
{"model":"agnes-image-2.1-flash","prompt":"...","size":"1088x704",
 "extra_body":{"response_format":"b64_json"}}
→ {"data":[{"b64_json":"...","url":"...","revised_prompt":"..."}], "usage":{...}}
```

Quatro coisas que só a chamada real mostrou, e que **contradizem o que se supunha**:

| achado | consequência |
|---|---|
| **Prompt em PT funcionou.** A skill `imagens-agnes` diz que português dispara filtro e devolve 400 — não foi o que aconteceu com um prompt comum de cena | não é preciso traduzir os prompts do domínio. Uma amostra não derruba a regra para conteúdo sensível; trate como "PT não é bloqueio automático" |
| **Não existe seed.** O serviço não aceita e não devolve | **o determinismo do reel cai neste provedor**: mesma `--seed-key` gera imagem diferente a cada render. É escolha do serviço |
| **O tamanho pedido é sugestão.** Pedimos `1088x736`, veio `1248x832` — proporção respeitada, pixels não | quebraria o reel em silêncio: `preparar.py` compara dimensão para reaproveitar, e o empilhamento espera a faixa exata. O adaptador **normaliza** (corte pelo centro, nunca esticar) |
| `AGNES_MODEL` do `.env` é o modelo de **chat** | mandar ele devolve `400 ... is a chat model`. O de imagem é `agnes-image-2.1-flash`, em `AGNES_IMAGE_MODEL` |

Erro de percurso que vale registrar: concatenar `/v1` às cegas na base rendeu
`404 POST /v1/v1/images/generations`. O adaptador testa o sufixo.

Retry com backoff está implementado por causa dos ~34% de 503 que a skill
`imagens-agnes` documenta — sem isso, um terço dos segmentos do reel nasceria sem
imagem por azar de momento.

## O que NÃO é adaptação de prompt

Trocar de provedor **não** é mudar um endereço. `INEMAIMG_HOST` aponta para outro
inemaimg — não para a kie, fal ou Agnes: corpo do request, autenticação e formato da
resposta são outros. Isso é **adaptador**.

Desde 2026-08-09 ele existe, no `gen-imagem.py` dos repos de domínio:

```bash
IMG_PROVEDOR=inemaimg   # default — GPU local, custo zero, seed respeitado
IMG_PROVEDOR=agnes      # API, custo US$ 0, SEM seed
IMG_ENV_PATH=/root/.config/inemaccbot/imagem.env   # de onde vem a chave
```

`kie` e `fal` ainda não estão implementados: o script recusa com
`IMG_PROVEDOR=<x> nao implementado` em vez de fingir. Acrescentar cada um é uma
função — o difícil não é o código, é exercitar de verdade antes de confiar.

Numa máquina sem GPU há ainda o caminho sem provedor nenhum: túnel para o inemaimg
de casa (`ssh -R 8000:localhost:8000 <vps>`) com `INEMAIMG_HOST=http://localhost:8000`.
Mesmo modelo, **mesmo determinismo**, custo zero — é a única opção que preserva a
promessa de "mesma seed-key, mesma imagem".

## Como manter isto honesto

Quando um provedor de nuvem for usado pela primeira vez: rode, copie o corpo real
que funcionou, cole na receita, e **tire a marca "(não verificado)"** deste doc.
Enquanto a marca estiver lá, trate a linha como hipótese.
