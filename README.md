# PIVC — Sistema de Identificação de Desperdício de Materiais em Obras

Sistema de visão computacional desenvolvido para a APS (5º/6º Ciência da Computação),
dentro do tema **"Desenvolvimento de um sistema de identificação por imagens para
verificação de componentes relacionados à Engenharia Civil em obras (PIVC)"**.

O sistema **recebe uma imagem de uma obra e julga se há desperdício de materiais**
(entulho de papelão, madeira, gesso, plástico etc. espalhado na cena). A cada
julgamento, um **operador humano confirma ou corrige** a resposta — e o sistema
**registra a experiência numa memória persistente e retreina seu modelo**,
aprendendo progressivamente e reduzindo erros futuros.

A stack escolhida foi **Python + OpenCV + NumPy + scikit-learn**, uma das opções
previstas no roteiro da APS, funcionando 100% offline (sem APIs pagas).

---

## 1. Como instalar e rodar

**Modo "swipe" (recomendado para a bancada de demonstração):**

**Dois cliques no `executar.bat`** — abre o site no navegador sozinho
(mantendo a janela preta do servidor aberta enquanto usa; se a porta padrão
estiver ocupada, o sistema sobe na próxima livre automaticamente). Por linha
de comando, o mesmo efeito é:

```bat
executar.bat site
```

Abre no navegador o front-end web com cartões arrastáveis estilo Tinder
(tema preto + amarelo de capacete de EPI): solte as fotos da obra, veja o
veredito e a probabilidade **abaixo da foto**, e confirme arrastando o
cartão — **→ direita = desperdício, ← esquerda = sem desperdício**. Um toque
na foto alterna para a visão com as regiões detectadas contornadas. A
distância de captura é padronizada em 5 m (parametrizada em
`pivc_core/configuracao.py`, registrada nos metadados de cada experiência —
não ocupa espaço na tela). Cada swipe vai para a mesma memória persistente
dos outros modos e retreina o modelo.

**Importante (Windows):** este computador tem mais de um Python instalado. Para
não depender de qual Python o terminal escolhe, use sempre o **`executar.bat`** —
ele cria e usa um ambiente virtual próprio do projeto (`.venv/`) na primeira
execução e garante que as bibliotecas certas sejam carregadas:

```bat
executar.bat analisar "foto.jpg"      :: analisa uma imagem e pede confirmação
executar.bat painel                   :: interface gráfica de janela (Tkinter)
executar.bat status                   :: estado da memória e do modelo
```

Alternativa manual (se preferir gerenciar o ambiente por conta própria):

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # ou: pip install opencv-python numpy scikit-learn pillow flask
.venv\Scripts\python main.py analisar "foto.jpg"
```

Todas as bibliotecas são gratuitas e o sistema funciona 100% offline.

| Comando | O que faz |
|---|---|
| `python main.py analisar foto.jpg` | Analisa uma imagem e pede sua confirmação |
| `python main.py analisar foto1.jpg foto2.png` | Analisa várias imagens |
| `python main.py analisar --pasta fotos_obra/` | Analisa todas as imagens de uma pasta |
| `python main.py analisar foto.jpg --largura-m 5` | Informa a largura real da cena (m) para a área do descarte em m² |
| `python main.py analisar foto.jpg --sem-feedback` | Só julga, sem pedir confirmação |
| `python main.py adicionar pasta/ --rotulo d` | Ensina um lote inteiro (`d`=desperdício, `l`=limpo) |
| `python main.py status` | Mostra memória, modelo e taxa de acerto |
| `python main.py relatorio` | Exporta relatório de experiências (Markdown + CSV) |
| `python main.py painel` | Abre a interface gráfica |
| `python main.py resetar` | Apaga toda a memória aprendida |

A interface gráfica também abre direto com `python gui.py`.

### Fluxo de confirmação humana (o coração do sistema)

```
VEREDITO: DESPERDÍCIO DE MATERIAIS DETECTADO
Probabilidade de desperdício: 85% (confiança alta)
─────────────────────────────────────────────
Linha de pensamento do sistema:
  • Passo 1 — Pré-processamento: ...
  • Passo 2 — Indicadores visuais (Canny, Sobel, HSV, contornos): ...
  • Passo 3 — Heurística de engenharia (conhecimento a priori): p = 0.68.
  • Passo 4 — Memória de casos (analogia forte): 4 casos semelhantes...
  • Passo 5 — Modelo incremental (regressão logística, 4 experiências): p = 0.00.
  • Passo 6 — Fusão adaptativa (pesos: ...) → p final = 0.45.
  • Passo 7 — Identificação de material: assinaturas visuais por região
    (cor/textura/contorno) indicam Papelão 100%. Memória de material
    (2 confirmações, peso 0.23) vota: Papelão 90% → material predominante:
    Papelão (confiança 94%).
  • Passo 8 — Área do descarte (cena ≈ 12,0 m², largura 4,00 m):
    Papelão 2,5 m² → total ≈ 2,5 m².
O sistema acertou? [s]im / [n]não, corrigir / [p]ular:
Rótulo correto — [d] desperdício / [l] local limpo:
Material predominante — [g]esso, [p]apelão, [m]adeira, [l]lonas/plástico,
[a]lvenaria, [o]utro (Enter mantém Papelão):
Observação (opcional, Enter pula):
✔ Experiência #4 registrada na memória (o sistema ERROU e foi CORRIGIDO).
  Modelo incremental consolidado com 4 experiências.
```

### Identificação do tipo de material e área do descarte (m²)

Além do veredito (há/não há desperdício), o sistema identifica **o tipo de
material predominante** — papelão, madeira, gesso/drywall, plástico/lonas,
entulho de alvenaria — classificando **cada região candidata** por assinaturas
visuais (marrom saturado x empoeirado, claro, cinza médio, vibrante, textura
interna — `pivc_core/materiais.py`). A imagem-relatório ganha uma **cor por
material e legenda** com a composição do desperdício.

O operador confirma ou corrige o material junto com o veredito; essas
confirmações formam uma **memória de material** que vota por analogia nas
análises seguintes (mesmo princípio da memória de casos, com peso que cresce
com o número de confirmações e cai quando a analogia é fraca).

O sistema também **conta os aglomerados** (montes) de desperdício por
material — cada região de ≥ 1% da cena é um monte ("2 aglomerados: Papelão 1,
Gesso 1"). A unidade de contagem é o monte porque peças individuais dentro de
uma pilha se tocam e se sobrepõem numa foto 2D: contá-las exigiria detecção
de objetos por deep learning (trabalho futuro).

Sobre o material identificado, o sistema estima **quanta área de material está
desperdiçada**, em m²:

```
área da cena (m²) = largura_real² × (altura/largura da imagem)
área por material = fração da cena detectada × área da cena
```

A **largura real da cena** não vem da foto (imagem 2D não tem escala absoluta):
o operador informa `--largura-m` (terminal) ou o campo "Largura real da cena"
(painel); sem isso vale o padrão de 4,0 m — e a saída indica qual largura foi
usada.

Cada análise gera também uma **imagem-relatório anotada** em
`dados/anotacoes/`, com as regiões candidatas a entulho contornadas.

### Captura pela câmera (imagem inserida na hora)

No **site (Engenhender)**, o botão **📷 câmera** da barra de ferramentas abre
um modal com **pré-visualização ao vivo** (câmera do próprio navegador):
enquadre a área da obra no padrão de 5 m e clique em *capturar* — a foto vai
direto para a pilha de cartões, sem passar por arquivo ou explorador. Na
interface Tkinter há o equivalente **📸 Capturar da câmera**. A imagem fica
em `dados/capturas/` e entra no mesmo fluxo de confirmação humana. Isso
atende à aquisição por **câmera** prevista no roteiro da APS ("arquivos de
imagem ou vídeo… ou câmera").

### Login "Engenhender"

Ao abrir o site, a tela de login pede apenas o **nome do operador** — não é
controle de acesso, é a assinatura de quem confirma: cada experiência gravada
na memória carrega o campo `operador`, deixando o histórico auditável por
pessoa (relatórios). A tagline do produto: *"aqui o match é com a economia"*.

### Limpar a inteligência do modelo

O botão **🧹 limpar IA** da barra de ferramentas abre um modal com o resumo
do que será apagado e **confirmação em duas etapas**. Confirmado, apaga todas
as experiências da memória e o modelo aprendido — o sistema volta ao
conhecimento a priori (heurística de engenharia) e recomeça a aprender do
zero com as próximas confirmações. Equivalente no terminal:
`python main.py resetar`.

### Materiais identificados (e como cadastrar mais)

| Material | Assinatura visual | Cor |
|---|---|---|
| Papelão | marrom empoeirado + textura fibrosa | marrom |
| Madeira | marrom saturado + bordas internas (veios) | laranja |
| Gesso/Drywall | claro, pouco saturado | amarelo claro |
| Plástico/Lonas | cores vibrantes, superfície lisa | magenta |
| Entulho de alvenaria | cinza médio dominante + irregular | azul claro |
| **Tijolo/Cerâmica** | vermelho-cerâmico saturado | vermelho |
| **Metal/Tubos** | brilho especular + superfície lisa | azul-aço |

O passo a passo para cadastrar novos materiais (5 pontos do arquivo) está
documentado no topo de `pivc_core/materiais.py`.

---

## 2. Arquitetura

```
                       ┌────────────────────────────────────────────┐
 imagem (arquivo) ───► │  MÓDULO DE VISÃO (pivc_core/visao.py)      │
 ou câmera ao vivo    │  • pré-processamento (CLAHE, Gauss)        │
                       │  • pré-processamento (CLAHE, Gauss)        │
                       │  • bordas: Canny + Sobel                   │
                       │  • cor: estatísticas HSV                   │
                       │  • textura: blocos 16x16 + entropia        │
                       │  • segmentação/morfologia + contornos      │
                       │  • supressão de sombras projetadas         │
                       └───────────────┬────────────────────────────┘
                                       │  vetor de 18 características
                       ┌───────────────▼────────────────────────────┐
                       │  DETETOR (pivc_core/detetor.py)            │
                       │                                            │
                       │  3 LINHAS DE PENSAMENTO:                   │
                       │  1. Heurística de engenharia (a priori)    │
                       │  2. Memória de casos (k-NN por analogia)   │
                       │  3. Modelo incremental (regressão logística)│
                       │                                            │
                       │  FUSÃO ADAPTATIVA → probabilidade final    │
                       └───────────────┬────────────────────────────┘
                                       │  veredito + cadeia de raciocínio
                                       ▼
                        OPERADOR HUMANO confirma [s] ou corrige [n]
                                       │
                       ┌───────────────▼────────────────────────────┐
                       │  MEMÓRIA (pivc_core/memoria.py)            │
                       │  • memoria.json — banco de casos (vetores) │
                       │  • o modelo é reconstruído deste arquivo   │
                       │    a cada inicialização (sem binário)      │
                       └────────────────────────────────────────────┘
```

### As três linhas de pensamento

1. **Heurística de engenharia** — conhecimento a priori que funciona desde o
   primeiro uso (fase fria): soma ponderada de 8 indicadores visuais de
   entulho. O mais forte é composto: *pilha de entulho forma UMA região
   grande da cena e de contorno irregular (solidez baixa)* — painéis
   organizados têm solidez alta e sombras/objetos soltos formam regiões
   pequenas.

2. **Memória de casos (raciocínio por analogia)** — as imagens já
   confirmadas por humanos ficam guardadas com seus vetores de
   características em `memoria.json`. Uma imagem nova é comparada aos casos
   (k-vizinhos mais próximos, peso 1/(1+distância)) e vota pelo julgamento
   dos casos mais parecidos — o mesmo princípio dos bancos vetoriais usados
   em sistemas de busca semântica, aqui implementado de forma explícita e
   auditável.

3. **Modelo incremental** — uma regressão logística (scikit-learn,
   `SGDClassifier`) consolidada sobre TODA a memória a cada nova confirmação.
   Ela generaliza o que os casos memorizam e funciona mesmo quando nenhum
   caso individual é muito parecido.

**Fusão adaptativa:** o peso da experiência aprendida cresce com o número de
confirmações, `w_aprendizado = n/(n+6)` (teto 0,75), e a heurística cede
espaço na mesma medida. Nos primeiros usos o sistema decide pela heurística;
com o tempo, decide pelo que **aprendeu com os operadores** — é assim que ele
"aprende cada vez mais para evitar possíveis erros".

### Exemplo real medido durante o desenvolvimento

| Cena | 1ª análise (só heurística) | Após correções humanas |
|---|---|---|
| Pilha de papelão/madeira (imagem modelo 1) | 85% desperdício ✔ | 87% ✔ |
| Entulho de gesso/madeira (imagem modelo 2) | 60% desperdício ✔ | 72% ✔ |
| Piso de obra vazio | 24% (limpo) ✔ | 17% ✔ |
| Painéis OSB organizados (caso difícil) | **68% desperdício ✘** | **45% → limpo ✔** |

O caso dos painéis OSB mostra o aprendizado: a heurística erra (painel é
marrom e texturizado, como entulho), o operador corrige, e memória + modelo
passam a anular o erro — o veredito muda de 68–91% para 45%.

---

## 3. Técnicas de visão computacional (mapa com as aulas)

| Técnica | Onde é usada | Aula |
|---|---|---|
| Redimensionamento, conversões de espaço de cor | `visao.py: carregar_imagem` | Aula 02 |
| Manipulação matricial (reshape por blocos, máscaras booleanas) | `visao.py: _estatisticas_por_bloco` | Aula 03 |
| CLAHE, suavização gaussiana (pré-processamento) | `visao.py: analisar_cena` | Aulas 04/05 |
| Detecção de bordas Canny (limiares automáticos pela mediana) e Sobel | `visao.py: analisar_cena` | Aula 06 |
| Morfologia matemática (dilatação, abertura, fechamento) | `visao.py: gerar_mascara_residuos` | Aula 05/06 |
| Análise de contornos (área, solidez, extensão, circularidade) | `visao.py: _analise_contornos` | Aula 06 |
| Supressão de sombras projetadas e vãos de luz estourados (fundo local + guard de textura + filtros por região) | `visao.py: mapa_sombras, _remover_sombras_regioes, _remover_vaos_de_luz` | (extra — robustez em campo) |
| Aprendizado de máquina supervisionado incremental | `memoria.py: ModeloIncremental` | (extra) |

As 18 características extraídas (em ordem fixa, ver `NOMES_CARACTERISTICAS`)
são todas interpretáveis — por isso o sistema consegue mostrar a cadeia de
raciocínio que justifica cada veredito (explicabilidade).

---

## 4. Arquivos gerados em execução

```
dados/
├── memoria.json                 # banco de casos (experiências + vetores) — legível
│                                  # (o modelo incremental é reconstruído deste arquivo
│                                  #  a cada inicialização — sem binário dependente de versão)
├── anotacoes/                   # imagens-relatório com as regiões contornadas
├── capturas/                    # fotos tiradas na hora pelo botão da câmera
└── relatorios/                  # relatórios exportados (Markdown + CSV)
```

- `memoria.json` é gravado atomicamente (`.tmp` + `os.replace`) e, se
  corrompido, é preservado com sufixo `.corrompido-...` enquanto o sistema
  recomeça vazio — tratamento de erros sem perda silenciosa de dados.
- O modelo incremental não é salvo em arquivo binário: ele é reconstruído
  a partir de memoria.json em cada inicialização, então o sistema não quebra
  por diferença de versão do scikit-learn entre computadores.
- **Evolução do pipeline sem perder aprendizado:** cada experiência guarda a
  versão das características com que foi extraída. Quando o pipeline de visão
  evolui (VERSÃO_CARACTERISTICAS muda), o sistema reextrai automaticamente os
  vetores das imagens originais na próxima inicialização, preservando os
  rótulos confirmados por humanos — a memória e o modelo ficam sempre na
  "língua" atual.
- A memória do repositório já vem com as **experiências de calibração** da
  equipe (imagens modelo, recortes negativos e cenas de teste). Em um clone
  novo ela já funciona; use `python main.py resetar` para começar do zero e
  ensinar a partir da sua própria base de imagens.

## 5. Estrutura do código

```
pivc-desperdicio/
├── main.py                  # CLI: analisar, adicionar, status, relatorio, painel, site, resetar
├── webapp.py                # back-end Flask do front-end swipe (API JSON)
├── templates/index.html     # página do modo swipe
├── static/                  # estilo (tema preto + amarelo EPI) e lógica do baralho
├── gui.py                   # interface gráfica Tkinter (confirmação por botões)
├── executar.bat             # atalho Windows (cria/usa o ambiente virtual .venv)
├── pivc_core/
│   ├── visao.py             # pipeline de extração de características (OpenCV)
│   ├── detetor.py           # 3 linhas de pensamento + fusão + raciocínio explicável
│   ├── materiais.py         # classificação do tipo de material por região (assinaturas)
│   └── memoria.py           # banco de casos JSON + modelo incremental
├── imagens_modelo/          # imagens modelo fornecidas no enunciado da APS
├── imagens_teste/           # cenas adicionais usadas nos testes da equipe
├── dados/exemplos/          # recortes de teste (piso limpo, painéis, faixa branca)
└── dados/                   # memória, custos, anotações e relatórios (criados ao rodar)
```

---

## 6. Limitações e trabalhos futuros (para a dissertação)

- **Detecção por região vs. global:** o sistema hoje julga a cena inteira
  (há/não há entulho) e contorna as regiões candidatas; um passo natural é
  classificar cada região isoladamente (ex.: YOLO treinado com as anotações
  que o próprio sistema acumula).
- **Escala da área:** foto 2D não tem escala absoluta — a área do descarte em
  m² depende da largura real da cena informada pelo operador (padrão de 4,0 m,
  indicado na saída). Objeto de referência na cena ou captura padronizada
  (altura/ângulo fixos) reduziria esse erro.
- **Condições de captura:** iluminação forte, sombras duras e ângulos muito
  oblíquos alteram os indicadores; o aprendizado por confirmação mitiga, mas
  padronização de captura (altura, horário) melhora a precisão.
- **Sombras listradas em solo saturado:** sombras periódicas (grade, tapume)
  sobre terra/areia marrom podem se fundir à região de entulho adjacente, pois
  as manchas de sol entre as listras têm textura e cor de material espalhado.
  Sombras lisas (pessoas, equipamentos) sobre concreto são suprimidas; o caso
  listrado em solo saturado é a fronteira da abordagem clássica — resolve-se
  bem com detecção de objetos por deep learning (item abaixo).
- **Escala do modelo:** a memória de casos compara a imagem nova com todas as
  anteriores (O(n)); com milhares de experiências, vale trocar por um banco
  vetorial (ex.: FAISS) sem mudança conceitual.
- **Integração:** o módulo `pivc_core` é independente da interface — pode ser
  acoplado a câmeras em tempo real (vídeo), a um app móvil ou a um serviço
  web, atendendo à extensão "identificação em obras de Goiânia" do tema.

## 7. Requisitos

- Python 3.10+
- `opencv-python`, `numpy`, `scikit-learn`, `pillow`, `flask` (todas gratuitas)
- Funciona offline; nenhuma chave de API é necessária.
