"""
visao.py — Extração de características visuais para o sistema PIVC.

Técnicas de Visão Computacional aplicadas (alinhadas às Aulas 02 a 06):
  - Pré-processamento: redimensionamento, CLAHE (equalização adaptativa de
    contraste) e suavização gaussiana (Aulas 04/05).
  - Manipulação matricial: estatísticas calculadas por blocos via reshape
    de matrizes NumPy (Aula 03).
  - Detecção de bordas: operador Sobel (gradiente) e Canny com limiares
    automáticos derivados da mediana da imagem (Aula 06).
  - Segmentação: morfologia matemática (dilatação/abertura/fechamento) e
    análise de contornos (área, solidez, extensão, circularidade).

Cada característica é um número interpretável — o sistema consegue explicar
"por que" chegou a um veredito (cadeia de raciocínio auditável).
"""

import os
import unicodedata

import cv2
import numpy as np

# Versão do vetor de características: se mudar a lista abaixo OU o modo de
# cálculo que as origina, incremente. A memória guarda esta versão junto de
# cada experiência e recalcula os vetores automaticamente ao evoluir.
VERSAO_CARACTERISTICAS = 2

# Ordem fixa das características (o vetor armazenado na memória segue exatamente esta ordem).
NOMES_CARACTERISTICAS = [
    "densidade_bordas",
    "entropia_textura",
    "fracao_blocos_texturizados",
    "media_matiz",
    "media_saturacao",
    "media_brilho",
    "desvio_brilho",
    "fracao_tons_terrosos",
    "fracao_cinza_concreto",
    "fracao_branco_gesso",
    "fracao_cores_vibrantes",
    "gradiente_medio",
    "gradiente_desvio",
    "maior_contorno_area",
    "solidade_contornos",
    "extensao_contornos",
    "n_contornos_grandes",
    "irregularidade_contornos",
]

LARGURA_MAXIMA = 640
BLOCO = 16  # grade de blocos 16x16 para análise de textura local


def carregar_imagem(caminho):
    """Carrega uma imagem do disco de forma robusta a caminhos com acentos
    (Windows) e redimensiona preservando a proporção.

    Retorna a imagem no formato BGR do OpenCV ou levanta ValueError.
    """
    if not os.path.isfile(caminho):
        raise ValueError(f"Arquivo não encontrado: {caminho}")
    # np.fromfile + imdecode lê arquivos com caracteres especiais no caminho,
    # coisa que o cv2.imread falha em fazer no Windows.
    dados = np.fromfile(caminho, dtype=np.uint8)
    if dados.size == 0:
        raise ValueError(f"Arquivo vazio ou ilegível: {caminho}")
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Formato de imagem não suportado ou arquivo corrompido: {caminho}")

    altura, largura = img.shape[:2]
    if largura > LARGURA_MAXIMA:
        escala = LARGURA_MAXIMA / largura
        nova = (LARGURA_MAXIMA, max(1, int(altura * escala)))
        img = cv2.resize(img, nova, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:  # imagem sem canal de cor
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _entropia_normalizada(gray):
    """Entropia de Shannon do histograma de intensidades, normalizada em [0,1].
    Cenas com muitos materiais diferentes (entulho) têm entropia alta; uma
    parede ou piso uniforme tem entropia baixa."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / 8.0)


def _estatisticas_por_bloco(gray):
    """Manipulação matricial (Aula 03): a imagem é recortada em blocos
    16x16 via reshape e calcula-se o desvio-padrão de cada bloco.
    Blocos com desvio alto indicam textura (materiais espalhados)."""
    h, w = gray.shape
    nb_h, nb_w = h // BLOCO, w // BLOCO
    if nb_h < 1 or nb_w < 1:
        return 0.0, 0.0
    recorte = gray[: nb_h * BLOCO, : nb_w * BLOCO]
    blocos = (
        recorte.reshape(nb_h, BLOCO, nb_w, BLOCO)
        .swapaxes(1, 2)
        .reshape(nb_h * nb_w, BLOCO * BLOCO)
        .astype(np.float32)
    )
    desvios = blocos.std(axis=1)
    texturizados = float((desvios > 18.0).mean())
    return texturizados, float(desvios.mean())


def mapa_sombras(hsv):
    """Mapa por pixel de sombra projetada (ex.: sombra de pessoas/equipamentos
    no concreto) + estimativa do fundo iluminado.

    Sombra preserva o matiz do solo e apenas REDUZ a intensidade: um pixel é
    sombra quando é bem mais escuro que o fundo local e quase não tem
    saturação (cinza). Além disso, só marca quando o ENTORNO é homogêneo
    (desvio-padrão local baixo): sombra de verdade cai sobre piso liso, mas
    as partes escuras de uma pilha de entulho vivem num entorno texturizado —
    o guardo de textura impede que o próprio entulho (vãos entre peças de
    gesso, sombras internas da pilha) seja confundido com sombra projetada.

    Retorna (sombras, fundo_local) — o fundo é a reconstrução do brilho
    "como seria sem sombra", usada para segmentar sem os degradês de sombra.
    """
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    fundo_local = cv2.GaussianBlur(v, (0, 0), 21)
    m1 = cv2.GaussianBlur(v, (0, 0), 15)
    m2 = cv2.GaussianBlur(v * v, (0, 0), 15)
    desvio_local = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    mais_escuro = v < 0.75 * fundo_local
    sem_cor = s < 65
    entorno_homogeneo = desvio_local < 25
    sombras = ((mais_escuro & sem_cor & entorno_homogeneo) * 255).astype(np.uint8)
    return sombras, fundo_local


def gerar_mascara_residuos(edges, sombras=None):
    """Constrói uma máscara binária das regiões 'agitadas' da imagem:
    dilata as bordas detectadas, fecha pequenos buracos e remove ruído.
    Os pixels marcados como sombra (mapa_sombras) são removidos antes e
    depois da morfologia, para que sombras não virem nem regiões nem
    pontes que colam o entulho à sombra vizinha."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordas = edges.copy()
    if sombras is not None:
        bordas[sombras > 0] = 0
    mascara = cv2.dilate(bordas, kernel, iterations=1)
    # Fechamento 15x15: solda os vãos entre peças de uma MESMA pilha (gesso
    # quebrado, telhas) para a pilha virar uma região única; regiões distantes
    # entre si continuam separadas — e sombras já foram achatadas antes da
    # morfologia, então não há ponte falsa para reconstruir.
    kernel_fech = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_fech, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    if sombras is not None:
        # Cresce a zona de sombra antes de subtrair: corta as pontes finas que
        # a morfologia poderia construir através de sombras/listras, sem
        # alcançar o interior das pilhas (protegido pelo guard de textura).
        sombras_crescidas = cv2.dilate(sombras, kernel, iterations=3)
        mascara[sombras_crescidas > 0] = 0
    return mascara


def _analise_contornos(mascara):
    """Analisa os contornos das regiões candidatas a entulho.
    Entulho tende a formar regiões grandes e IRREGULARES (solidez baixa,
    circularidade baixa). Objetos organizados tendem a ser compactos."""
    res = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = res[0] if len(res) == 2 else res[1]
    area_img = mascara.shape[0] * mascara.shape[1]

    validos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 0.005 * area_img:  # ignora ruído minúsculo
            continue
        perimetro = cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        area_hull = cv2.contourArea(hull) or 1.0
        solidez = area / area_hull
        extensao = area / (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] or 1.0)
        circularidade = (4.0 * np.pi * area / (perimetro ** 2)) if perimetro > 0 else 0.0
        irregularidade = min(6.0, 1.0 / circularidade) if circularidade > 0 else 6.0
        validos.append(
            {
                "contorno": c,
                "area_frac": area / area_img,
                "solidez": solidez,
                "extensao": extensao,
                "irregularidade": irregularidade,
            }
        )

    validos.sort(key=lambda d: d["area_frac"], reverse=True)
    return validos


def _resumo_contornos(validos):
    """Estatísticas dos contornos usadas como características do vetor."""
    if not validos:
        return {
            "maior_contorno_area": 0.0,
            "solidade_contornos": 1.0,
            "extensao_contornos": 0.0,
            "n_contornos_grandes": 0.0,
            "irregularidade_contornos": 0.0,
        }
    top = validos[:5]
    return {
        "maior_contorno_area": top[0]["area_frac"],
        "solidade_contornos": float(np.mean([d["solidez"] for d in top])),
        "extensao_contornos": float(np.mean([d["extensao"] for d in top])),
        "n_contornos_grandes": float(len(validos)),
        "irregularidade_contornos": float(np.mean([d["irregularidade"] for d in top])),
    }


def _remover_sombras_regioes(validos, hsv, edges):
    """Filtro final por região: descarta contornos com perfil de sombra
    projetada que sobreviveram à remoção por pixel (ex.: bordas de sombra
    que ficaram do lado iluminado). Uma região é sombra quando passa nos
    TRÊS testes: mais escura que o anel ao redor, quase sem saturação e
    com interior liso (poucas bordas internas — entulho é texturizado).
    A textura interna é medida no interior ERODIDO (sem a banda junto ao
    contorno), para região grande não diluir a textura e passar por lisa."""
    if not validos:
        return validos
    altura, largura = hsv.shape[:2]
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mantidas = []
    for d in validos:
        regiao = np.zeros((altura, largura), np.uint8)
        cv2.drawContours(regiao, [d["contorno"]], -1, 255, -1)
        dentro = regiao > 0
        nucleo = cv2.erode(regiao, kernel_erode) > 0
        interior_textura = nucleo if nucleo.any() else dentro
        anel = (cv2.dilate(regiao, kernel) > 0) & ~dentro
        if not anel.any():
            mantidas.append(d)
            continue
        mais_escura = v[dentro].mean() < 0.80 * v[anel].mean()
        sem_cor = s[dentro].mean() < 75.0
        lisa = ((edges > 0) & interior_textura).sum() / max(interior_textura.sum(), 1) < 0.06
        if mais_escura and sem_cor and lisa:
            continue
        mantidas.append(d)
    return mantidas


def _remover_vaos_de_luz(validos, hsv):
    """Descarta regiões de 'vão de luz estourado' — janelas, portas abertas
    para o exterior, claraboias: interior quase branco (V p95 >= 242) e pouco
    saturado, muito mais claro que o entorno. Entulho claro (gesso novo) fica
    abaixo desse teto de brilho ou tem cor/terra; e um vão de luz não é
    material — é ausência de material."""
    if not validos:
        return validos
    v = hsv[..., 2]
    s = hsv[..., 1].astype(np.float32)
    mantidas = []
    for d in validos:
        regiao = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(regiao, [d["contorno"]], -1, 255, -1)
        dentro = regiao > 0
        if np.percentile(v[dentro], 95) >= 242 and s[dentro].mean() < 55:
            continue
        mantidas.append(d)
    return mantidas


def analisar_cena(img):
    """Pipeline completo em uma única passada.

    Retorna (caracteristicas, contornos) — o dicionário de características
    segue NOMES_CARACTERISTICAS e 'contornos' são as regiões candidatas a
    entulho, já com metadados (área, solidez, irregularidade) para anotação.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Pré-processamento (Aulas 04/05) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    suavizada = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # --- Detecção de bordas (Aula 06): Canny com limiares automáticos ---
    mediana = float(np.median(suavizada))
    lim_inf = max(0, int(0.66 * mediana))
    lim_sup = min(255, int(1.33 * mediana))
    edges = cv2.Canny(suavizada, lim_inf or 20, lim_sup or 60)
    densidade_bordas = float((edges > 0).mean())

    # Gradiente de Sobel (Aula 06) — intensidade média das transições
    sobx = cv2.Sobel(suavizada, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(suavizada, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobx, soby)

    # --- Textura ---
    entropia = _entropia_normalizada(gray_eq)
    frac_blocos_text, _ = _estatisticas_por_bloco(gray_eq)

    # --- Cor (HSV) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    total = h.size
    coloridos = s > 40  # ignora pixels quase pretos/brancos na média de matiz
    media_matiz = float(h[coloridos].mean() / 179.0) if coloridos.any() else 0.0
    media_saturacao = float(s.mean())
    media_brilho = float(v.mean())
    desvio_brilho = float(v.std())

    # Terrosos exigem saturação alta: piso de concreto empoeirado tem tons
    # parecidos, mas com saturação baixa (o corte em S>=70 evita esse falso positivo).
    tons_terrosos = ((h >= 5) & (h <= 35) & (s >= 70) & (v >= 60)).sum() / total
    cinza_concreto = ((s < 60) & (v > 60) & (v < 220)).sum() / total
    branco_gesso = ((v > 200) & (s < 60)).sum() / total
    vibrantes = (s > 120).sum() / total

    # --- Segmentação e contornos (com supressão de sombras) ---
    # As bordas da segmentação são calculadas sobre uma reconstrução "sem
    # sombra" do brilho: pixels de sombra recebem o fundo iluminado local.
    # Assim os degradês/listras de sombras projetadas não geram contornos,
    # enquanto materiais com cor ou textura própria continuam visíveis.
    sombras, fundo_local = mapa_sombras(hsv)
    brilho_sem_sombra = np.where(sombras > 0, fundo_local, hsv[..., 2])
    gray_regiao = np.clip(brilho_sem_sombra, 0, 255).astype(np.uint8)
    mediana_regiao = float(np.median(gray_regiao))
    edges_regiao = cv2.Canny(
        cv2.GaussianBlur(gray_regiao, (5, 5), 0),
        max(0, int(0.66 * mediana_regiao)) or 20,
        min(255, int(1.33 * mediana_regiao)) or 60,
    )
    mascara = gerar_mascara_residuos(edges_regiao, sombras)
    contornos = _analise_contornos(mascara)
    contornos = _remover_sombras_regioes(contornos, hsv, edges)
    contornos = _remover_vaos_de_luz(contornos, hsv)
    resumo_contornos = _resumo_contornos(contornos)

    caracteristicas = {
        "densidade_bordas": densidade_bordas,
        "entropia_textura": entropia,
        "fracao_blocos_texturizados": frac_blocos_text,
        "media_matiz": media_matiz,
        "media_saturacao": media_saturacao,
        "media_brilho": media_brilho,
        "desvio_brilho": desvio_brilho,
        "fracao_tons_terrosos": float(tons_terrosos),
        "fracao_cinza_concreto": float(cinza_concreto),
        "fracao_branco_gesso": float(branco_gesso),
        "fracao_cores_vibrantes": float(vibrantes),
        "gradiente_medio": float(magnitude.mean()),
        "gradiente_desvio": float(magnitude.std()),
    }
    caracteristicas.update(resumo_contornos)
    return caracteristicas, contornos


def extrair_caracteristicas(img):
    """Versão resumida do pipeline: devolve apenas o dicionário de características."""
    return analisar_cena(img)[0]


def vetor_de(caracteristicas):
    """Converte o dicionário de características no vetor numérico fixo
    (mesma ordem de NOMES_CARACTERISTICAS) usado pela memória e pelo modelo."""
    return np.array([float(caracteristicas[n]) for n in NOMES_CARACTERISTICAS], dtype=np.float64)


def anotar_imagem(img, contornos, veredito_desperdicio, probabilidade,
                  cores_regioes=None, legenda=None):
    """Gera a imagem 'relatório': contorna as regiões candidatas a entulho
    e carimba o veredito no canto superior.

    `cores_regioes` (lista de tuplas BGR alinhada com 'contornos') permite uma
    cor por tipo de material; sem ela, vale a regra antiga: vermelho =
    desperdício detectado, verde = cena sem desperdício. `legenda` é uma
    lista de (texto, cor) desenhada abaixo do carimbo (composição do
    desperdício por material)."""
    anotada = img.copy()
    cor_padrao = (0, 0, 230) if veredito_desperdicio else (40, 180, 40)
    for i, d in enumerate(contornos):
        cor = cores_regioes[i] if cores_regioes else cor_padrao
        cv2.drawContours(anotada, [d["contorno"]], -1, cor, 2)
        x, y, w, h = cv2.boundingRect(d["contorno"])
        cv2.putText(
            anotada, f"{int(d['area_frac'] * 100)}%", (max(0, x), max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA,
        )

    rotulo = "DESPERDICIO DE MATERIAIS" if veredito_desperdicio else "SEM DESPERDICIO"
    texto = f"{rotulo}: {int(round(probabilidade * 100))}%"
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(anotada, (8, 8), (16 + tw, 20 + th + base), (20, 20, 20), -1)
    cv2.putText(anotada, texto, (12, 18 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_padrao, 2, cv2.LINE_AA)

    if legenda:
        # Faixa de fundo para a legenda ficar legível sobre qualquer cena.
        y0 = 20 + th + base + 12
        altura_item = 20
        largura_legenda = max(
            cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for t, _ in legenda
        )
        cv2.rectangle(
            anotada, (8, y0), (34 + largura_legenda, y0 + altura_item * len(legenda) + 6),
            (20, 20, 20), -1,
        )
        for i, (texto_item, cor_item) in enumerate(legenda):
            y = y0 + 4 + i * altura_item
            cv2.rectangle(anotada, (14, y), (28, y + 12), cor_item, -1)
            cv2.putText(
                anotada, texto_item, (34, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
            )
    return anotada


def salvar_anotacao(img, caminho_saida):
    """Salva a imagem anotada (robusto a caminhos com acentos no Windows)."""
    os.makedirs(os.path.dirname(caminho_saida) or ".", exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"Falha ao codificar imagem para {caminho_saida}")
    buf.tofile(caminho_saida)


def nome_base_sem_acento(caminho):
    """Nome do arquivo sem extensão, sem acentos — evita problemas ao
    gravar as imagens anotadas em sistemas com codificação diferente."""
    base = os.path.splitext(os.path.basename(caminho))[0]
    return unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
