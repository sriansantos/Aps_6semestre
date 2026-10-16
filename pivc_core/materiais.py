"""
materiais.py — Identificação do TIPO de material desperdiçado, por região.

Para cada região candidata a entulho (contorno devolvido pelo pipeline de
visão), mede-se a cor e a textura DENTRO dela e compara-se com assinaturas
visuais por material:

  • papelao    — tons terrosos com textura fibrosa moderada
  • madeira    — tons terrosos com muitos bordos internos (veios, tábuas)
  • gesso      — quase branco, pouco saturado e claro
  • plástico   — cores vibrantes com superfície lisa (lonas, embalagens)
  • alvenaria  — cinza concreto, regiões grandes e irregulares
  • tijolo     — vermelho-cerâmico saturado (tijolos, telhas, lajotas)
  • metal      — cinza-brilhante especular e liso (tubos, perfis, chapas)
  • outro      — nenhuma assinatura supera o limiar de confiança

CADASTRO DE NOVOS MATERIAIS — para registrar um material novo basta tocar em
5 pontos deste arquivo:
  1. tupla MATERIAIS (abaixo);
  2. nome bonito em NOME_MATERIAIS;
  3. cor de anotação em CORES_MATERIAIS (BGR) — usada na imagem-relatório e
     nos pontos de cor do cartão no front-end web;
  4. um indicador de cor NOVO em _medidas_regiao (uma fração de pixels na
     faixa de matiz/saturação/brilho que caracteriza o material);
  5. a assinatura ponderada em _pontuar (pesos somando 1) + tecla em
     CHAVES_CLI, se quiser confirmá-lo pelo terminal.
Como no resto do sistema, o operador confirma/corrige o material na hora do
feedback e a memória de casos (detetor.py) usa essas confirmações para votar
pelo material de imagens parecidas.
"""

import cv2
import numpy as np

MATERIAIS = ("papelao", "madeira", "gesso", "plastico", "alvenaria", "tijolo", "metal", "outro")

NOME_MATERIAIS = {
    "papelao": "Papelão",
    "madeira": "Madeira",
    "gesso": "Gesso/Drywall",
    "plastico": "Plástico/Lonas",
    "alvenaria": "Entulho de alvenaria",
    "tijolo": "Tijolo/Cerâmica",
    "metal": "Metal/Tubos",
    "outro": "Outro material",
}

# Cores (BGR) usadas na imagem-relatório anotada, uma por material.
CORES_MATERIAIS = {
    "papelao": (42, 105, 168),     # marrom terroso
    "madeira": (0, 140, 255),      # laranja (madeira mais saturada)
    "gesso": (0, 235, 255),        # amarelo claro (visível sobre gesso branco)
    "plastico": (255, 0, 255),     # magenta
    "alvenaria": (255, 160, 0),    # azul claro (cinza some no concreto)
    "tijolo": (60, 60, 225),       # vermelho-cerâmico
    "metal": (200, 150, 80),       # azul-aço
    "outro": (200, 200, 200),      # cinza
}

# Chaves da pergunta de material no terminal: [g]esso, [p]apelão, [m]adeira,
# [l]onas/plástico, [a]lvenaria, [t]ijolo, m[e]tal, [o]utro.
CHAVES_CLI = {
    "g": "gesso",
    "p": "papelao",
    "m": "madeira",
    "l": "plastico",
    "a": "alvenaria",
    "t": "tijolo",
    "e": "metal",
    "o": "outro",
}

# Abaixo desta pontuação nenhuma assinatura é forte o bastante — vira "outro".
LIMIAR_CONF_MATERIAL = 0.35


def _ramp(x, inicio, fim):
    """Rampa linear de 0 a 1 entre 'inicio' e 'fim' (satura fora da faixa)."""
    if fim <= inicio:
        return 1.0 if x >= fim else 0.0
    return float(min(1.0, max(0.0, (x - inicio) / (fim - inicio))))


def _medidas_regiao(hsv, edges, contorno):
    """Mede cor e textura DENTRO de um contorno.

    As frações de cor separam o que a calibração em obra mostrou ser
    discriminante:
      • fracao_terrosa  — marrom SATURADO (S>=70): madeira nova, compensado
      • fracao_marrom   — marrom EMPOEIRADO (S 40-70): papelão, caixas usadas
      • fracao_clara    — claro/branco (V>=170, S<70): placas de gesso/drywall
      • fracao_cinza_media — cinza MÉDIO (S<45, V 80-170): concreto/alvenaria
        (o piso de concreto ao redor da pilha entra aqui; por isso a
        assinatura de alvenaria exige cinza DOMINANTE, não presente)
      • fracao_vibrante — S>120: lonas, plásticos
    A textura é medida no interior ERODIDO (sem a banda junto ao contorno).
    """
    regiao = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(regiao, [contorno], -1, 255, -1)
    dentro = regiao > 0
    nucleo = cv2.erode(regiao, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))) > 0
    if not nucleo.any():
        nucleo = dentro

    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    total = max(int(dentro.sum()), 1)

    coloridos = dentro & (s > 40)
    sat_media = float(s[dentro].mean())
    brilho_medio = float(v[dentro].mean())
    matiz_media = float(h[coloridos].mean() / 179.0) if coloridos.any() else 0.0

    fracao = lambda mascara: float((mascara & dentro).sum() / total)
    return {
        "sat_media": sat_media,
        "brilho_medio": brilho_medio,
        "matiz_media": matiz_media,
        "fracao_terrosa": fracao((h >= 5) & (h <= 35) & (s >= 70) & (v >= 60)),
        "fracao_marrom": fracao((h >= 5) & (h <= 35) & (s >= 40) & (s < 70) & (v >= 60)),
        "fracao_clara": fracao((v >= 170) & (s < 70)),
        "fracao_cinza_media": fracao((s < 45) & (v > 80) & (v < 170)),
        "fracao_vibrante": fracao(s > 120),
        # --- materiais cadastrados depois (ver docstring do módulo) ---
        "fracao_vermelha": fracao(((h <= 12) | (h >= 170)) & (s >= 90) & (v >= 50)),
        "fracao_metal": fracao((v >= 200) & (s < 60)),
        "densidade_bordas_interna": float(((edges > 0) & nucleo).sum() / max(int(nucleo.sum()), 1)),
    }


def _pontuar(metadados, m):
    """Assinaturas visuais: soma ponderada de indicadores (pesos somam 1).

    Papelão x madeira se diferenciam pela saturação e pela textura interna:
    papelão é marrom empoeirado de saturação moderada; madeira (tábuas, veios,
    compensado) tem marrom saturado e muitos bordos internos. Gesso é claro e
    pouco saturado; alvenaria exige cinza médio DOMINANTE (o piso de concreto
    do entorno não pode bastar para rotular a região como entulho de obra
    civil); plástico é vibrante e liso, e não é marrom.
    """
    marrom_total = m["fracao_terrosa"] + m["fracao_marrom"]
    return {
        "papelao": (
            0.45 * _ramp(marrom_total, 0.25, 0.55)
            + 0.25 * _ramp(m["densidade_bordas_interna"], 0.04, 0.13)
            + 0.15 * (1.0 - _ramp(m["sat_media"], 100, 150))
            + 0.15 * (1.0 - _ramp(m["fracao_cinza_media"], 0.25, 0.50))
        ),
        "madeira": (
            0.35 * _ramp(m["fracao_terrosa"], 0.12, 0.40)
            + 0.40 * _ramp(m["densidade_bordas_interna"], 0.10, 0.22)
            + 0.25 * _ramp(m["sat_media"], 55, 110)
        ),
        "gesso": (
            0.55 * _ramp(m["fracao_clara"], 0.15, 0.40)
            + 0.25 * (1.0 - _ramp(m["sat_media"], 25, 80))
            + 0.20 * (1.0 - _ramp(m["fracao_metal"], 0.30, 0.70))
        ),
        "plastico": (
            # vibrante FORA da faixa vermelho-cerâmica (que é do tijolo)
            0.60 * _ramp(max(0.0, m["fracao_vibrante"] - m["fracao_vermelha"]), 0.10, 0.40)
            + 0.25 * (1.0 - _ramp(m["densidade_bordas_interna"], 0.03, 0.12))
            + 0.15 * (1.0 - _ramp(marrom_total, 0.30, 0.60))
        ),
        "alvenaria": (
            0.60 * _ramp(m["fracao_cinza_media"], 0.25, 0.55)
            + 0.20 * _ramp(metadados["irregularidade"], 1.5, 4.0)
            + 0.20 * (1.0 - _ramp(m["sat_media"], 45, 90))
        ),
        # Tijolo/cerâmica: vermelho-cerâmico dominante. O termo de matiz
        # penaliza o marrom do papelão (matiz 0.08-0.25) — tijolo é matiz
        # ~0.00-0.07 com saturação alta.
        "tijolo": (
            0.60 * _ramp(m["fracao_vermelha"], 0.15, 0.45)
            + 0.20 * _ramp(m["sat_media"], 90, 140)
            + 0.20 * (1.0 - _ramp(m["matiz_media"], 0.10, 0.25))
        ),
        # Metal/tubos: brilho especular (muito claro, quase sem cor) em
        # superfície lisa. Diferencia do gesso pelo teto de brilho (V>=200)
        # e pela textura interna baixa (chapa lisa x gesso quebrado).
        "metal": (
            0.50 * _ramp(m["fracao_metal"], 0.20, 0.55)
            + 0.25 * _ramp(m["brilho_medio"], 170, 220)
            + 0.25 * (1.0 - _ramp(m["densidade_bordas_interna"], 0.03, 0.10))
        ),
    }


def classificar_regioes(img, contornos):
    """Classifica o material de cada região candidata a entulho.

    Retorna (materiais_por_regiao, distribuicao) — a lista alinhada com
    `contornos` (cada dicionário ganha 'material' e 'material_pontuacao') e a
    distribuição de fração de CENA por material (soma das área_frac). Essas
    frações, multiplicadas pela área real da cena, dão a área do descarte
    em m² (detetor.py).
    """
    materiais = []
    distribuicao = {}
    if not contornos:
        return materiais, distribuicao

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mediana = float(np.median(gray))
    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        max(0, int(0.66 * mediana)) or 20,
        min(255, int(1.33 * mediana)) or 60,
    )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for d in contornos:
        medidas = _medidas_regiao(hsv, edges, d["contorno"])
        pontuacoes = _pontuar(d, medidas)
        material, pontuacao = max(pontuacoes.items(), key=lambda kv: kv[1])
        if pontuacao < LIMIAR_CONF_MATERIAL:
            material, pontuacao = "outro", pontuacao
        d["material"] = material
        d["material_pontuacao"] = float(pontuacao)
        materiais.append(material)
        distribuicao[material] = distribuicao.get(material, 0.0) + d["area_frac"]

    return materiais, distribuicao
