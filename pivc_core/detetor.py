"""
detetor.py — O "cérebro" do sistema: julga se há desperdício de materiais.

O veredito é formado por uma FUSÃO ADAPTATIVA de três linhas de pensamento,
cada uma votando com uma probabilidade entre 0 e 1:

  1. Heurística de engenharia (conhecimento a priori, funciona desde o 1º uso):
     soma ponderada de indicadores visuais de entulho — bordas, textura,
     tons terrosos (papelão/madeira/terra), contornos grandes e irregulares.

  2. Memória de casos (raciocínio por analogia): compara a imagem atual com
     os casos já confirmados por humanos na memória e vota pelo julgamento
     dos casos mais parecidos (k-vizinhos mais próximos ponderados).

  3. Modelo incremental: regressão logística treinada sobre a memória —
     generaliza os padrões confirmados pelos operadores.

Nos primeiros usos a heurística domina. A cada confirmação humana, o peso da
experiência aprendida cresce (curva n/(n+6)) — o sistema "aprende cada vez
mais" e guarda tudo em disco entre as execuções.

Cada análise devolve também sua CADEIA DE RACIOCÍNIO: uma lista de passos
com os números que justificam o veredito (sistema auditável/explicável).
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .materiais import (
    classificar_regioes,
    CORES_MATERIAIS,
    NOME_MATERIAIS,
)
from .memoria import MemoriaExperiencias, ModeloIncremental, ROTULO_DESPERDICIO
from .visao import (
    analisar_cena,
    anotar_imagem,
    carregar_imagem,
    nome_base_sem_acento,
    salvar_anotacao,
    vetor_de,
    VERSAO_CARACTERISTICAS,
)

# Largura real presumida da cena (m) quando o operador não informa --largura-m.
# Foto 2D não tem escala absoluta; este padrão é assumido e está indicado na saída.
LARGURA_PADRAO_M = 4.0


def _num(valor, casas=1):
    """Número com vírgula decimal (pt-BR) para a saída textual."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _ramp(x, inicio, fim):
    """Rampa linear de 0 a 1 entre 'inicio' e 'fim' (satura fora da faixa)."""
    if fim <= inicio:
        return 1.0 if x >= fim else 0.0
    return float(min(1.0, max(0.0, (x - inicio) / (fim - inicio))))


def _heuristica_engenharia(c):
    """Linha de pensamento 1 — probabilidade a priori de desperdício a partir
    dos indicadores visuais. Os pesos foram calibrados em imagens de obra
    (pilhas de entulho x áreas organizadas/limpas) e somam 1.0.

    O indicador mais forte é composto: pilha de entulho forma UMA região
    grande da cena e com contorno irregular (solidez baixa). Painéis e
    sacos organizados formam regiões compactas (solidez alta); sombras e
    objetos isolados no piso formam regiões pequenas.
    """
    regiao_pilha = _ramp(c["maior_contorno_area"], 0.10, 0.45) * (
        1.0 - _ramp(c["solidade_contornos"], 0.60, 0.90)
    )
    indicadores = [
        (0.28, regiao_pilha, "maior região grande e irregular (perfil de pilha de entulho)"),
        (0.14, _ramp(c["fracao_blocos_texturizados"], 0.30, 0.70), "blocos 16x16 texturizados"),
        (0.14, _ramp(c["fracao_tons_terrosos"], 0.05, 0.20), "tons terrosos saturados (papelão, madeira, terra)"),
        (0.10, _ramp(c["densidade_bordas"], 0.03, 0.10), "densidade de bordas (Canny)"),
        (0.10, _ramp(c["entropia_textura"], 0.80, 0.95), "entropia de textura (variedade de materiais)"),
        (0.10, _ramp(c["gradiente_medio"], 30.0, 90.0), "intensidade do gradiente (Sobel)"),
        (0.06, _ramp(c["fracao_cores_vibrantes"], 0.02, 0.15), "cores vibrantes (lonas, plásticos)"),
        (0.08, _ramp(c["irregularidade_contornos"], 1.6, 4.0), "irregularidade média dos contornos"),
    ]
    p = sum(peso * valor for peso, valor, _ in indicadores)
    return float(min(1.0, p)), indicadores


@dataclass
class ResultadoAnalise:
    caminho: str
    probabilidade: float
    veredito_desperdicio: bool
    confianca: str
    raciocinio: list = field(default_factory=list)
    caracteristicas: dict = field(default_factory=dict)
    contornos: list = field(default_factory=list)
    p_heuristica: float = None
    p_memoria: float = None
    p_modelo: float = None
    casos_analogos: list = field(default_factory=list)
    material_dominante: str = None       # chave em materiais.MATERIAIS
    material_confianca: float = 0.0
    material_distribuicao_final: dict = field(default_factory=dict)  # assinaturas + memória, soma 1
    distribuicao_materiais: dict = field(default_factory=dict)  # material → fração da cena
    areas_m2: dict = field(default_factory=dict)  # material → área do descarte em m²
    aglomerados_por_material: dict = field(default_factory=dict)  # material → nº de montes
    materiais_regioes: list = field(default_factory=list)       # alinhado com 'contornos'
    largura_m: float = None              # largura real da cena usada na área
    imagem_largura: int = 0
    imagem_altura: int = 0
    vetor: object = None  # np.ndarray — vetor de características desta análise
    imagem: object = None  # np.ndarray — imagem normalizada usada na análise

    @property
    def rotulo_previsto(self):
        return ROTULO_DESPERDICIO if self.veredito_desperdicio else 0

    @property
    def percentual(self):
        return f"{self.probabilidade * 100:.0f}%"


class DetetorDesperdicio:
    def __init__(self, pasta_dados):
        self.pasta_dados = pasta_dados
        self.pasta_anotacoes = os.path.join(pasta_dados, "anotacoes")
        self.memoria = MemoriaExperiencias(pasta_dados)
        self.modelo = ModeloIncremental()
        self.memoria_recalculada = False
        # O pipeline de visão evoluiu (nova VERSAO_CARACTERISTICAS)? Reextrai
        # os vetores das experiências antigas das imagens originais, preservando
        # os rótulos confirmados por humanos — a memória e o modelo passam a
        # falar a "língua" atual sem perder o aprendizado.
        if self.memoria.total > 0 and self.memoria.versao != VERSAO_CARACTERISTICAS:
            self.memoria_recalculada = self.memoria.atualizar_vetores(self._extrair_vetor)
        # O modelo não tem arquivo binário: é reconstruído da memória (JSON)
        # a cada inicialização — rápido e imune a diferenças de versão do
        # scikit-learn entre computadores.
        if self.memoria.total >= 2:
            X, y = self.memoria.matrizes()
            self.modelo.consolidar(X, y)

    def _extrair_vetor(self, caminho):
        img = carregar_imagem(caminho)
        c, _ = analisar_cena(img)
        return vetor_de(c)

    # ------------------------------------------------------------------ #
    # Análise completa de uma imagem
    # ------------------------------------------------------------------ #
    def analisar(self, caminho_imagem, largura_m=None):
        img = carregar_imagem(caminho_imagem)
        altura, largura = img.shape[:2]
        c, contornos = analisar_cena(img)
        vetor = vetor_de(c)

        p_heur, indicadores = _heuristica_engenharia(c)
        p_mem, casos, nivel_analogia = self._julgamento_por_analogia(vetor)
        p_mod = self.modelo.probabilidade(vetor)
        prob, pesos = self._fundir(p_heur, p_mem, p_mod, nivel_analogia)

        # --- Identificação do tipo de material -------------------------- #
        # 1) assinaturas visuais por região (a priori);
        # 2) voto da memória de casos sobre imagens parecidas com material
        #    confirmado pelo operador; fusão das duas distribuições (o nível
        #    da analogia acima também qualifica o voto de material).
        materiais_regioes, dist_assinaturas = classificar_regioes(img, contornos)
        dist_memoria_material, n_casos_material = self._voto_material_memoria(vetor)
        dist_material, peso_memoria_material = self._fundir_material(
            dist_assinaturas, dist_memoria_material, n_casos_material, nivel_analogia
        )
        material_dominante = None
        material_confianca = 0.0
        if dist_material:
            material_dominante = max(dist_material, key=dist_material.get)
            material_confianca = dist_material[material_dominante]

        # Contagem de aglomerados (montes) de desperdício por material. Cada
        # região de >= 1% da cena é um monte — a unidade que a visão clássica
        # conta com honestidade (peças individuais dentro da pilha se fundem
        # e se sobrepõem; contá-las exigiria detecção de objetos por deep
        # learning — ver trabalhos futuros no README).
        aglomerados = {}
        for d in contornos:
            if d["area_frac"] >= 0.01:
                m = d.get("material", "outro")
                aglomerados[m] = aglomerados.get(m, 0) + 1

        # --- Área do descarte (m²) --------------------------------------- #
        # Escala: o operador informa a largura real da cena (--largura-m ou o
        # painel); sem isso vale LARGURA_PADRAO_M (indicado na saída).
        if largura_m is None:
            largura_m = LARGURA_PADRAO_M
        area_cena = largura_m * largura_m * (altura / largura)
        areas_m2 = {
            m: f * area_cena for m, f in dist_assinaturas.items() if f * area_cena >= 0.01
        }

        veredito = prob >= 0.5
        resultado = ResultadoAnalise(
            caminho=str(caminho_imagem),
            probabilidade=prob,
            veredito_desperdicio=veredito,
            confianca=self._rotulo_confianca(prob),
            caracteristicas=c,
            contornos=[d for d in contornos if d["area_frac"] >= 0.01][:8],
            p_heuristica=p_heur,
            p_memoria=p_mem,
            p_modelo=p_mod,
            casos_analogos=casos,
            material_dominante=material_dominante,
            material_confianca=material_confianca,
            material_distribuicao_final=dist_material,
            # Frações de cena por material MEDIDAS nas regiões (área física) —
            # é esta distribuição que dá a área do descarte. O rótulo do material
            # DOMINANTE (acima) vem da fusão assinaturas + memória.
            distribuicao_materiais=dist_assinaturas,
            materiais_regioes=[d.get("material") for d in contornos
                               if d["area_frac"] >= 0.01][:8],
            areas_m2=areas_m2,
            aglomerados_por_material=aglomerados,
            largura_m=largura_m,
            imagem_largura=largura,
            imagem_altura=altura,
            vetor=vetor,
            imagem=img,
        )
        resultado.raciocinio = self._montar_raciocinio(
            largura, altura, p_heur, indicadores, p_mem, casos, nivel_analogia, p_mod, pesos, prob,
            dist_assinaturas, dist_memoria_material, n_casos_material, dist_material,
            material_dominante, peso_memoria_material, largura_m, aglomerados,
        )
        return resultado

    # ------------------------------------------------------------------ #
    # Linhas de pensamento 2 e 3 + fusão
    # ------------------------------------------------------------------ #
    def _julgamento_por_analogia(self, vetor):
        """Linha de pensamento 2 — memória de casos (k-NN ponderado).
        Os k casos mais próximos votam com peso 1/(1+distância).
        Retorna (probabilidade | None, casos_usados, nível_da_analogia)."""
        if self.memoria.total == 0:
            return None, [], "inexistente"
        X, y = self.memoria.matrizes()
        if self.modelo.treinado:
            Xn = self.modelo.scaler.transform(X)
            v = self.modelo.scaler.transform(np.array([vetor]))[0]
        else:
            escalonador = _EscalonadorProvisorio(X)
            Xn = escalonador.transformar(X)
            v = escalonador.transformar(np.array([vetor]))[0]

        dist = np.linalg.norm(Xn - v, axis=1)
        k = min(5, len(X))
        idx = np.argsort(dist)[:k]
        pesos = 1.0 / (1.0 + dist[idx])
        votos = float((pesos * y[idx]).sum() / pesos.sum())
        casos = [
            {
                "id": int(self.memoria.experiencias[i]["id"]),
                "caminho": self.memoria.experiencias[i]["caminho"],
                "rotulo": int(y[i]),
                "distancia": float(dist[i]),
            }
            for i in idx
        ]
        d_media = float(dist[idx].mean())
        nivel = "forte" if d_media < 3.0 else ("média" if d_media < 6.0 else "fraca")
        return votos, casos, nivel

    def _fundir(self, p_heur, p_mem, p_mod, nivel_analogia):
        """Fusão adaptativa das três linhas de pensamento.

        O peso da experiência aprendida cresce com o número de confirmações
        humanas (n/(n+6), teto 0.75); a heurística cede espaço aos poucos.
        Analogia 'fraca' (nenhum caso realmente parecido na memória) reduz o
        peso relativo da memória, que é redistribuído para o modelo — os
        pesos somam sempre 1.0.
        """
        n = self.memoria.total
        w_aprend = min(0.75, n / (n + 6.0))
        w_heur = 1.0 - w_aprend

        sinais = []  # (nome, probabilidade, peso_relativo)
        if p_mod is not None:
            sinais.append(("modelo", p_mod, 1.0))
        if p_mem is not None:
            sinais.append(("memoria", p_mem, 1.0 if nivel_analogia != "fraca" else 0.5))
        if not sinais:
            return p_heur, {"heurística": 1.0}

        soma = sum(s[2] for s in sinais)
        pesos = {"heurística": w_heur}
        p_final = w_heur * p_heur
        for nome, prob_sinal, peso_rel in sinais:
            w = w_aprend * (peso_rel / soma)
            pesos[nome] = pesos.get(nome, 0.0) + w
            p_final += w * prob_sinal
        return float(min(1.0, max(0.0, p_final))), pesos

    # ------------------------------------------------------------------ #
    # Identificação de material: voto da memória + fusão
    # ------------------------------------------------------------------ #
    def _voto_material_memoria(self, vetor):
        """Voto de material por analogia: as experiências com material
        confirmado pelo operador mais parecidas com a imagem atual votam
        (k-NN ponderado, mesmo princípio da linha de pensamento 2).
        Retorna (distribuição material→peso | None, n_casos_com_material)."""
        casos = [e for e in self.memoria.experiencias if e.get("material")]
        if not casos:
            return None, 0
        X = np.array([e["vetor"] for e in casos], dtype=np.float64)
        materiais = [e["material"] for e in casos]
        if self.modelo.treinado:
            Xn = self.modelo.scaler.transform(X)
            v = self.modelo.scaler.transform(np.array([vetor]))[0]
        else:
            escalonador = _EscalonadorProvisorio(X)
            Xn = escalonador.transformar(X)
            v = escalonador.transformar(np.array([vetor]))[0]
        dist = np.linalg.norm(Xn - v, axis=1)
        k = min(5, len(X))
        idx = np.argsort(dist)[:k]
        pesos = 1.0 / (1.0 + dist[idx])
        distribuicao = {}
        for i, peso in zip(idx, pesos):
            distribuicao[materiais[i]] = distribuicao.get(materiais[i], 0.0) + float(peso)
        total = sum(distribuicao.values()) or 1.0
        return {m: p / total for m, p in distribuicao.items()}, len(casos)

    @staticmethod
    def _fundir_material(dist_assinaturas, dist_memoria, n_casos_material, nivel_analogia):
        """Fusão das assinaturas visuais com o voto da memória de material.

        As frações de área medidas nas regiões são normalizadas para uma
        distribuição (soma 1) antes de misturar com o voto da memória —
        área bruta da cena e distribuição de materiais são grandezas
        diferentes. O peso da memória cresce com o número de confirmações
        (n/(n+4), teto 0.6) e cai quando a analogia com os casos é fraca
        (forte ×1.0, média ×0.7, fraca ×0.3). Sem região detectada, não há
        material a rotular."""
        if not dist_assinaturas:
            return {}, 0.0
        soma = sum(dist_assinaturas.values()) or 1.0
        dist_normalizada = {m: f / soma for m, f in dist_assinaturas.items()}
        if dist_memoria is None or n_casos_material == 0:
            return dist_normalizada, 0.0
        fator_analogia = {"forte": 1.0, "média": 0.7, "fraca": 0.3}.get(nivel_analogia, 1.0)
        w_mem = min(0.6, n_casos_material / (n_casos_material + 4.0)) * fator_analogia
        chaves = set(dist_normalizada) | set(dist_memoria)
        dist_final = {
            m: (1.0 - w_mem) * dist_normalizada.get(m, 0.0)
            + w_mem * dist_memoria.get(m, 0.0)
            for m in chaves
        }
        return dist_final, w_mem

    # ------------------------------------------------------------------ #
    # Explicabilidade (cadeia de raciocínio)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rotulo_confianca(p):
        if p >= 0.80 or p <= 0.20:
            return "alta"
        if p >= 0.65 or p <= 0.35:
            return "média"
        return "baixa"

    def _montar_raciocinio(
        self, largura, altura, p_heur, indicadores, p_mem, casos, nivel, p_mod, pesos, p_final,
        dist_assinaturas, dist_memoria, n_casos_material, dist_material, material_dominante,
        peso_memoria_material, largura_m, aglomerados,
    ):
        passos = []
        passos.append(
            f"Passo 1 — Pré-processamento: imagem normalizada para {largura}x{altura} px, "
            f"contraste equalizado (CLAHE) e suavização gaussiana 5x5."
        )
        detalhes = ", ".join(f"{nome} {valor:.2f}" for _, valor, nome in indicadores)
        passos.append(f"Passo 2 — Indicadores visuais (Canny, Sobel, HSV, contornos): {detalhes}.")
        passos.append(f"Passo 3 — Heurística de engenharia (conhecimento a priori): p = {p_heur:.2f}.")

        if p_mem is None:
            passos.append(
                "Passo 4 — Memória de casos: ainda vazia; sem analogias disponíveis. "
                "As primeiras confirmações humanas vão alimentá-la."
            )
        else:
            n_analogos = len(casos)
            rot = sum(1 for c in casos if c["rotulo"] == ROTULO_DESPERDICIO)
            d_med = sum(c["distancia"] for c in casos) / n_analogos
            passos.append(
                f"Passo 4 — Memória de casos (analogia {nivel}): {n_analogos} casos semelhantes "
                f"consultados (distância média {d_med:.2f}); {rot} de {n_analogos} registravam "
                f"desperdício → p = {p_mem:.2f}."
            )
        if p_mod is None:
            passos.append(
                "Passo 5 — Modelo incremental: ainda não treinado (precisa de "
                "confirmações das duas classes para aprender)."
            )
        else:
            passos.append(
                f"Passo 5 — Modelo incremental (regressão logística, {self.modelo.n_treino} "
                f"experiências no treino): p = {p_mod:.2f}."
            )
        pesos_txt = " + ".join(f"{nome} {w:.2f}" for nome, w in pesos.items())
        passos.append(
            f"Passo 6 — Fusão adaptativa (memória com {self.memoria.total} experiências; "
            f"pesos: {pesos_txt}) → p final = {p_final:.2f}."
        )

        # Passo 7 — tipo de material (assinaturas + memória de material)
        if not dist_assinaturas:
            passos.append(
                "Passo 7 — Identificação de material: nenhuma região candidata a "
                "entulho, nada a classificar."
            )
        else:
            soma = sum(dist_assinaturas.values()) or 1.0
            rank = sorted(dist_assinaturas.items(), key=lambda kv: -kv[1])
            composicao = ", ".join(
                f"{NOME_MATERIAIS[m]} {f / soma * 100:.0f}%" for m, f in rank[:3] if f > 0
            )
            texto = (
                f"Passo 7 — Identificação de material: assinaturas visuais por região "
                f"(cor/textura/contorno) indicam {composicao}."
            )
            if dist_memoria is not None:
                rank_mem = sorted(dist_memoria.items(), key=lambda kv: -kv[1])[:2]
                voto_txt = ", ".join(f"{NOME_MATERIAIS[m]} {p * 100:.0f}%" for m, p in rank_mem)
                texto += (
                    f" Memória de material ({n_casos_material} confirmações, peso "
                    f"{peso_memoria_material:.2f}) vota: {voto_txt}."
                )
            if material_dominante is not None:
                total_agl = sum(aglomerados.values())
                agl_txt = ""
                if total_agl:
                    por_material = ", ".join(
                        f"{NOME_MATERIAIS[m]} {n}" for m, n in
                        sorted(aglomerados.items(), key=lambda kv: -kv[1])
                    )
                    plural = "" if total_agl == 1 else "s"
                    agl_txt = (
                        f" {total_agl} aglomerado{plural} de desperdício detectado{plural} "
                        f"({por_material})."
                    )
                texto += (
                    f" → material predominante: {NOME_MATERIAIS[material_dominante]} "
                    f"(confiança {dist_material[material_dominante] * 100:.0f}%)."
                    + agl_txt
                )
            passos.append(texto)

        # Passo 8 — área do descarte (escala: largura informada pelo operador)
        if not dist_assinaturas:
            passos.append(
                "Passo 8 — Área do descarte: nenhuma área de material detectada na cena."
            )
        else:
            area_cena = largura_m * largura_m * (altura / largura)
            linhas = sorted(
                ((m, f * area_cena) for m, f in dist_assinaturas.items()),
                key=lambda kv: -kv[1],
            )
            linhas = [(m, a) for m, a in linhas if a >= 0.01][:4]
            areas_txt = "; ".join(
                f"{NOME_MATERIAIS[m]} {_num(a)} m²" for m, a in linhas
            )
            rotulo = ("Área do descarte" if p_final >= 0.5
                      else "Área CANDIDATA a descarte (veredito: sem desperdício)")
            passos.append(
                f"Passo 8 — {rotulo} (cena ≈ {_num(area_cena)} m², largura "
                f"{_num(largura_m, 2)} m): {areas_txt} "
                f"→ total ≈ {_num(sum(a for _, a in linhas))} m²."
            )
        return passos

    # ------------------------------------------------------------------ #
    # Aprendizado com o humano
    # ------------------------------------------------------------------ #
    def registrar_feedback(self, resultado, rotulo_humano, observacao="", material=None):
        """Registra a experiência confirmada pelo operador e consolida o
        aprendizado. `material` é o tipo de material confirmado/corrigido
        (chave de materiais.MATERIAIS) — alimente-o quando o rótulo for
        DESPERDÍCIO; cenas limpas não têm material. Retorna
        (id_da_experiencia, sistema_acertou, consolidou)."""
        vetor = resultado.vetor
        if vetor is None:
            vetor = vetor_de(resultado.caracteristicas)
        acertou = int(rotulo_humano) == resultado.rotulo_previsto
        exp_id = self.memoria.adicionar(
            caminho_imagem=resultado.caminho,
            rotulo=rotulo_humano,
            probabilidade_sistema=resultado.probabilidade,
            acertou=acertou,
            observacao=observacao,
            vetor=vetor,
            material=material,
        )
        X, y = self.memoria.matrizes()
        consolidou = self.modelo.consolidar(X, y)
        return exp_id, acertou, consolidou

    # ------------------------------------------------------------------ #
    # Imagem-relatório anotada
    # ------------------------------------------------------------------ #
    def salvar_imagem_anotada(self, resultado):
        # Uma cor por tipo de material + legenda com a composição do desperdício.
        cores = [
            CORES_MATERIAIS.get(m, CORES_MATERIAIS["outro"])
            for m in resultado.materiais_regioes
        ]
        legenda = None
        if resultado.distribuicao_materiais:
            soma = sum(resultado.distribuicao_materiais.values()) or 1.0
            rank = sorted(resultado.distribuicao_materiais.items(), key=lambda kv: -kv[1])
            legenda = [
                (f"{NOME_MATERIAIS[m]} — {f / soma * 100:.0f}%",
                 CORES_MATERIAIS.get(m, CORES_MATERIAIS["outro"]))
                for m, f in rank[:4] if f > 0
            ]
        anotada = anotar_imagem(
            resultado.imagem, resultado.contornos,
            resultado.veredito_desperdicio, resultado.probabilidade,
            cores_regioes=cores, legenda=legenda,
        )
        nome = f"{nome_base_sem_acento(resultado.caminho)}_anotada.jpg"
        caminho = os.path.join(self.pasta_anotacoes, nome)
        salvar_anotacao(anotada, caminho)
        return caminho, anotada

    def status(self):
        d, l = self.memoria.contagem_rotulos()
        materiais = {}
        for e in self.memoria.experiencias:
            if e.get("material"):
                materiais[e["material"]] = materiais.get(e["material"], 0) + 1
        return {
            "total": self.memoria.total,
            "desperdicio": d,
            "limpo": l,
            "taxa_acerto": self.memoria.taxa_acerto(),
            "modelo_treinado": self.modelo.treinado,
            "n_treino": self.modelo.n_treino,
            "versao_caracteristicas": VERSAO_CARACTERISTICAS,
            "materiais": materiais,
        }

    def resetar(self):
        self.memoria.resetar()
        self.modelo.resetar()


class _EscalonadorProvisorio:
    """Padroniza vetores quando o modelo incremental ainda não existe
    (memória com 1 exemplo): usa média/desvio da própria memória."""

    def __init__(self, X):
        self.media = X.mean(axis=0)
        self.desvio = X.std(axis=0)
        self.desvio[self.desvio == 0] = 1.0

    def transformar(self, X):
        return (np.asarray(X, dtype=np.float64) - self.media) / self.desvio
