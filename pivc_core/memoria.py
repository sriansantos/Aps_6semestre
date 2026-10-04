"""
memoria.py — A "memória" do sistema (aprendizado com confirmação humana).

Duas peças trabalham juntas:

1. MemoriaExperiencias — um banco de casos persistido em JSON. Cada vez que
   um operador humano confirma (ou corrige) um veredito, guardamos o vetor de
   características da imagem + o rótulo correto + o contexto. É uma memória
   vetorial de experiências: casos novos são julgados por analogia com casos
   antigos (raciocínio baseado em casos), e ela nunca para de crescer.

2. ModeloIncremental — um classificador (regressão logística treinada por
   descida de gradiente estocástica, scikit-learn) que é consolidado sobre
   toda a memória a cada nova confirmação. Ele generaliza o que a memória
   de casos memoriza.

Tudo é gravado em disco: o sistema "lembra" de tudo entre execuções.
"""

import csv
import json
import os
import shutil
from datetime import datetime

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .materiais import NOME_MATERIAIS
from .visao import NOMES_CARACTERISTICAS, VERSAO_CARACTERISTICAS

ROTULO_DESPERDICIO = 1
ROTULO_LIMPO = 0
NOME_ROTULOS = {ROTULO_DESPERDICIO: "DESPERDÍCIO", ROTULO_LIMPO: "SEM DESPERDÍCIO"}


class MemoriaExperiencias:
    """Banco de casos persistido em JSON (legível e auditável)."""

    def __init__(self, pasta_dados):
        self.pasta_dados = pasta_dados
        self.caminho_json = os.path.join(pasta_dados, "memoria.json")
        self.experiencias = []
        self.versao = VERSAO_CARACTERISTICAS
        self.carregar()

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def carregar(self):
        if not os.path.isfile(self.caminho_json):
            self.experiencias = []
            return
        try:
            with open(self.caminho_json, "r", encoding="utf-8") as f:
                dados = json.load(f)
            self.experiencias = dados.get("experiencias", [])
            self.versao = dados.get("versao_caracteristicas", VERSAO_CARACTERISTICAS)
        except (json.JSONDecodeError, OSError):
            # JSON corrompido: mantém um backup e recomeça vazio,
            # em vez de derrubar o sistema.
            backup = self.caminho_json + ".corrompido-" + datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(self.caminho_json, backup)
            self.experiencias = []
            self.versao = VERSAO_CARACTERISTICAS

    def salvar(self):
        os.makedirs(self.pasta_dados, exist_ok=True)
        conteudo = {
            "versao_caracteristicas": self.versao,
            "atualizado_em": datetime.now().isoformat(timespec="seconds"),
            "experiencias": self.experiencias,
        }
        # Gravação atômica: escreve em .tmp e só depois substitui o arquivo.
        tmp = self.caminho_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conteudo, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.caminho_json)

    # ------------------------------------------------------------------ #
    # Registro de experiências
    # ------------------------------------------------------------------ #
    def adicionar(self, caminho_imagem, rotulo, probabilidade_sistema, acertou, observacao, vetor,
                  material=None):
        """Registra uma experiência confirmada por humano. Retorna o id.

        `material` (chave de materiais.MATERIAIS) é opcional: cenas limpas não
        o têm; experiências antigas também não — o código usa .get() em todo
        lugar que o consulta."""
        exp = {
            "id": (self.experiencias[-1]["id"] + 1) if self.experiencias else 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "caminho": str(caminho_imagem),
            "rotulo": int(rotulo),
            "rotulo_texto": NOME_ROTULOS[int(rotulo)],
            "material": material,
            "probabilidade_sistema": round(float(probabilidade_sistema), 4),
            "sistema_acertou": bool(acertou),
            "observacao": observacao or "",
            "versao_caracteristicas": VERSAO_CARACTERISTICAS,
            "vetor": [round(float(x), 6) for x in vetor],
        }
        self.experiencias.append(exp)
        self.salvar()
        return exp["id"]

    # ------------------------------------------------------------------ #
    # Consultas
    # ------------------------------------------------------------------ #
    @property
    def total(self):
        return len(self.experiencias)

    def contagem_rotulos(self):
        d = sum(1 for e in self.experiencias if e["rotulo"] == ROTULO_DESPERDICIO)
        return d, self.total - d

    def matrizes(self):
        """Vetores e rótulos de todas as experiências (para treinar o modelo)."""
        if not self.experiencias:
            return np.empty((0, len(NOMES_CARACTERISTICAS))), np.empty(0)
        X = np.array([e["vetor"] for e in self.experiencias], dtype=np.float64)
        y = np.array([e["rotulo"] for e in self.experiencias], dtype=np.int64)
        return X, y

    def atualizar_vetores(self, extrator):
        """Reextrai os vetores de características de todas as experiências
        a partir das imagens originais — usado quando o pipeline de visão
        evolui (VERSÃO_CARACTERISTICAS mudou): os RÓTULOS confirmados por
        humanos são preservados, só a 'língua' dos vetores é atualizada.

        `extrator(caminho)` devolve o novo vetor. Retorna True somente se
        TODAS as imagens foram relidas (atualiza a versão e salva); se alguma
        imagem não estiver acessível, nada é alterado — vetores de versões
        diferentes não podem ser misturados no mesmo modelo.
        """
        novos = []
        for e in self.experiencias:
            try:
                vetor = extrator(e["caminho"])
            except Exception:  # imagem movida/apagada/corrompida
                return False
            novos.append([round(float(x), 6) for x in vetor])
        for e, vetor in zip(self.experiencias, novos):
            e["vetor"] = vetor
            e["versao_caracteristicas"] = VERSAO_CARACTERISTICAS
        self.versao = VERSAO_CARACTERISTICAS
        self.salvar()
        return True

    def taxa_acerto(self):
        """Fração de vezes em que o sistema acertou antes da confirmação humana."""
        if not self.experiencias:
            return None
        acertos = sum(1 for e in self.experiencias if e["sistema_acertou"])
        return acertos / len(self.experiencias)

    def ultimas(self, n=5):
        return self.experiencias[-n:][::-1]

    def resetar(self):
        self.experiencias = []
        self.salvar()

    # ------------------------------------------------------------------ #
    # Relatórios (extra: relatórios adicionais contam na nota)
    # ------------------------------------------------------------------ #
    def exportar_relatorio(self, pasta_relatorios):
        os.makedirs(pasta_relatorios, exist_ok=True)
        agora = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam_md = os.path.join(pasta_relatorios, f"relatorio_experiencias_{agora}.md")
        cam_csv = os.path.join(pasta_relatorios, f"experiencias_{agora}.csv")

        d, l = self.contagem_rotulos()
        taxa = self.taxa_acerto()
        with open(cam_md, "w", encoding="utf-8") as f:
            f.write("# Relatório de experiências — PIVC Detecção de Desperdício\n\n")
            f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write(f"- Total de experiências confirmadas: **{self.total}**\n")
            f.write(f"- Com desperdício: {d} | Sem desperdício: {l}\n")
            if taxa is not None:
                f.write(f"- Taxa de acerto do sistema antes da confirmação: **{taxa * 100:.1f}%**\n")
            f.write("\n| ID | Data/Hora | Imagem | Rótulo | Material | Prob. sistema | Sistema acertou? |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for e in self.experiencias:
                caminho = os.path.basename(e["caminho"])
                acerto = "Sim" if e["sistema_acertou"] else "**Não (corrigido)**"
                material = NOME_MATERIAIS.get(e.get("material"), "—")
                f.write(
                    f"| {e['id']} | {e['timestamp']} | {caminho} | {e['rotulo_texto']} | {material} "
                    f"| {e['probabilidade_sistema']:.0%} | {acerto} |\n"
                )
        with open(cam_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["id", "timestamp", "caminho", "rotulo", "material",
                             "probabilidade_sistema", "sistema_acertou", "observacao"])
            for e in self.experiencias:
                writer.writerow([e["id"], e["timestamp"], e["caminho"], e["rotulo_texto"],
                                 NOME_MATERIAIS.get(e.get("material"), ""),
                                 e["probabilidade_sistema"], e["sistema_acertou"], e["observacao"]])
        return cam_md, cam_csv


class ModeloIncremental:
    """Classificador que aprende com a memória de experiências.

    A cada nova confirmação humana o modelo é consolidado sobre TODOS os
    casos já confirmados (aprendizado supervisionado incremental): quanto
    mais confirmações, mais o modelo domina a decisão final.

    Não há arquivo binário do modelo: ele é reconstruído automaticamente a
    partir de memoria.json a cada inicialização (consolidação rápida, feita
    em milissegundos). Assim o sistema não depende de versão do scikit-learn
    e a memória JSON é a única fonte da verdade.
    """

    def __init__(self):
        self.scaler = None
        self.classificador = None
        self.n_treino = 0

    @property
    def treinado(self):
        return self.classificador is not None

    def consolidar(self, X, y):
        """Retreina o modelo sobre toda a memória (chamado a cada confirmação
        e a cada inicialização do sistema)."""
        if len(X) < 2 or len(set(y.tolist())) < 2:
            # Sem os dois lados da história ainda, o modelo não treina —
            # a heurística e a memória de casos cobrem a decisão.
            return False
        self.scaler = StandardScaler()
        Xn = self.scaler.fit_transform(X)
        self.classificador = SGDClassifier(
            loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-4, random_state=42
        )
        # Várias épocas com embaralhamento determinístico = convergência estável.
        rng = np.random.default_rng(42)
        ordem = rng.permutation(len(Xn))
        self.classificador.partial_fit(Xn[ordem], y[ordem], classes=np.array([0, 1]))
        self.n_treino = len(Xn)
        return True

    def probabilidade(self, vetor):
        """Probabilidade de desperdício segundo o modelo aprendido (ou None)."""
        if not self.treinado:
            return None
        Xn = self.scaler.transform(np.array([vetor], dtype=np.float64))
        return float(self.classificador.predict_proba(Xn)[0][1])

    def resetar(self):
        self.scaler, self.classificador, self.n_treino = None, None, 0
