"""
webapp.py — Front-end web do PIVC com "swipe" estilo Tinder (Engenhender).

O operador faz login (apenas o nome — é identificação de quem confirma, não
segurança), solta as fotos da obra ou fotografa pela câmera; para cada foto o
sistema mostra o veredito, a probabilidade e a linha de pensamento ABAIXO da
foto, e a confirmação humana é feita ARRASTANDO o cartão:
  → arrastar para a DIREITA  =  "é desperdício"
  → arrastar para a ESQUERDA =  "sem desperdício (local limpo)"
Cada swipe vai para a memória persistente e retreina o modelo — igual aos
outros modos (CLI e painel), agora com o gesto do Tinder.

Há ainda o botão "limpar inteligência" (apaga memória + modelo aprendido,
voltando ao conhecimento a priori da heurística).

Tema visual: preto + amarelo de capacete de EPI de obra.

Executar:  python main.py site     (ou:  python webapp.py [--porta 5000])
A distância de captura é padronizada (5 m) e fica nos metadados — ver
pivc_core/configuracao.py — sem aparecer na tela.
"""

import glob
import os
import threading
import uuid
import webbrowser
from datetime import datetime
from urllib.parse import quote

from flask import Flask, jsonify, render_template, request, send_file, session

from pivc_core.detetor import DetetorDesperdicio
from pivc_core.materiais import CORES_MATERIAIS, MATERIAIS, NOME_MATERIAIS
from pivc_core.visao import carregar_imagem, nome_base_sem_acento

PASTA_APP = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA_APP, "dados")
PASTA_UPLOADS = os.path.join(PASTA_DADOS, "capturas")
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

app = Flask(__name__)
# Chave de sessão estável: o "login" do Engenhender guarda o nome do operador
# na sessão do navegador apenas para assinar as confirmações na memória.
app.secret_key = "engenhender-pivc-local"
detetor = DetetorDesperdicio(PASTA_DADOS)

# Caminhos liberados nesta sessão (só analisa/serva o que entrou pela API —
# evita expor arquivos arbitrários do disco via ?caminho=).
LIBERADOS = set()
# Resultados das análises da sessão, para registrar o feedback sem reanalisar.
RESULTADOS = {}


def _abspath(caminho):
    return os.path.abspath(caminho)


def _liberar(caminho):
    LIBERADOS.add(_abspath(caminho))


def _liberado(caminho):
    return _abspath(caminho) in LIBERADOS


# ---------------------------------------------------------------------- #
# Página e arquivos
# ---------------------------------------------------------------------- #
@app.route("/")
def pagina():
    return render_template("index.html")


@app.route("/imagem")
def imagem():
    caminho = request.args.get("caminho", "")
    if not _liberado(caminho):
        return jsonify({"erro": "arquivo fora da sessão atual"}), 403
    return send_file(_abspath(caminho))


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #
@app.route("/api/status")
def api_status():
    return jsonify(detetor.status())


@app.route("/api/login", methods=["POST"])
def api_login():
    """Login do Engenhender: registra o nome do operador na sessão.
    Não é controle de acesso — é a assinatura de quem confirma, gravada em
    cada experiência da memória (auditável no relatório)."""
    nome = (request.get_json(silent=True) or {}).get("nome", "").strip()
    if not nome or len(nome) > 60:
        return jsonify({"erro": "informe um nome válido (até 60 caracteres)"}), 400
    session["operador"] = nome
    return jsonify({"ok": True, "nome": nome})


@app.route("/api/sair", methods=["POST"])
def api_sair():
    session.pop("operador", None)
    return jsonify({"ok": True})


@app.route("/api/materiais")
def api_materiais():
    """Catálogo de materiais para o front-end: nome bonito + cor (RGB web)
    de cada ponto de cor nos cartões."""
    catalogo = []
    for chave in MATERIAIS:
        b, g, r = CORES_MATERIAIS[chave]
        catalogo.append({
            "chave": chave,
            "nome": NOME_MATERIAIS[chave],
            "cor": f"#{r:02x}{g:02x}{b:02x}",
        })
    return jsonify(catalogo)


@app.route("/api/captura", methods=["POST"])
def api_captura():
    """Recebe um quadro fotografado pela câmera do navegador (dataURL JPEG).
    Igual ao botão 'Capturar da câmera' do painel Tkinter, agora na web."""
    dados = (request.get_json(silent=True) or {}).get("quadro", "")
    if not dados.startswith("data:image/"):
        return jsonify({"erro": "quadro de câmera inválido"}), 400
    try:
        import base64

        bruto = base64.b64decode(dados.split(",", 1)[1])
    except (ValueError, IndexError):
        return jsonify({"erro": "quadro de câmera ilegível"}), 400

    os.makedirs(PASTA_UPLOADS, exist_ok=True)
    destino = os.path.join(
        PASTA_UPLOADS,
        f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.jpg",
    )
    with open(destino, "wb") as f:
        f.write(bruto)
    try:
        carregar_imagem(destino)  # valida de verdade antes de aceitar
    except (ValueError, OSError):
        os.remove(destino)
        return jsonify({"erro": "não foi possível decodificar a foto capturada"}), 400
    _liberar(destino)
    return jsonify({"adicionadas": [destino]})


@app.route("/api/limpar_inteligencia", methods=["POST"])
def api_limpar_inteligencia():
    """Apaga TODA a memória de experiências e o modelo aprendido (com dupla
    confirmação no front-end). O sistema volta ao conhecimento a priori da
    heurística de engenharia e volta a aprender com as confirmações."""
    if (request.get_json(silent=True) or {}).get("confirmar") != "APAGAR":
        return jsonify({"erro": "confirmação ausente"}), 400
    detetor.resetar()
    RESULTADOS.clear()
    return jsonify({"ok": True, "status": detetor.status()})


@app.route("/api/carregar", methods=["POST"])
def api_carregar():
    """Recebe uploads de fotos (input de arquivos / arrastar-e-soltar)."""
    adicionadas, ignoradas = [], []
    os.makedirs(PASTA_UPLOADS, exist_ok=True)
    for arquivo in request.files.getlist("fotos"):
        ext = os.path.splitext(arquivo.filename)[1].lower()
        if ext not in EXTENSOES:
            ignoradas.append(arquivo.filename)
            continue
        base = nome_base_sem_acento(arquivo.filename) or "foto"
        destino = os.path.join(PASTA_UPLOADS, f"web_{uuid.uuid4().hex[:8]}_{base}{ext}")
        arquivo.save(destino)
        try:
            carregar_imagem(destino)  # valida de verdade antes de aceitar
        except (ValueError, OSError):
            ignoradas.append(arquivo.filename)
            continue
        _liberar(destino)
        adicionadas.append(destino)
    return jsonify({"adicionadas": adicionadas, "ignoradas": ignoradas})


@app.route("/api/carregar_pasta", methods=["POST"])
def api_carregar_pasta():
    """Carrega todas as imagens de uma pasta do servidor (uso em demonstração)."""
    pasta = (request.get_json(silent=True) or {}).get("pasta", "").strip()
    if not os.path.isdir(pasta):
        return jsonify({"erro": f"Pasta não encontrada no servidor: {pasta}"}), 400
    caminhos = []
    for ext in EXTENSOES:
        caminhos += glob.glob(os.path.join(pasta, f"*{ext}"))
        caminhos += glob.glob(os.path.join(pasta, f"*{ext.upper()}"))
    caminhos = sorted(set(caminhos))
    for caminho in caminhos:
        _liberar(caminho)
    return jsonify({"adicionadas": caminhos})


@app.route("/api/analizar")
def api_analizar():
    caminho = request.args.get("caminho", "")
    if not _liberado(caminho):
        return jsonify({"erro": "arquivo fora da sessão atual"}), 403
    chave = _abspath(caminho)
    if chave in RESULTADOS:
        res = RESULTADOS[chave]
    else:
        try:
            res = detetor.analisar(caminho)
        except (ValueError, OSError) as erro:
            return jsonify({"erro": str(erro)}), 400
        RESULTADOS[chave] = res
    cam_anotada, _ = detetor.salvar_imagem_anotada(res)
    _liberar(cam_anotada)

    maior = max((d["area_frac"] for d in res.contornos), default=0.0)
    return jsonify({
        "caminho": caminho,
        "desperdicio": res.veredito_desperdicio,
        "rotulo_texto": "DESPERDÍCIO PROVÁVEL" if res.veredito_desperdicio else "SEM DESPERDÍCIO",
        "probabilidade": round(res.probabilidade * 100),
        "confianca": res.confianca,
        "n_regioes": len(res.contornos),
        "maior_regiao": round(maior * 100),
        "raciocinio": res.raciocinio,
        # identificação do tipo de material (materiais.py): o que é, quanto
        # ocupa em m², em quantos montes e a COR de anotação — exibidos
        # abaixo da foto como pontos coloridos
        "materiais": [
            {"chave": m,
             "nome": NOME_MATERIAIS.get(m, m.capitalize()),
             "area_m2": round(area, 1),
             "montes": res.aglomerados_por_material.get(m, 0),
             "cor": "#{:02x}{:02x}{:02x}".format(*reversed(CORES_MATERIAIS.get(m, (200, 200, 200))))}
            for m, area in sorted(res.areas_m2.items(), key=lambda kv: -kv[1])
        ],
        "url_imagem": "/imagem?caminho=" + quote(chave),
        "url_anotada": "/imagem?caminho=" + quote(_abspath(cam_anotada)),
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Registra o resultado do swipe: 1 = desperdício, 0 = sem desperdício."""
    dados = request.get_json(silent=True) or {}
    rotulo = dados.get("rotulo")
    caminho = dados.get("caminho", "")
    if rotulo not in (0, 1) or not _liberado(caminho):
        return jsonify({"erro": "requisição inválida"}), 400

    chave = _abspath(caminho)
    res = RESULTADOS.get(chave)
    if res is None:  # servidor reiniciou no meio da sessão: reanalisa
        try:
            res = detetor.analisar(caminho)
        except (ValueError, OSError) as erro:
            return jsonify({"erro": str(erro)}), 400
        RESULTADOS[chave] = res

    exp_id, acertou, consolidou = detetor.registrar_feedback(
        res, int(rotulo),
        material=dados.get("material"),          # confirma/corrige o tipo
        operador=session.get("operador"),        # quem confirmou (login)
    )
    return jsonify({
        "ok": True,
        "exp_id": exp_id,
        "acertou": acertou,
        "consolidou": consolidou,
        "status": detetor.status(),
    })


# ---------------------------------------------------------------------- #
# Portas que os navegadores (Chromium) recusam carregar mesmo livres —
# o fallback precisa pulá-las junto com as ocupadas.
PORTAS_BLOQUEADAS = {5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6697, 10080}


def _porta_livre(porta):
    if porta in PORTAS_BLOQUEADAS:
        return False
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soquete:
        try:
            soquete.bind(("127.0.0.1", porta))
            return True
        except OSError:
            return False


def executar(porta=5000, abrir_navegador=True):
    # Se a porta pedida estiver ocupada (outro PIVC aberto, outro programa),
    # sobe na próxima livre — clique duplo nunca falha por porta.
    for tentativa in range(10):
        if _porta_livre(porta + tentativa):
            porta += tentativa
            break
    else:
        print(f"Nenhuma porta livre entre {porta} e {porta + 9}. "
              f"Encerre outro PIVC aberto ou use --porta.")
        return
    if abrir_navegador:
        threading.Timer(
            1.2, lambda: webbrowser.open(f"http://127.0.0.1:{porta}")
        ).start()
    print(f"PIVC web rodando em http://127.0.0.1:{porta}  (Ctrl+C encerra)")
    app.run(host="127.0.0.1", port=porta, threaded=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PIVC — front-end swipe (Tinder)")
    parser.add_argument("--porta", type=int, default=5000)
    parser.add_argument("--nao-abrir-navegador", action="store_true")
    args = parser.parse_args()
    executar(porta=args.porta, abrir_navegador=not args.nao_abrir_navegador)
