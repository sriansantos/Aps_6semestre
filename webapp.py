"""
webapp.py — Front-end web do PIVC com "swipe" estilo Tinder.

O operador solta as fotos da obra na página; para cada foto o sistema mostra
o veredito, a probabilidade e a linha de pensamento ABAIXO da foto, e a
confirmação humana é feita ARRASTANDO o cartão:
  → arrastar para a DIREITA  =  "é desperdício"
  → arrastar para a ESQUERDA =  "sem desperdício (local limpo)"
Cada swipe vai para a memória persistente e retreina o modelo — igual aos
outros modos (CLI e painel), agora com o gesto do Tinder.

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
from urllib.parse import quote

from flask import Flask, jsonify, render_template, request, send_file

from pivc_core.detetor import DetetorDesperdicio
from pivc_core.visao import carregar_imagem, nome_base_sem_acento

PASTA_APP = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA_APP, "dados")
PASTA_UPLOADS = os.path.join(PASTA_DADOS, "capturas")
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

app = Flask(__name__)
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

    exp_id, acertou, consolidou = detetor.registrar_feedback(res, int(rotulo))
    return jsonify({
        "ok": True,
        "exp_id": exp_id,
        "acertou": acertou,
        "consolidou": consolidou,
        "status": detetor.status(),
    })


# ---------------------------------------------------------------------- #
def executar(porta=5000, abrir_navegador=True):
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
