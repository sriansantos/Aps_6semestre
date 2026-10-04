#!/usr/bin/env python3
"""
main.py — PIVC: Detecção de desperdício de materiais em obras (linha de comando).

Sistema de identificação por imagens (APS 5º/6º CC) que recebe uma imagem de
uma obra e julga se há DESPERDÍCIO DE MATERIAIS nela. A cada veredito, um
operador humano confirma ou corrige a resposta — e o sistema registra a
experiência na memória persistente, aprendendo progressivamente.

Comandos:
  python main.py analisar <imagens...>          Analisa imagens e pede confirmação humana
  python main.py analisar --pasta <dir>         Analisa todas as imagens de uma pasta
  python main.py analisar ... --largura-m 5     Informa a largura real da cena (área em m²)
  python main.py analisar ... --sem-feedback    Analisa sem pedir confirmação
  python main.py adicionar <dir> --rotulo d|l   Ensina um lote inteiro (d=desperdício, l=limpo)
  python main.py status                         Mostra o estado da memória e do modelo
  python main.py relatorio                      Exporta relatório (Markdown + CSV)
  python main.py painel                         Abre a interface gráfica
  python main.py resetar                        Apaga a memória (pede confirmação)

Exemplos:
  python main.py analisar "obra1.jpg" "obra2.png"
  python main.py analisar --pasta "fotos_obra/"
  python main.py adicionar "dataset/desperdicio" --rotulo d
"""

import argparse
import glob
import os
import sys

try:
    import cv2  # noqa: F401  (verificação antecipada da principal dependência)

    from pivc_core.detetor import DetetorDesperdicio
    from pivc_core.materiais import CHAVES_CLI, NOME_MATERIAIS
    from pivc_core.memoria import NOME_ROTULOS, ROTULO_DESPERDICIO, ROTULO_LIMPO
    from pivc_core.visao import carregar_imagem
except ModuleNotFoundError as erro:
    sys.stderr.write(
        f"\nERRO: falta a biblioteca '{erro.name}' neste Python.\n\n"
        "Este computador tem mais de um Python instalado e o terminal está usando\n"
        "um que não tem as dependências do projeto.\n\n"
        "Solução 1 (mais fácil): rode tudo pelo 'executar.bat', na pasta do projeto:\n"
        "    executar.bat analisar \"imagem.jpg\"\n"
        "    executar.bat painel\n\n"
        "Solução 2: instale as dependências no Python que o seu terminal usa:\n"
        f"    \"{sys.executable}\" -m pip install opencv-python numpy scikit-learn pillow joblib\n\n"
    )
    sys.exit(1)

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LARGURA_LINHA = 74


def num(valor, casas=1):
    """Número com vírgula decimal (pt-BR) para a saída textual."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------- #
# Utilidades de apresentação
# ---------------------------------------------------------------------- #
def linha(caractere="─"):
    print(caractere * LARGURA_LINHA)


def mostrar_analise(resultado, caminho_anotada=None):
    linha("═")
    print(f"IMAGEM: {resultado.caminho}")
    veredito = "DESPERDÍCIO DE MATERIAIS DETECTADO" if resultado.veredito_desperdicio else "SEM DESPERDÍCIO DE MATERIAIS"
    print(f"VEREDITO: {veredito}")
    print(
        f"Probabilidade de desperdício: {resultado.percentual} "
        f"(confiança {resultado.confianca})"
    )
    linha()
    print("Linha de pensamento do sistema:")
    for passo in resultado.raciocinio:
        print(f"  • {passo}")
    if resultado.casos_analogos:
        print("  Casos da memória consultados:")
        for caso in resultado.casos_analogos:
            rotulo = NOME_ROTULOS[caso["rotulo"]]
            print(f"    - Experiência #{caso['id']} ({os.path.basename(caso['caminho'])}): "
                  f"{rotulo}, distância {caso['distancia']:.2f}")
    n_regioes = len(resultado.contornos)
    if n_regioes:
        maior = max(d["area_frac"] for d in resultado.contornos)
        plural = "" if n_regioes == 1 else "s"
        detalhe_agl = ""
        if resultado.aglomerados_por_material:
            por_material = ", ".join(
                f"{NOME_MATERIAIS.get(m, m)} {n}" for m, n in
                sorted(resultado.aglomerados_por_material.items(), key=lambda kv: -kv[1])
            )
            detalhe_agl = f" — por material: {por_material}"
        print(f"  Aglomerado{plural} de desperdício: {n_regioes}{detalhe_agl} "
              f"(o maior ocupa {maior * 100:.0f}% da cena)")
    if resultado.material_dominante:
        frac = resultado.material_distribuicao_final.get(resultado.material_dominante, 0.0)
        print(f"  Material predominante: {NOME_MATERIAIS[resultado.material_dominante]} "
              f"({frac * 100:.0f}% da área desperdiçada; confiança {resultado.material_confianca * 100:.0f}%)")
    if resultado.areas_m2:
        aspecto = resultado.imagem_altura / max(resultado.imagem_largura, 1)
        area_cena = (resultado.largura_m or 0) ** 2 * aspecto
        rotulo_area = ("Área do descarte" if resultado.veredito_desperdicio
                       else "Área CANDIDATA a descarte (veredito: sem desperdício)")
        print(f"  {rotulo_area} — cena ≈ {num(area_cena)} m² "
              f"(largura {num(resultado.largura_m, 2)} m):")
        for material, area in sorted(resultado.areas_m2.items(), key=lambda kv: -kv[1])[:4]:
            print(f"    • {NOME_MATERIAIS.get(material, material)}: {num(area)} m²")
        print(f"    TOTAL: {num(sum(resultado.areas_m2.values()))} m²")
    elif resultado.veredito_desperdicio:
        print("  Área do descarte: nenhuma área de material medida.")
    if caminho_anotada:
        print(f"  Imagem-relatório salva em: {caminho_anotada}")
    linha("═")


def perguntar_feedback(resultado):
    """Loop de confirmação humana. Retorna o rótulo confirmado ou None (pular)."""
    while True:
        resposta = input("O sistema acertou? [s]im / [n]não, corrigir / [p]ular: ").strip().lower()
        if resposta in ("s", "sim", ""):
            return resultado.rotulo_previsto
        if resposta in ("n", "nao", "não"):
            break
        if resposta in ("p", "pular"):
            return None
        print("  Resposta não reconhecida — digite s, n ou p.")

    while True:
        resposta = input("Rótulo correto — [d] desperdício / [l] local limpo: ").strip().lower()
        if resposta in ("d", "desperdicio", "desperdício"):
            return ROTULO_DESPERDICIO
        if resposta in ("l", "limpo"):
            return ROTULO_LIMPO
        print("  Resposta não reconhecida — digite d ou l.")


def perguntar_material(palpite):
    """Pergunta o material predominante quando o rótulo confirmado é
    desperdício. Enter mantém o palpite do sistema (ou 'outro' se não houver)."""
    padrao = palpite if palpite else "outro"
    ajuda = ", ".join(
        f"[{chave}]{nome.lower()}" for chave, nome in
        (("g", "gesso"), ("p", "papelão"), ("m", "madeira"), ("l", "lonas/plástico"),
         ("a", "alvenaria"), ("o", "outro"))
    )
    while True:
        resposta = input(
            f"Material predominante — {ajuda} (Enter mantém {NOME_MATERIAIS[padrao]}): "
        ).strip().lower()
        if resposta == "":
            return padrao
        if resposta in CHAVES_CLI:
            return CHAVES_CLI[resposta]
        if resposta in NOME_MATERIAIS:  # aceita também o nome completo
            return resposta
        print("  Resposta não reconhecida — use g, p, m, l, a ou o.")


def processar_feedback(detetor, resultado):
    try:
        rotulo = perguntar_feedback(resultado)
    except (EOFError, KeyboardInterrupt):
        print("\nFeedback interrompido — experiência NÃO registrada.")
        return False
    if rotulo is None:
        print("• Sem confirmação: esta imagem não entrou na memória.")
        return True

    material = None
    if rotulo == ROTULO_DESPERDICIO:
        try:
            material = perguntar_material(resultado.material_dominante)
        except (EOFError, KeyboardInterrupt):
            print("\nFeedback interrompido — experiência NÃO registrada.")
            return False

    observacao = ""
    try:
        observacao = input("Observação (opcional, Enter pula): ").strip()
    except (EOFError, KeyboardInterrupt):
        pass

    exp_id, acertou, consolidou = detetor.registrar_feedback(
        resultado, rotulo, observacao, material=material)
    situacao = "o sistema ACERTOU" if acertou else "o sistema ERROU e foi CORRIGIDO"
    print(f"✔ Experiência #{exp_id} registrada na memória ({situacao}).")
    if material:
        print(f"  Material confirmado: {NOME_MATERIAIS[material]} "
              f"(a memória de material aprendeu com isso).")
    if consolidou:
        print(f"  Modelo incremental consolidado com {detetor.memoria.total} experiências.")
    return True


# ---------------------------------------------------------------------- #
# Comandos
# ---------------------------------------------------------------------- #
def cmd_analisar(args, detetor):
    caminhos = list(args.imagens)
    if args.pasta:
        if not os.path.isdir(args.pasta):
            print(f"ERRO: pasta não encontrada: {args.pasta}")
            return 1
        for extensao in EXTENSOES:
            caminhos.extend(glob.glob(os.path.join(args.pasta, f"*{extensao}")))
            caminhos.extend(glob.glob(os.path.join(args.pasta, f"*{extensao.upper()}")))
    if not caminhos:
        print("ERRO: informe ao menos uma imagem ou use --pasta <diretório>.")
        return 1

    interrompido = False
    for i, caminho in enumerate(sorted(set(caminhos)), 1):
        if len(set(caminhos)) > 1:
            print(f"\n[{i}/{len(set(caminhos))}]")
        try:
            resultado = detetor.analisar(caminho, largura_m=args.largura_m)
        except ValueError as erro:
            print(f"✖ Não foi possível analisar '{caminho}': {erro}")
            continue
        except OSError as erro:
            print(f"✖ Erro de leitura em '{caminho}': {erro}")
            continue

        caminho_anotada = None
        try:
            caminho_anotada, _ = detetor.salvar_imagem_anotada(resultado)
        except (OSError, ValueError) as erro:
            print(f"(aviso: não foi possível salvar a imagem anotada: {erro})")
        mostrar_analise(resultado, caminho_anotada)

        if not args.sem_feedback:
            if not processar_feedback(detetor, resultado):
                interrompido = True
                break
    return 0 if not interrompido else 130


def cmd_adicionar(args, detetor):
    mapa = {"d": ROTULO_DESPERDICIO, "desperdicio": ROTULO_DESPERDICIO,
            "l": ROTULO_LIMPO, "limpo": ROTULO_LIMPO}
    chave = args.rotulo.lower()
    if chave not in mapa:
        print("ERRO: --rotulo deve ser 'd' (desperdício) ou 'l' (limpo).")
        return 1
    rotulo = mapa[chave]
    if not os.path.isdir(args.pasta):
        print(f"ERRO: pasta não encontrada: {args.pasta}")
        return 1

    caminhos = []
    for extensao in EXTENSOES:
        caminhos.extend(glob.glob(os.path.join(args.pasta, f"*{extensao}")))
        caminhos.extend(glob.glob(os.path.join(args.pasta, f"*{extensao.upper()}")))
    caminhos = sorted(set(caminhos))
    if not caminhos:
        print(f"ERRO: nenhuma imagem encontrada em {args.pasta}")
        return 1

    print(f"Ensinando o sistema com {len(caminhos)} imagens da pasta "
          f"'{args.pasta}' como {NOME_ROTULOS[rotulo]}...")
    adicionadas = 0
    for caminho in caminhos:
        try:
            resultado = detetor.analisar(caminho)
            exp_id, _, _ = detetor.registrar_feedback(
                resultado, rotulo, observacao="lote de treinamento (adicionar)")
            adicionadas += 1
            print(f"  ✔ #{exp_id} {os.path.basename(caminho)} "
                  f"(o sistema previu: {resultado.percentual} desperdício)")
        except (ValueError, OSError) as erro:
            print(f"  ✖ {os.path.basename(caminho)}: {erro}")
    print(f"\nConcluído: {adicionadas} experiências adicionadas. "
          f"Memória total: {detetor.memoria.total}.")
    return 0


def cmd_status(args, detetor):
    s = detetor.status()
    linha("═")
    print("ESTADO DA MEMÓRIA E DO MODELO")
    linha()
    print(f"  Experiências confirmadas por humanos : {s['total']}")
    print(f"    - com desperdício                  : {s['desperdicio']}")
    print(f"    - sem desperdício (local limpo)    : {s['limpo']}")
    if s["taxa_acerto"] is not None:
        print(f"  Taxa de acerto do sistema antes da confirmação: {s['taxa_acerto'] * 100:.1f}%")
    if s["materiais"]:
        lista_materiais = ", ".join(
            f"{NOME_MATERIAIS.get(m, m)} {n}" for m, n in sorted(s["materiais"].items())
        )
        print(f"  Materiais confirmados pelo operador: {lista_materiais}")
    estado = (f"treinado com {s['n_treino']} experiências" if s["modelo_treinado"]
              else "ainda não treinado (falta confirmação das duas classes)")
    print(f"  Modelo incremental: {estado} (reconstruído da memória a cada inicialização)")
    print(f"  Versão das características: {s['versao_caracteristicas']}")
    print(f"  Arquivo da memória: {detetor.pasta_dados}{os.sep}memoria.json")
    linha("═")
    return 0


def cmd_relatorio(args, detetor):
    if detetor.memoria.total == 0:
        print("A memória está vazia — confirme algumas análises antes de gerar o relatório.")
        return 1
    pasta = os.path.join(detetor.pasta_dados, "relatorios")
    cam_md, cam_csv = detetor.memoria.exportar_relatorio(pasta)
    print("Relatórios exportados:")
    print(f"  • {cam_md}")
    print(f"  • {cam_csv}")
    return 0


def cmd_resetar(args, detetor):
    print("ATENÇÃO: isto apaga TODA a memória de experiências e o modelo aprendido.")
    resposta = input(f"Digite RESETAR para confirmar: ").strip()
    if resposta == "RESETAR":
        detetor.resetar()
        print("Memória apagada. O sistema voltou ao estado de conhecimento inicial (heurística).")
        return 0
    print("Cancelado — nada foi apagado.")
    return 1


def cmd_painel(args, detetor):
    import gui
    gui.executar(detetor)
    return 0


# ---------------------------------------------------------------------- #
# Programa principal
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="PIVC — Detecção de desperdício de materiais em obras com aprendizado por confirmação humana.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Comandos:")[1] if "Comandos:" in __doc__ else None,
    )
    parser.add_argument("--dados", default=None,
                        help="pasta onde a memória/modelo ficam guardados (padrão: ./dados)")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_an = sub.add_parser("analisar", help="analisa imagens e pede confirmação humana")
    p_an.add_argument("imagens", nargs="*", help="caminhos das imagens")
    p_an.add_argument("--pasta", default=None, help="analisa todas as imagens desta pasta")
    p_an.add_argument("--largura-m", type=float, default=None,
                      help="largura real da cena em metros (para a área do descarte em m²; "
                           "padrão: 4,0 m)")
    p_an.add_argument("--sem-feedback", action="store_true",
                      help="não pede confirmação humana (só julga)")
    p_an.set_defaults(func=cmd_analisar)

    p_ad = sub.add_parser("adicionar", help="ensina o sistema com um lote de imagens rotuladas")
    p_ad.add_argument("pasta", help="pasta com as imagens")
    p_ad.add_argument("--rotulo", required=True, choices=["d", "l"],
                      help="d = desperdício, l = local limpo")
    p_ad.set_defaults(func=cmd_adicionar)

    p_st = sub.add_parser("status", help="mostra o estado da memória e do modelo")
    p_st.set_defaults(func=cmd_status)

    p_re = sub.add_parser("relatorio", help="exporta relatório de experiências (Markdown + CSV)")
    p_re.set_defaults(func=cmd_relatorio)

    p_pa = sub.add_parser("painel", help="abre a interface gráfica")
    p_pa.set_defaults(func=cmd_painel)

    p_rs = sub.add_parser("resetar", help="apaga toda a memória aprendida")
    p_rs.set_defaults(func=cmd_resetar)

    args = parser.parse_args()
    pasta_dados = args.dados or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
    os.makedirs(pasta_dados, exist_ok=True)
    detetor = DetetorDesperdicio(pasta_dados)

    try:
        sys.exit(args.func(args, detetor))
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
        sys.exit(130)


if __name__ == "__main__":
    main()
