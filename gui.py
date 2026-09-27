#!/usr/bin/env python3
"""
gui.py — Painel gráfico do PIVC (Tkinter, biblioteca padrão do Python).

Fluxo humano-no-ciclo:
  1. O operador abre uma imagem (ou uma pasta com várias).
  2. O sistema mostra o veredito, a probabilidade, a cadeia de raciocínio
     e a imagem anotada com as regiões candidatas a entulho.
  3. O operador confirma ("É desperdício" / "Sem desperdício") — e a
     experiência vai para a memória persistente, que retreina o modelo.

Pode ser aberto por `python main.py painel` ou diretamente: `python gui.py`.
"""

import glob
import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    from PIL import Image, ImageTk

    from pivc_core.detetor import DetetorDesperdicio
except ModuleNotFoundError as erro:
    print(
        f"\nERRO: falta a biblioteca '{erro.name}' neste Python.\n\n"
        "Rode o sistema pelo 'executar.bat' (na pasta do projeto), que usa o\n"
        "ambiente virtual com tudo instalado, ou instale as dependências:\n"
        f"    \"{sys.executable}\" -m pip install opencv-python numpy scikit-learn pillow joblib\n"
    )
    try:
        input("Pressione Enter para sair...")
    except EOFError:
        pass
    sys.exit(1)

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LARG_IMG, ALT_IMG = 470, 320
VERMELHO, VERDE, CINZA_TXT = "#c62828", "#2e7d32", "#424242"


class JanelaCaptura(tk.Toplevel):
    """Janela de captura pela câmera (aquisição de imagem em tempo real).

    O roteiro da APS prevê a obtenção da imagem "através de arquivos" e
    também por "scanner ou câmera" — esta janela cobre a câmera: mostra a
    pré-visualização ao vivo e, ao capturar, salva o quadro em
    dados/capturas/ e envia direto para a análise.
    """

    LARGURA_PREVIEW = 480

    def __init__(self, mestre, pasta_capturas, callback):
        super().__init__(mestre)
        self.title("PIVC — Capturar imagem da câmera")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.cancelar)
        self.pasta_capturas = pasta_capturas
        self.callback = callback
        self.captura = None        # objeto cv2.VideoCapture
        self.frame_atual = None    # último quadro lido (BGR, resolução cheia)
        self._foto = None          # referência do PhotoImage (evita coleta de lixo)
        self._tarefa = None        # id do after() do laço de preview

        ttk.Label(
            self, text="Enquadre a área da obra e clique em Capturar (ou barra de espaço). ESC cancela.",
            padding=(8, 6),
        ).pack()
        self.lbl_preview = tk.Label(self, bg="#111111", width=66, height=22)
        self.lbl_preview.pack(padx=8)
        quadro = ttk.Frame(self, padding=8)
        quadro.pack(fill=tk.X)
        self.btn_capturar = ttk.Button(quadro, text="📸 Capturar", command=self.capturar, state=tk.DISABLED)
        self.btn_capturar.pack(side=tk.LEFT)
        ttk.Button(quadro, text="✖ Cancelar", command=self.cancelar).pack(side=tk.LEFT, padx=6)
        self.bind("<space>", lambda _e: self.capturar())
        self.bind("<Escape>", lambda _e: self.cancelar())

        self.lbl_status = ttk.Label(self, text="Abrindo a câmera, aguarde…", padding=(8, 4), foreground=CINZA_TXT)
        self.lbl_status.pack(fill=tk.X)
        # Abre a câmera depois que a janela já está renderizada (a abertura trava ~1s).
        self.after(60, self._abrir_e_iniciar)

    # ---- abertura da câmera ------------------------------------------ #
    def _abrir_e_iniciar(self):
        self.captura = self._abrir_camera()
        if self.captura is None:
            self.lbl_status.configure(text="")
            messagebox.showerror(
                "Câmera não encontrada",
                "Nenhuma câmera pôde ser aberta (índices 0 a 2).\n\n"
                "Verifique se o computador possui câmera, se nenhum outro\n"
                "programa a está usando e se as permissões estão liberadas.",
                parent=self,
            )
            self.cancelar()
            return
        self.btn_capturar.configure(state=tk.NORMAL)
        self.lbl_status.configure(text="Câmera ativa — pronta para capturar.")
        self._atualizar_preview()

    def _abrir_camera(self):
        """Tenta as câmeras 0, 1 e 2 e devolve a primeira que responde."""
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0  # DSHOW abre mais rápido no Windows
        for indice in range(3):
            cap = cv2.VideoCapture(indice, backend)
            ok, _ = cap.read()
            if cap.isOpened() and ok:
                return cap
            cap.release()
        return None

    # ---- laço de pré-visualização ------------------------------------ #
    def _atualizar_preview(self):
        if self.captura is None:
            return
        ok, frame = self.captura.read()
        if not ok:
            messagebox.showerror(
                "Câmera perdida",
                "A câmera parou de responder (desconectada ou em uso por outro programa).",
                parent=self,
            )
            self.cancelar()
            return
        self.frame_atual = cv2.flip(frame, 1)  # espelhado, como esperamos ao apontar
        rgb = cv2.cvtColor(self.frame_atual, cv2.COLOR_BGR2RGB)
        imagem = Image.fromarray(rgb)
        imagem.thumbnail((self.LARGURA_PREVIEW, 360))
        self._foto = ImageTk.PhotoImage(imagem)
        self.lbl_preview.configure(image=self._foto, width=0, height=0)
        self._tarefa = self.after(40, self._atualizar_preview)  # ~25 quadros/s

    # ---- ações -------------------------------------------------------- #
    def capturar(self):
        if self.frame_atual is None:
            return
        os.makedirs(self.pasta_capturas, exist_ok=True)
        nome = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        caminho = os.path.join(self.pasta_capturas, nome)
        ok, buf = cv2.imencode(".jpg", self.frame_atual, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            messagebox.showerror("Falha ao capturar", "Não foi possível gravar o quadro capturado.", parent=self)
            return
        buf.tofile(caminho)  # gravação robusta a caminhos com acento
        self._liberar()
        self.destroy()
        self.callback(caminho)  # segue direto para a análise no painel

    def cancelar(self):
        self._liberar()
        self.destroy()

    def _liberar(self):
        if self._tarefa is not None:
            try:
                self.after_cancel(self._tarefa)
            except tk.TclError:
                pass
            self._tarefa = None
        if self.captura is not None:
            self.captura.release()
            self.captura = None
        self.frame_atual = None


class PainelPIVC:
    def __init__(self, detetor):
        self.detetor = detetor
        self.fila = []
        self.resultado = None

        self.root = tk.Tk()
        self.root.title("PIVC — Detecção de Desperdício de Materiais em Obras")
        self.root.geometry("1020x760")
        self.root.minsize(960, 700)
        self._montar_widgets()

    # ------------------------------------------------------------------ #
    # Construção da tela
    # ------------------------------------------------------------------ #
    def _montar_widgets(self):
        fonte = ("Segoe UI", 10)
        self.root.option_add("*Font", fonte)

        # Barra superior -------------------------------------------------
        topo = ttk.Frame(self.root, padding=8)
        topo.pack(fill=tk.X)
        ttk.Button(topo, text="📂 Abrir imagem…", command=self.abrir_imagem).pack(side=tk.LEFT)
        ttk.Button(topo, text="📸 Capturar da câmera", command=self.capturar_da_camera).pack(side=tk.LEFT, padx=6)
        ttk.Button(topo, text="📁 Abrir pasta…", command=self.abrir_pasta).pack(side=tk.LEFT, padx=6)
        self.lbl_status = ttk.Label(topo, anchor=tk.E, foreground=CINZA_TXT)
        self.lbl_status.pack(side=tk.RIGHT)

        # Painel de imagens ----------------------------------------------
        painel_imgs = ttk.Frame(self.root, padding=(8, 0))
        painel_imgs.pack(fill=tk.X)
        quadro_a = ttk.LabelFrame(painel_imgs, text=" Imagem analisada ", padding=4)
        quadro_a.grid(row=0, column=0, sticky="nsew")
        quadro_b = ttk.LabelFrame(painel_imgs, text=" Regiões detectadas (imagem-relatório) ", padding=4)
        quadro_b.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        painel_imgs.columnconfigure(1, weight=1)
        self.lbl_imagem_a = ttk.Label(quadro_a, anchor=tk.CENTER)
        self.lbl_imagem_a.pack()
        self.lbl_imagem_b = ttk.Label(quadro_b, anchor=tk.CENTER)
        self.lbl_imagem_b.pack()

        # Veredito --------------------------------------------------------
        quadro_v = ttk.LabelFrame(self.root, text=" Veredito do sistema ", padding=8)
        quadro_v.pack(fill=tk.X, padx=8, pady=6)
        self.lbl_veredito = tk.Label(quadro_v, text="Nenhuma imagem analisada ainda.",
                                     font=("Segoe UI", 15, "bold"), fg=CINZA_TXT, anchor="w")
        self.lbl_veredito.pack(fill=tk.X)
        barra_conf = ttk.Frame(quadro_v)
        barra_conf.pack(fill=tk.X, pady=4)
        ttk.Label(barra_conf, text="Probabilidade de desperdício:").pack(side=tk.LEFT)
        self.barra = ttk.Scale(barra_conf, from_=0, to=100, value=0, orient=tk.HORIZONTAL)
        self.barra.configure(state="disabled")
        self.barra.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_conf = ttk.Label(barra_conf, text="—", width=26)
        self.lbl_conf.pack(side=tk.LEFT)

        # Raciocínio ------------------------------------------------------
        quadro_r = ttk.LabelFrame(self.root, text=" Linha de pensamento do sistema ", padding=6)
        quadro_r.pack(fill=tk.BOTH, expand=True, padx=8)
        self.txt_raciocinio = tk.Text(quadro_r, height=9, wrap=tk.WORD, state=tk.DISABLED,
                                      background="#fafafa", relief=tk.FLAT)
        scroll = ttk.Scrollbar(quadro_r, command=self.txt_raciocinio.yview)
        self.txt_raciocinio.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_raciocinio.pack(fill=tk.BOTH, expand=True)

        # Confirmação humana ----------------------------------------------
        quadro_f = ttk.LabelFrame(self.root, text=" Confirmação humana (o sistema aprende com você) ", padding=8)
        quadro_f.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.btn_desperdicio = ttk.Button(quadro_f, text="✔ É desperdício", command=lambda: self.confirmar(1),
                                          state=tk.DISABLED)
        self.btn_limpo = ttk.Button(quadro_f, text="✔ Sem desperdício", command=lambda: self.confirmar(0),
                                    state=tk.DISABLED)
        self.btn_pular = ttk.Button(quadro_f, text="⏭ Pular (não registrar)", command=self.pular,
                                    state=tk.DISABLED)
        self.btn_desperdicio.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_limpo.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_pular.pack(side=tk.LEFT)
        self.lbl_feedback = ttk.Label(quadro_f, foreground=CINZA_TXT)
        self.lbl_feedback.pack(side=tk.LEFT, padx=12)

        self._atualizar_status()

    # ------------------------------------------------------------------ #
    # Entrada de imagens
    # ------------------------------------------------------------------ #
    def abrir_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecione a imagem da obra",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")],
        )
        if caminho:
            self.fila = [caminho]
            self._processar_fila()

    # ------------------------------------------------------------------ #
    # Captura pela câmera (imagem inserida na hora)
    # ------------------------------------------------------------------ #
    def capturar_da_camera(self):
        janela = getattr(self, "_janela_captura", None)
        if janela is not None:
            try:
                if janela.winfo_exists():  # já existe uma captura aberta
                    janela.lift()
                    janela.focus_force()
                    return
            except tk.TclError:
                pass
        pasta_capturas = os.path.join(self.detetor.pasta_dados, "capturas")
        self._janela_captura = JanelaCaptura(self.root, pasta_capturas, callback=self._receber_captura)

    def _receber_captura(self, caminho):
        """Recebe o arquivo capturado pela câmera e segue para a análise."""
        self.fila = [caminho]
        self._processar_fila()

    def abrir_pasta(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta com as imagens da obra")
        if not pasta:
            return
        caminhos = []
        for ext in EXTENSOES:
            caminhos.extend(glob.glob(os.path.join(pasta, f"*{ext}")))
            caminhos.extend(glob.glob(os.path.join(pasta, f"*{ext.upper()}")))
        if not caminhos:
            messagebox.showwarning("Nada encontrado", f"Nenhuma imagem encontrada em:\n{pasta}")
            return
        self.fila = sorted(set(caminhos))
        self._processar_fila()

    def _processar_fila(self):
        if not self.fila:
            return
        caminho = self.fila.pop(0)
        try:
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            self.resultado = self.detetor.analisar(caminho)
            self._mostrar_resultado()
        except (ValueError, OSError) as erro:
            messagebox.showerror("Erro na análise", f"Não foi possível analisar:\n{caminho}\n\n{erro}")
        finally:
            self.root.config(cursor="")
            if self.resultado is not None:
                for b in (self.btn_desperdicio, self.btn_limpo, self.btn_pular):
                    b.configure(state=tk.NORMAL)

    # ------------------------------------------------------------------ #
    # Exibição do resultado
    # ------------------------------------------------------------------ #
    def _renderizar(self, matriz_bgr):
        rgb = cv2.cvtColor(matriz_bgr, cv2.COLOR_BGR2RGB)
        imagem = Image.fromarray(rgb)
        imagem.thumbnail((LARG_IMG, ALT_IMG))
        return ImageTk.PhotoImage(imagem)

    def _mostrar_resultado(self):
        r = self.resultado

        foto_a = self._renderizar(r.imagem)
        self.lbl_imagem_a.configure(image=foto_a)
        self.lbl_imagem_a.image = foto_a
        _, anotada = self.detetor.salvar_imagem_anotada(r)
        foto_b = self._renderizar(anotada)
        self.lbl_imagem_b.configure(image=foto_b)
        self.lbl_imagem_b.image = foto_b

        if r.veredito_desperdicio:
            texto, cor = "⚠ DESPERDÍCIO DE MATERIAIS DETECTADO", VERMELHO
        else:
            texto, cor = "✔ SEM DESPERDÍCIO DE MATERIAIS", VERDE
        self.lbl_veredito.configure(text=f"{texto}  —  {r.percentual} (confiança {r.confianca})", fg=cor)

        self.barra.configure(state=tk.NORMAL)
        self.barra.set(r.probabilidade * 100)
        self.barra.configure(state=tk.DISABLED)
        self.lbl_conf.configure(text=f"{r.percentual} de probabilidade")

        self.txt_raciocinio.configure(state=tk.NORMAL)
        self.txt_raciocinio.delete("1.0", tk.END)
        self.txt_raciocinio.insert(tk.END, f"Arquivo: {r.caminho}\n\n")
        for passo in r.raciocinio:
            self.txt_raciocinio.insert(tk.END, f"• {passo}\n")
        if r.casos_analogos:
            self.txt_raciocinio.insert(tk.END, "\nCasos da memória consultados:\n")
            for caso in r.casos_analogos:
                rotulo = "DESPERDÍCIO" if caso["rotulo"] == 1 else "SEM DESPERDÍCIO"
                self.txt_raciocinio.insert(
                    tk.END, f"   - Experiência #{caso['id']}: {rotulo} (distância {caso['distancia']:.2f})\n")
        self.txt_raciocinio.configure(state=tk.DISABLED)

        self.lbl_feedback.configure(text="")
        self._atualizar_status()

    # ------------------------------------------------------------------ #
    # Confirmação humana → memória
    # ------------------------------------------------------------------ #
    def confirmar(self, rotulo):
        if self.resultado is None:
            return
        exp_id, acertou, consolidou = self.detetor.registrar_feedback(self.resultado, rotulo)
        situacao = ("o sistema havia acertado" if acertou
                    else "o sistema havia errado — memória CORRIGIDA")
        msg = (f"✔ Experiência #{exp_id} registrada ({situacao})."
               + (f" Modelo consolidado com {self.detetor.memoria.total} experiências." if consolidou else ""))
        self.lbl_feedback.configure(text=msg, foreground=VERDE if acertou else VERMELHO)
        self._atualizar_status()
        self._proxima_da_fila()

    def pular(self):
        self.lbl_feedback.configure(text="Imagem ignorada — não foi registrada na memória.")
        self._proxima_da_fila()

    def _proxima_da_fila(self):
        self.resultado = None
        for b in (self.btn_desperdicio, self.btn_limpo, self.btn_pular):
            b.configure(state=tk.DISABLED)
        if self.fila:
            self._processar_fila()

    def _atualizar_status(self):
        s = self.detetor.status()
        taxa = f"{s['taxa_acerto'] * 100:.0f}% de acerto antes da confirmação" if s["taxa_acerto"] is not None else "sem histórico de acerto"
        modelo = "modelo treinado" if s["modelo_treinado"] else "modelo aguardando treinamento"
        self.lbl_status.configure(
            text=f"Memória: {s['total']} experiências ({s['desperdicio']} desperdício / {s['limpo']} limpo) · {modelo} · {taxa}")

    # ------------------------------------------------------------------ #
    def executar(self):
        self.root.mainloop()


def executar(detetor=None):
    if detetor is None:
        pasta_dados = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
        os.makedirs(pasta_dados, exist_ok=True)
        detetor = DetetorDesperdicio(pasta_dados)
    PainelPIVC(detetor).executar()


if __name__ == "__main__":
    executar()
