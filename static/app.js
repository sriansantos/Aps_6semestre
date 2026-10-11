/* ============================================================
   PIVC — lógica do baralho estilo Tinder
   arrastar →  direita = DESPERDÍCIO · esquerda = SEM DESPERDÍCIO
   ============================================================ */
"use strict";

const el = {
  entrada: () => document.getElementById("tela-entrada"),
  jogo: () => document.getElementById("tela-jogo"),
  fim: () => document.getElementById("tela-fim"),
  gota: () => document.getElementById("gota"),
  inputFotos: () => document.getElementById("input-fotos"),
  inputPasta: () => document.getElementById("input-pasta"),
  btnPasta: () => document.getElementById("btn-pasta"),
  btnEscolher: () => document.getElementById("btn-escolher"),
  btnMais: () => document.getElementById("btn-mais"),
  pilha: () => document.getElementById("pilha"),
  progresso: () => document.getElementById("progresso"),
  chip: () => document.getElementById("chip-memoria"),
  toast: () => document.getElementById("toast"),
  fimResumo: () => document.getElementById("fim-resumo"),
};

let fila = [];              // {caminho, dados, decidido, rotulo}
let indice = 0;             // cartão atual
let registradasSessao = 0;  // confirmações nesta sessão
let tokenRender = 0;        // invalida preenchimentos atrasados
let arraste = null;

/* ---------------- utilidades ---------------- */
async function api(rota, opcoes) {
  const resposta = await fetch(rota, opcoes);
  const json = await resposta.json().catch(() => ({ erro: "resposta inválida" }));
  if (!resposta.ok) throw new Error(json.erro || `erro ${resposta.status}`);
  return json;
}

let timerToast = null;
function toast(mensagem, erro = false) {
  const t = el.toast();
  t.textContent = mensagem;
  t.classList.toggle("erro", erro);
  t.hidden = false;
  clearTimeout(timerToast);
  timerToast = setTimeout(() => { t.hidden = true; }, 2600);
}

function atualizarChip(status) {
  if (!status) return;
  const taxa = status.taxa_acerto === null ? "" :
    ` · ${Math.round(status.taxa_acerto * 100)}% acerto`;
  el.chip().innerHTML =
    `🧠 memória: <b>${status.total} exp</b>${taxa}`;
}

function urlImagem(caminho) {
  return "/imagem?caminho=" + encodeURIComponent(caminho);
}

/* ---------------- entrada de fotos ---------------- */
async function carregarUploads(arquivos) {
  if (!arquivos || arquivos.length === 0) return;
  toast(`Analisando ${arquivos.length} foto(s)…`);
  const fd = new FormData();
  [...arquivos].forEach(f => fd.append("fotos", f));
  try {
    const r = await api("/api/carregar", { method: "POST", body: fd });
    if (r.ignoradas.length) toast(`(${r.ignoradas.length} arquivo(s) ignorados — não são imagens válidas)`, true);
    iniciarFila(r.adicionadas);
  } catch (e) { toast("Falha ao carregar: " + e.message, true); }
}

async function carregarPasta(nome) {
  if (!nome) return;
  try {
    const r = await api("/api/carregar_pasta", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pasta: nome }),
    });
    iniciarFila(r.adicionadas);
    if (r.adicionadas.length === 0) toast("Nenhuma imagem nessa pasta", true);
  } catch (e) { toast("Falha: " + e.message, true); }
}

function iniciarFila(novas) {
  novas.forEach(c => fila.push({ caminho: c, dados: null, decidido: false }));
  api("/api/status").then(atualizarChip).catch(() => {});
  if (indice < fila.length) {
    el.entrada().hidden = true;
    el.fim().hidden = true;
    el.jogo().hidden = false;
    renderizar();
  }
}

/* ---------------- baralho ---------------- */
async function analisar(item) {
  if (item.dados) return item.dados;
  item.dados = await api("/api/analizar?caminho=" + encodeURIComponent(item.caminho));
  return item.dados;
}

function renderizar() {
  const meuToken = ++tokenRender;
  const pilha = el.pilha();
  const atual = fila[indice];

  if (!atual) {   // acabou a pilha
    el.jogo().hidden = true;
    el.entrada().hidden = false;
    el.fim().hidden = false;
    el.fimResumo().textContent =
      `${registradasSessao} confirmação(ões) registrada(s) nesta sessão — ` +
      `cada uma deixou o modelo mais esperto.`;
    return;
  }

  // cartões de trás (só visual, esmaecidos)
  let html = "";
  for (let k = Math.min(fila.length, indice + 3) - 1; k > indice; k--) {
    html += `<div class="carta fundo">
      <div class="foto"><img src="${urlImagem(fila[k].caminho)}" alt=""></div></div>`;
  }
  html += `
  <div class="carta topo" id="carta-topo">
    <div class="foto" id="area-foto">
      <img id="foto-principal" src="${urlImagem(atual.caminho)}" alt="foto da obra">
      <div class="selo desperdicio" id="selo-desperdicio">DESPERDÍCIO</div>
      <div class="selo limpo" id="selo-limpo">LIMPO</div>
      <div class="dica-flip">toque p/ ver as regiões</div>
    </div>
    <div class="info" id="info-cartao">
      <div class="carregando-info">analisando a cena…</div>
    </div>
  </div>`;
  pilha.innerHTML = html;
  el.progresso().textContent = `${indice + 1} / ${fila.length}`;
  ligarArraste();

  // dados da análise (com guarda contra respostas atrasadas)
  analisar(atual).then(dados => {
    if (meuToken !== tokenRender) return;
    preencherInfo(dados);
    const pre = new Image(); pre.src = dados.url_anotada;  // prepara o flip
  }).catch(e => {
    if (meuToken !== tokenRender) return;
    document.getElementById("info-cartao").innerHTML =
      `<div class="carregando-info">⚠ ${e.message}</div>`;
  });

  // pré-analisa as duas próximas em segundo plano
  fila.slice(indice + 1, indice + 3).forEach(it => analisar(it).catch(() => {}));
}

function preencherInfo(dados) {
  const cor = dados.desperdicio ? "amarelo" : "verde";
  document.getElementById("info-cartao").innerHTML = `
    <div class="veredito ${cor}">${dados.rotulo_texto}</div>
    <div class="sub">${dados.probabilidade}% de probabilidade · confiança ${dados.confianca}</div>
    <div class="regioes-linha">📍 ${dados.n_regioes} região(ões) candidata(s) — a maior ocupa ${dados.maior_regiao}% da cena</div>
    <details>
      <summary>linha de pensamento do sistema</summary>
      <ul>${dados.raciocinio.map(p => `<li>${p}</li>`).join("")}</ul>
    </details>`;
}

/* ---------------- arrastar (pointer events) ---------------- */
function ligarArraste() {
  const carta = document.getElementById("carta-topo");
  const foto = document.getElementById("area-foto");
  const seloD = document.getElementById("selo-desperdicio");
  const seloL = document.getElementById("selo-limpo");
  if (!carta || !foto) return;
  let flipPendente = false;

  foto.addEventListener("pointerdown", (e) => {
    arraste = { x0: e.clientX, y0: e.clientY, dx: 0, clicou: true };
    foto.setPointerCapture(e.pointerId);
    carta.style.transition = "none";
  });

  foto.addEventListener("pointermove", (e) => {
    if (!arraste) return;
    arraste.dx = e.clientX - arraste.x0;
    const dy = (e.clientY - arraste.y0) * 0.25;
    if (Math.abs(arraste.dx) > 6) arraste.clicou = false;
    carta.style.transform =
      `translate(${arraste.dx}px, ${dy}px) rotate(${arraste.dx * 0.06}deg)`;
    seloD.style.opacity = Math.min(1, Math.max(0, arraste.dx / 90));
    seloL.style.opacity = Math.min(1, Math.max(0, -arraste.dx / 90));
  });

  foto.addEventListener("pointerup", () => {
    if (!arraste) return;
    const dx = arraste.dx;
    const clicou = arraste.clicou;
    arraste = null;
    carta.style.transition = "";
    if (dx > 110) decidir(1);
    else if (dx < -110) decidir(0);
    else if (clicou) alternarRegioes();   // foi um toque, não um arrasto
    else {
      carta.style.transform = "";
      seloD.style.opacity = 0;
      seloL.style.opacity = 0;
    }
  });
  foto.addEventListener("pointercancel", () => { arraste = null; });
}

function alternarRegioes() {
  const item = fila[indice];
  if (!item || !item.dados) return;
  const img = document.getElementById("foto-principal");
  item.mostrandoAnotada = !item.mostrandoAnotada;
  img.src = item.mostrandoAnotada ? item.dados.url_anotada : item.dados.url_imagem;
}

/* ---------------- decisões ---------------- */
function decidir(rotulo) {
  const item = fila[indice];
  const carta = document.getElementById("carta-topo");
  if (!item || item.decidido || !carta) return;
  item.decidido = true;
  item.rotulo = rotulo;
  registradasSessao += rotulo === 0 || rotulo === 1 ? 1 : 0;

  // sai voando como no Tinder
  carta.classList.add("saindo");
  carta.style.transform =
    `translate(${rotulo === 1 ? 560 : -560}px, -50px) rotate(${rotulo === 1 ? 22 : -22}deg)`;

  api("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ caminho: item.caminho, rotulo }),
  }).then(r => {
    atualizarChip(r.status);
    toast(r.acertou
      ? `✔ Experiência #${r.exp_id} — o sistema havia acertado`
      : `✏ Experiência #${r.exp_id} — memória CORRIGIDA pelo operador`);
  }).catch(e => toast("Falha ao registrar: " + e.message, true));

  indice++;
  setTimeout(renderizar, 290);
}

function pular() {
  const item = fila[indice];
  const carta = document.getElementById("carta-topo");
  if (!item || item.decidido || !carta) return;
  item.decidido = true;
  carta.classList.add("saindo");
  carta.style.transform = "translateY(-120%) rotate(4deg)";
  indice++;
  toast("cartão pulado — nada foi registrado na memória");
  setTimeout(renderizar, 290);
}

/* ---------------- ligações da página ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  el.btnEscolher().addEventListener("click", () => el.inputFotos().click());
  el.inputFotos().addEventListener("change", (e) => carregarUploads(e.target.files));
  el.btnPasta().addEventListener("click", () => carregarPasta(el.inputPasta().value.trim()));
  el.inputPasta().addEventListener("keydown", (e) => {
    if (e.key === "Enter") carregarPasta(e.target.value.trim());
  });
  el.btnMais().addEventListener("click", () => {
    el.fim().hidden = true;
    el.entrada().hidden = false;
  });
  document.getElementById("btn-limpo").addEventListener("click", () => decidir(0));
  document.getElementById("btn-desperdicio").addEventListener("click", () => decidir(1));
  document.getElementById("btn-pular").addEventListener("click", pular);

  document.addEventListener("keydown", (e) => {
    if (el.jogo().hidden) return;
    if (e.key === "ArrowLeft") decidir(0);
    else if (e.key === "ArrowRight") decidir(1);
    else if (e.key.toLowerCase() === "p") pular();
  });

  // arrastar-e-soltar de arquivos sobre a área de entrada
  const gota = el.gota();
  ["dragover", "dragenter"].forEach(ev =>
    document.addEventListener(ev, (e) => { e.preventDefault(); gota.classList.add("sobre"); }));
  ["dragleave", "drop"].forEach(ev =>
    document.addEventListener(ev, (e) => { e.preventDefault(); gota.classList.remove("sobre"); }));
  document.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) carregarUploads(e.dataTransfer.files);
  });

  api("/api/status").then(atualizarChip).catch(() =>
    toast("Servidor PIVC não responde — rode: python main.py site", true));
});
