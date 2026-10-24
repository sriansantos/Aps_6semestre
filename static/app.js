/* ============================================================
   Engenhender — PIVC com "swipe" estilo Tinder
   arrastar →  direita = DESPERDÍCIO · esquerda = SEM DESPERDÍCIO
   ============================================================ */
"use strict";

const el = {
  login: () => document.getElementById("tela-login"),
  cabecalho: () => document.getElementById("cabecalho"),
  faixaTopo: () => document.getElementById("faixa-topo"),
  barra: () => document.getElementById("barra-ferramentas"),
  inputNome: () => document.getElementById("input-nome"),
  btnEntrar: () => document.getElementById("btn-entrar"),
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
  // câmera
  modalCamera: () => document.getElementById("modal-camera"),
  video: () => document.getElementById("video-camera"),
  btnCamera: () => document.getElementById("btn-camera"),
  btnCapturar: () => document.getElementById("btn-capturar"),
  btnFecharCamera: () => document.getElementById("btn-fechar-camera"),
  cameraDica: () => document.getElementById("camera-dica"),
  btnSair: () => document.getElementById("btn-sair"),
  nomeOperador: () => document.getElementById("nome-operador"),
  // limpar IA
  modalLimpar: () => document.getElementById("modal-limpar"),
  btnLimpar: () => document.getElementById("btn-limpar-ia"),
  btnLimparCancelar: () => document.getElementById("btn-limpar-cancelar"),
  btnLimparConfirmar: () => document.getElementById("btn-limpar-confirmar"),
  limparResumo: () => document.getElementById("limpar-resumo"),
};

let operador = null;
let CATALOGO_MATERIAIS = [];   // [{chave, nome, cor}] de /api/materiais
let fila = [];              // {caminho, dados, decidido, rotulo}
let indice = 0;             // cartão atual
let registradasSessao = 0;  // confirmações nesta sessão
let tokenRender = 0;        // invalida preenchimentos atrasados
let arraste = null;
let fluxoCamera = null;     // MediaStream da câmera aberta

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
    preencherInfo(dados, atual);
    const pre = new Image(); pre.src = dados.url_anotada;  // prepara o flip
  }).catch(e => {
    if (meuToken !== tokenRender) return;
    document.getElementById("info-cartao").innerHTML =
      `<div class="carregando-info">⚠ ${e.message}</div>`;
  });

  // pré-analisa as duas próximas em segundo plano
  fila.slice(indice + 1, indice + 3).forEach(it => analisar(it).catch(() => {}));
}

function preencherInfo(dados, item) {
  const cor = dados.desperdicio ? "amarelo" : "verde";
  const linhaMateriais = (dados.materiais || [])
    .slice(0, 3)
    .map(m => `<span class="ponto-cor" style="background:${m.cor}"></span>${m.nome}
      ~${String(m.area_m2).replace(".", ",")} m²`
      + (m.montes > 1 ? ` (${m.montes} montes)` : ""))
    .join(" · ");
  const alertaQualidade = dados.qualidade && dados.qualidade.alerta;
  const linhaPessoas = dados.pessoas > 0
    ? `<div class="regioes-linha">👷 ${dados.pessoas} pessoa(s) na cena — região(ões) desconsiderada(s) da análise</div>`
    : "";
  const linhaAlerta = alertaQualidade
    ? `<div class="regioes-linha alerta-foto">📸 ${alertaQualidade}</div>`
    : "";

  // Seletor de material: default = material dominante detectado; o operador
  // pode corrigir antes de arrastar — a confirmação treina a identificação.
  const defaultMaterial = item.materialSelecionado || (dados.materiais || [])[0]?.chave || null;
  item.materialSelecionado = defaultMaterial;
  const chips = CATALOGO_MATERIAIS
    .filter(m => m.chave !== "outro")
    .map(m => `<button type="button" class="chip-material${m.chave === defaultMaterial ? " selecionado" : ""}"
      data-chave="${m.chave}" title="${m.nome}">
      <span class="ponto-cor" style="background:${m.cor}"></span>${m.nome.split("/")[0]}</button>`)
    .join("");

  document.getElementById("info-cartao").innerHTML = `
    <div class="veredito ${cor}">${dados.rotulo_texto}</div>
    <div class="sub">${dados.probabilidade}% de probabilidade · confiança ${dados.confianca}</div>
    ${linhaMateriais ? `<div class="regioes-linha">🔧 ${linhaMateriais}</div>` : ""}
    <div class="regioes-linha">📍 ${dados.n_regioes} região(ões) candidata(s) — a maior ocupa ${dados.maior_regiao}% da cena</div>
    ${linhaPessoas}
    ${linhaAlerta}
    <div class="seletor-material">
      <span class="seletor-rotulo">é o quê? (toque p/ corrigir)</span>
      <div class="chips-material">${chips}</div>
    </div>
    <details>
      <summary>linha de pensamento do sistema</summary>
      <ul>${dados.raciocinio.map(p => `<li>${p}</li>`).join("")}</ul>
    </details>`;

  document.getElementById("info-cartao")
    .querySelectorAll(".chip-material")
    .forEach(chip => chip.addEventListener("click", () => {
      item.materialSelecionado = chip.dataset.chave;
      document.querySelectorAll(".chip-material").forEach(c =>
        c.classList.toggle("selecionado", c.dataset.chave === item.materialSelecionado));
    }));
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

  // Material confirmado: nos swipes de DESPERDÍCIO vai o material do seletor
  // (dominante detectado, ou o que o operador corrigiu); cenas limpas não têm.
  const material = rotulo === 1 ? (item.materialSelecionado || null) : null;
  const nomeMaterial = material
    ? (CATALOGO_MATERIAIS.find(m => m.chave === material) || {}).nome || material
    : null;

  api("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ caminho: item.caminho, rotulo, material }),
  }).then(r => {
    atualizarChip(r.status);
    const base = r.acertou
      ? `✔ Experiência #${r.exp_id} — o sistema havia acertado`
      : `✏ Experiência #${r.exp_id} — memória CORRIGIDA pelo operador`;
    toast(nomeMaterial ? `${base} · ${nomeMaterial}` : base);
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

/* ---------------- login (Engenhender) ---------------- */
async function entrar() {
  const nome = el.inputNome().value.trim();
  if (!nome) { el.inputNome().focus(); return; }
  try {
    await api("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nome }),
    });
    operador = nome;
    el.nomeOperador().textContent = nome.split(" ")[0];
    el.login().hidden = true;
    el.cabecalho().hidden = false;
    el.faixaTopo().hidden = false;
    el.barra().hidden = false;
    el.entrada().hidden = false;
    api("/api/status").then(atualizarChip).catch(() => {});
    if (CATALOGO_MATERIAIS.length === 0) {
      api("/api/materiais").then(lista => { CATALOGO_MATERIAIS = lista; })
        .catch(() => {});
    }
    toast(`bem-vindo(a) à obra, ${nome.split(" ")[0]}!`);
  } catch (e) { toast(e.message, true); }
}

async function sair() {
  try { await api("/api/sair", { method: "POST" }); } catch (e) { /* sessão local de qualquer forma */ }
  operador = null;
  fila = []; indice = 0; registradasSessao = 0;
  el.jogo().hidden = true; el.fim().hidden = true; el.entrada().hidden = true;
  el.barra().hidden = true; el.faixaTopo().hidden = true; el.cabecalho().hidden = true;
  el.login().hidden = false;
  el.inputNome().value = "";
  el.inputNome().focus();
}

/* ---------------- câmera ---------------- */
async function abrirCamera() {
  el.modalCamera().hidden = false;
  el.btnCapturar().disabled = true;
  el.cameraDica().textContent = "abrindo a câmera…";
  try {
    fluxoCamera = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } },
      audio: false,
    });
    el.video().srcObject = fluxoCamera;
    el.btnCapturar().disabled = false;
    el.cameraDica().textContent = "aponte para a área da obra (padrão de 5 m) e capture";
  } catch (erro) {
    el.cameraDica().textContent =
      "não foi possível abrir a câmera — verifique permissões ou use outro navegador";
  }
}

function fecharCamera() {
  if (fluxoCamera) { fluxoCamera.getTracks().forEach(t => t.stop()); fluxoCamera = null; }
  el.modalCamera().hidden = true;
}

async function capturar() {
  const video = el.video();
  if (!video.videoWidth) return;
  const tela = document.createElement("canvas");
  tela.width = video.videoWidth;
  tela.height = video.videoHeight;
  const ctx = tela.getContext("2d");
  ctx.translate(tela.width, 0);        // grava espelhado, igual ao preview
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  const quadro = tela.toDataURL("image/jpeg", 0.92);
  try {
    const r = await api("/api/captura", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quadro }),
    });
    fecharCamera();
    toast("foto capturada — analisando…");
    iniciarFila(r.adicionadas);
  } catch (e) { toast("falha na captura: " + e.message, true); }
}

/* ---------------- limpar a inteligência ---------------- */
async function abrirLimpar() {
  try {
    const s = await api("/api/status");
    el.limparResumo().textContent =
      `memória atual: ${s.total} experiência(s) (${s.desperdicio} desperdício / ` +
      `${s.limpo} limpo)` + (s.taxa_acerto === null ? "" :
        ` · taxa de acerto até aqui: ${Math.round(s.taxa_acerto * 100)}%`);
  } catch (e) { el.limparResumo().textContent = ""; }
  el.btnLimparConfirmar().textContent = "apagar tudo";
  el.btnLimparConfirmar().dataset.armado = "";
  el.modalLimpar().hidden = false;
}

async function confirmarLimpar() {
  const botao = el.btnLimparConfirmar();
  if (!botao.dataset.armado) {          // dupla confirmação
    botao.dataset.armado = "1";
    botao.textContent = "tenho certeza — APAGAR";
    return;
  }
  try {
    const r = await api("/api/limpar_inteligencia", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmar: "APAGAR" }),
    });
    el.modalLimpar().hidden = true;
    atualizarChip(r.status);
    fila = []; indice = 0; registradasSessao = 0;
    el.jogo().hidden = true; el.fim().hidden = true; el.entrada().hidden = false;
    toast("memória e modelo apagados — o sistema recomeçou do zero");
  } catch (e) { toast(e.message, true); }
}

/* ---------------- ligações da página ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  // login / sair
  el.btnEntrar().addEventListener("click", entrar);
  el.inputNome().addEventListener("keydown", (e) => { if (e.key === "Enter") entrar(); });
  el.inputNome().focus();
  el.btnSair().addEventListener("click", sair);

  // câmera
  el.btnCamera().addEventListener("click", abrirCamera);
  el.btnCapturar().addEventListener("click", capturar);
  el.btnFecharCamera().addEventListener("click", fecharCamera);

  // limpar IA
  el.btnLimpar().addEventListener("click", abrirLimpar);
  el.btnLimparCancelar().addEventListener("click", () => { el.modalLimpar().hidden = true; });
  el.btnLimparConfirmar().addEventListener("click", confirmarLimpar);

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
    if (!el.login().hidden || !el.modalCamera().hidden || !el.modalLimpar().hidden) return;
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
});
