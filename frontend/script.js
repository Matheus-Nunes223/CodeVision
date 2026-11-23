(() => {
  "use strict";
 
  /* ================================================================
   * CONFIGURAÇÃO — ajuste aqui se os caminhos do seu backend forem outros
   * ================================================================ */
  const RECOGNITION_PATH = "/api/reconhecimento/"; // POST, multipart, campo "imagem"
  const STATUS_PATH = "/api/status/";              // GET (confira em backend/.../urls.py)
  const FILE_FIELD = "imagem";
  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
  const LOW_CONFIDENCE = 0.6; // abaixo disso o símbolo é destacado como incerto
  const REQUEST_TIMEOUT_MS = 30000;
 
  /* ================================================================
   * Elementos
   * ================================================================ */
  const $ = (id) => document.getElementById(id);
  const el = {
    apiUrl: $("api-url"),
    status: $("status"),
    statusText: $("status-text"),
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    dzEmpty: $("dz-empty"),
    dzPreview: $("dz-preview"),
    previewImg: $("preview-img"),
    fileName: $("file-name"),
    clearBtn: $("clear-btn"),
    submitBtn: $("submit-btn"),
    placeholder: $("placeholder"),
    loading: $("loading"),
    error: $("error"),
    result: $("result"),
    sequence: $("sequence"),
    decoded: $("decoded"),
    symbolsBlock: $("symbols-block"),
    symbols: $("symbols"),
    rawJson: $("raw-json"),
    curl: $("curl-cmd"),
  };
 
  let selectedFile = null;
  let previewUrl = null;
  let statusTimer = null;
 
  const apiBase = () => el.apiUrl.value.trim().replace(/\/+$/, "");
 
  /* ================================================================
   * Status do backend
   * ================================================================ */
  function setStatus(state, text) {
    el.status.dataset.state = state;
    el.statusText.textContent = text;
  }
 
  async function checkBackend() {
    setStatus("checking", "Verificando o backend…");
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 4000);
    try {
      const res = await fetch(apiBase() + STATUS_PATH, { signal: ctrl.signal });
      if (res.status === 503) {
        setStatus("no-model", "Backend online · modelo não treinado");
        return;
      }
      if (res.ok) {
        let data = null;
        try { data = await res.json(); } catch (_) { /* resposta não-JSON */ }
        const flag = data && pick(data, [
          "modelo_disponivel", "modelo_carregado", "model_loaded", "model_available", "disponivel",
        ]);
        if (flag === false) {
          setStatus("no-model", "Backend online · modelo não treinado");
          return;
        }
      }
      // 200, 404 etc.: o servidor respondeu, então está no ar.
      setStatus("online", "Backend online");
    } catch (_) {
      setStatus("offline", "Backend offline — rode python manage.py runserver");
    } finally {
      clearTimeout(timer);
    }
  }
 
  function scheduleStatusCheck() {
    clearTimeout(statusTimer);
    statusTimer = setTimeout(checkBackend, 400);
  }
 
  /* ================================================================
   * Seleção de arquivo
   * ================================================================ */
  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = false;
    el.placeholder.hidden = true;
    el.loading.hidden = true;
    el.result.hidden = true;
  }
 
  function resetOutput() {
    el.error.hidden = true;
    el.result.hidden = true;
    el.loading.hidden = true;
    el.placeholder.hidden = false;
  }
 
  function validate(file) {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Formato não suportado. Use PNG, JPEG, WEBP ou BMP.";
    }
    if (file.size > MAX_BYTES) {
      return "A imagem passa de 10 MB. Escolha um arquivo menor.";
    }
    return null;
  }
 
  function setFile(file) {
    const problem = validate(file);
    if (problem) {
      clearFile();
      showError(problem);
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    el.previewImg.src = previewUrl;
    el.fileName.textContent = file.name;
    el.dzEmpty.hidden = true;
    el.dzPreview.hidden = false;
    el.submitBtn.disabled = false;
    resetOutput();
    updateCurl();
  }
 
  function clearFile() {
    selectedFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    el.fileInput.value = "";
    el.previewImg.removeAttribute("src");
    el.dzPreview.hidden = true;
    el.dzEmpty.hidden = false;
    el.submitBtn.disabled = true;
    updateCurl();
  }
 
  el.dropzone.addEventListener("click", (e) => {
    if (e.target === el.clearBtn) return;
    el.fileInput.click();
  });
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.fileInput.click();
    }
  });
  el.clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
    resetOutput();
  });
  el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files[0]) setFile(el.fileInput.files[0]);
  });
 
  ["dragenter", "dragover"].forEach((type) =>
    el.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    el.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("is-over");
    })
  );
  el.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
 
  /* ================================================================
   * Envio para a API
   * ================================================================ */
  async function recognize() {
    if (!selectedFile) return;
 
    el.submitBtn.disabled = true;
    el.error.hidden = true;
    el.result.hidden = true;
    el.placeholder.hidden = true;
    el.loading.hidden = false;
 
    const body = new FormData();
    body.append(FILE_FIELD, selectedFile);
 
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
 
    try {
      const res = await fetch(apiBase() + RECOGNITION_PATH, {
        method: "POST",
        body,
        signal: ctrl.signal,
      });
 
      let data = null;
      const text = await res.text();
      try { data = text ? JSON.parse(text) : null; } catch (_) { /* não-JSON */ }
 
      if (!res.ok) {
        showError(httpErrorMessage(res.status, data));
        return;
      }
      render(data ?? { resposta: text });
    } catch (err) {
      if (err.name === "AbortError") {
        showError("A API demorou demais para responder. Tente novamente.");
      } else {
        showError(
          "Não foi possível falar com o backend. Confira se o servidor está rodando " +
          "e se a página foi aberta por http://127.0.0.1:5173 (não por file://), " +
          "por causa do CORS."
        );
        setStatus("offline", "Backend offline — rode python manage.py runserver");
      }
    } finally {
      clearTimeout(timer);
      el.loading.hidden = true;
      el.submitBtn.disabled = !selectedFile;
    }
  }
 
  function httpErrorMessage(status, data) {
    const detail = data && pick(data, ["erro", "error", "detail", "detalhe", "mensagem", "message"]);
    if (status === 503) {
      return "O modelo ainda não foi treinado. Rode python -m ia.treinamento na raiz do projeto e tente de novo.";
    }
    if (status === 413) return "A imagem é grande demais para a API.";
    if (detail) return `A API recusou o pedido (${status}): ${detail}`;
    return `A API respondeu com erro ${status}.`;
  }
 
  el.submitBtn.addEventListener("click", recognize);
 
  /* ================================================================
   * Renderização — tolera nomes de campos diferentes
   * ================================================================ */
  function pick(obj, keys) {
    for (const k of keys) {
      if (obj && Object.prototype.hasOwnProperty.call(obj, k) && obj[k] !== null && obj[k] !== undefined) {
        return obj[k];
      }
    }
    return undefined;
  }
 
  function normalizeSymbols(list) {
    if (!Array.isArray(list)) return [];
    return list.map((item) => {
      if (typeof item === "string") return { symbol: item, confidence: null };
      const symbol = pick(item, ["simbolo", "symbol", "classe", "class", "label", "rotulo"]);
      let conf = pick(item, ["confianca", "confidence", "probabilidade", "probability", "score"]);
      if (typeof conf === "number" && conf > 1) conf = conf / 100; // aceita 0–100
      return { symbol: String(symbol ?? "?"), confidence: typeof conf === "number" ? conf : null };
    });
  }
 
  function render(data) {
    el.rawJson.textContent = JSON.stringify(data, null, 2);
 
    const symbols = normalizeSymbols(
      pick(data, ["simbolos", "symbols", "resultados", "predicoes", "predictions", "deteccoes"])
    );
    const sequence =
      pick(data, ["sequencia", "sequence", "morse", "codigo", "code"]) ??
      (symbols.length ? symbols.map((s) => s.symbol).join("") : "");
    const decoded = pick(data, ["texto", "mensagem", "message", "text", "texto_decodificado", "decodificado"]);
 
    el.sequence.textContent = sequence || "(nenhum símbolo reconhecido)";
 
    if (decoded) {
      el.decoded.textContent = String(decoded);
      el.decoded.hidden = false;
    } else {
      el.decoded.hidden = true;
    }
 
    el.symbols.replaceChildren();
    symbols.forEach((s) => {
      const li = document.createElement("li");
      const sym = document.createElement("span");
      sym.className = "sym";
      sym.textContent = s.symbol;
      li.append(sym);
 
      if (s.confidence !== null) {
        const pct = document.createElement("span");
        pct.className = "pct";
        pct.textContent = `${(s.confidence * 100).toFixed(1)}%`;
        const bar = document.createElement("span");
        bar.className = "bar";
        const fill = document.createElement("i");
        fill.style.width = `${Math.max(0, Math.min(1, s.confidence)) * 100}%`;
        bar.append(fill);
        li.append(pct, bar);
        li.dataset.low = String(s.confidence < LOW_CONFIDENCE);
      }
      el.symbols.append(li);
    });
    el.symbolsBlock.hidden = symbols.length === 0;
 
    el.placeholder.hidden = true;
    el.result.hidden = false;
  }
 
  /* ================================================================
   * Bloco do curl (acompanha a URL da API e o arquivo escolhido)
   * ================================================================ */
  function updateCurl() {
    const name = selectedFile ? selectedFile.name : "mensagem.png";
    el.curl.textContent =
      `curl -X POST ${apiBase()}${RECOGNITION_PATH} \\\n` +
      `     -F "${FILE_FIELD}=@${name}"`;
  }
 
  el.apiUrl.addEventListener("input", () => {
    updateCurl();
    scheduleStatusCheck();
  });
 
  /* ================================================================
   * Início
   * ================================================================ */
  updateCurl();
  checkBackend();
})();
 