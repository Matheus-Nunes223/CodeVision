(() => {
  "use strict";

  const TAMANHO_MAXIMO = 10 * 1024 * 1024;
  const TIPOS_ACEITOS = ["image/png", "image/jpeg", "image/webp", "image/bmp"];

  const $ = (id) => document.getElementById(id);

  const el = {
    status: $("status"),
    statusText: $("status-text"),
    apiUrl: $("api-url"),
    form: $("form"),
    file: $("file"),
    dropZone: $("drop-zone"),
    dropText: $("drop-text"),
    preview: $("preview"),
    submit: $("submit"),
    empty: $("result-empty"),
    content: $("result-content"),
    meta: $("result-meta"),
    seq: $("result-seq"),
    symbols: $("result-symbols"),
    json: $("result-json"),
  };

  let arquivoAtual = null;
  let previewUrl = null;

  const baseApi = () => el.apiUrl.value.trim().replace(/\/+$/, "");

  function definirStatus(estado, texto) {
    el.status.dataset.state = estado;
    el.statusText.textContent = texto;
  }

  async function verificarBackend() {
    definirStatus("checking", "verificando backend…");

    try {
      const resposta = await fetch(`${baseApi()}/api/reconhecimento/status/`, {
        signal: AbortSignal.timeout(4000),
      });
      const dados = await resposta.json();

      if (!resposta.ok || dados.status !== "online") {
        definirStatus("offline", "backend respondeu com erro");
      } else if (!dados.modelo_disponivel) {
        definirStatus("partial", "backend online · modelo ainda não treinado");
      } else {
        definirStatus(
          "online",
          dados.modelo_carregado ? "backend online · modelo em memória" : "backend online · modelo disponível"
        );
      }
    } catch {
      definirStatus("offline", "backend offline — rode python manage.py runserver");
    }
  }

  function validarArquivo(arquivo) {
    if (!TIPOS_ACEITOS.includes(arquivo.type)) {
      return "Formato não aceito. Use PNG, JPEG, WEBP ou BMP.";
    }
    if (arquivo.size > TAMANHO_MAXIMO) {
      return "A imagem ultrapassa o limite de 10 MB.";
    }
    return null;
  }

  function selecionarArquivo(arquivo) {
    if (!arquivo) return;

    const erro = validarArquivo(arquivo);
    if (erro) {
      mostrarErro(erro);
      return;
    }

    arquivoAtual = arquivo;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(arquivo);

    el.preview.src = previewUrl;
    el.preview.alt = `Pré-visualização de ${arquivo.name}`;
    el.preview.hidden = false;
    el.dropText.hidden = true;
    el.submit.disabled = false;
  }

  function limparResultado() {
    el.meta.replaceChildren();
    el.seq.replaceChildren();
    el.symbols.replaceChildren();
    el.json.textContent = "";
  }

  function mostrarErro(mensagem, corpo) {
    limparResultado();
    el.empty.hidden = true;
    el.content.hidden = false;

    const p = document.createElement("p");
    p.className = "result__msg result__msg--error";
    p.textContent = mensagem;
    el.seq.appendChild(p);

    el.json.textContent = corpo ? JSON.stringify(corpo, null, 2) : "";
    el.json.parentElement.hidden = !corpo;
  }

  function adicionarMeta(texto) {
    const span = document.createElement("span");
    span.textContent = texto;
    el.meta.appendChild(span);
  }

  function formatarConfianca(valor) {
    return typeof valor === "number" ? `${(valor * 100).toFixed(1)}%` : "—";
  }

  function mostrarResultado(dados, statusHttp) {
    limparResultado();
    el.empty.hidden = true;
    el.content.hidden = false;
    el.json.parentElement.hidden = false;

    adicionarMeta(`HTTP ${statusHttp}`);
    adicionarMeta(`${dados.quantidade_simbolos ?? 0} símbolo(s)`);
    if (typeof dados.tempo_processamento_ms === "number") {
      adicionarMeta(`${dados.tempo_processamento_ms.toFixed(1)} ms`);
    }

    if (dados.texto_reconhecido) {
      el.seq.textContent = dados.texto_reconhecido;
    } else {
      const p = document.createElement("p");
      p.className = "result__msg";
      p.textContent = dados.mensagem || "Nenhum símbolo reconhecido.";
      el.seq.appendChild(p);
    }

    (dados.simbolos || []).forEach((s, i) => {
      const li = document.createElement("li");

      const sim = document.createElement("span");
      sim.textContent = s.simbolo ?? "?";

      const bar = document.createElement("span");
      bar.className = "bar";
      bar.style.setProperty("--v", typeof s.confianca === "number" ? s.confianca : 0);
      bar.title = `símbolo ${i + 1}`;

      const conf = document.createElement("span");
      conf.textContent = formatarConfianca(s.confianca);

      li.append(sim, bar, conf);
      el.symbols.appendChild(li);
    });

    el.json.textContent = JSON.stringify(dados, null, 2);
  }

  async function enviar(evento) {
    evento.preventDefault();
    if (!arquivoAtual) return;

    const formData = new FormData();
    formData.append("imagem", arquivoAtual);

    el.submit.disabled = true;
    el.submit.textContent = "Reconhecendo…";

    try {
      const resposta = await fetch(`${baseApi()}/api/reconhecimento/`, {
        method: "POST",
        body: formData,
      });

      let dados = null;
      try {
        dados = await resposta.json();
      } catch {
        dados = null;
      }

      if (resposta.ok && dados) {
        mostrarResultado(dados, resposta.status);
      } else {
        const mensagem = dados?.mensagem || `O servidor respondeu com HTTP ${resposta.status}.`;
        mostrarErro(`${resposta.status} · ${mensagem}`, dados);
      }
    } catch {
      mostrarErro(
        "Não foi possível alcançar a API. Verifique se o backend está rodando e se esta origem está em DJANGO_CORS_ALLOWED_ORIGINS."
      );
    } finally {
      el.submit.disabled = false;
      el.submit.textContent = "Reconhecer símbolos";
      verificarBackend();
    }
  }

  el.file.addEventListener("change", () => selecionarArquivo(el.file.files[0]));

  ["dragenter", "dragover"].forEach((tipo) =>
    el.dropZone.addEventListener(tipo, (e) => {
      e.preventDefault();
      el.dropZone.classList.add("is-over");
    })
  );

  ["dragleave", "drop"].forEach((tipo) =>
    el.dropZone.addEventListener(tipo, (e) => {
      e.preventDefault();
      el.dropZone.classList.remove("is-over");
    })
  );

  el.dropZone.addEventListener("drop", (e) => selecionarArquivo(e.dataTransfer.files[0]));

  el.form.addEventListener("submit", enviar);
  el.apiUrl.addEventListener("change", verificarBackend);

  verificarBackend();
})();
