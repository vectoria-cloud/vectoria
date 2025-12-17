function createIcon(src, className) {
  const img = document.createElement("img");
  img.src = src;
  img.className = className;
  img.alt = "";

  // força tamanho (mesmo que exista algum CSS global de img)
  img.width = 24;
  img.height = 24;
  img.style.width = "24px";
  img.style.height = "24px";
  img.style.maxWidth = "24px";
  img.style.maxHeight = "24px";
  img.style.borderRadius = "50%";
  img.style.objectFit = "cover";
  img.style.flex = "0 0 24px";

  return img;
}

function appendContent(container, content) {
  const text = String(content);

  // suporta ```codigo```
  if (!text.includes("```")) {
    const p = document.createElement("p");
    p.textContent = text;
    container.appendChild(p);
    return;
  }

  const parts = text.split("```");
  parts.forEach((part, i) => {
    const t = part.trim();
    if (!t) return;

    if (i % 2 === 0) {
      const p = document.createElement("p");
      p.textContent = t;
      container.appendChild(p);
    } else {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = t;
      pre.appendChild(code);
      container.appendChild(pre);
    }
  });
}

function addMessage(message, isUserMessage) {
  const chatBox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");

  // sem bootstrap aqui
  messageDiv.className = isUserMessage ? "user-message" : "bot-message";

  const icon = isUserMessage
    ? createIcon("https://cdn-icons-png.flaticon.com/512/17/17004.png", "user-icon")
    : createIcon("/static/imagens/gpt.jpg", "bot-icon");

  const contentWrap = document.createElement("div");
  contentWrap.className = "msg-content";
  appendContent(contentWrap, message);

  // IMPORTANTE:
  // bot-message já está row-reverse no seu CSS, então o ícone vai pra direita.
  messageDiv.appendChild(icon);
  messageDiv.appendChild(contentWrap);

  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message-input");
  const message = messageInput.value.trim();
  const btnSubmit = document.getElementById("send-btn");

  messageInput.value = "";

  if (message !== "") {
    btnSubmit.disabled = true;
    btnSubmit.style.cursor = "not-allowed";
    messageInput.disabled = true;
    messageInput.value = "Carregando...";

    addMessage(message, true);

    fetch("/api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    })
      .then((response) => response.json())
      .then((data) => {
        // cria msg do bot SEM bootstrap
        const messageDiv = document.createElement("div");
        messageDiv.className = "bot-message bot-div";

        const icon = createIcon("/static/imagens/gpt.jpg", "bot-icon");
        const contentWrap = document.createElement("div");
        contentWrap.className = "msg-content";

        appendContent(contentWrap, data.content);

        messageDiv.appendChild(icon);
        messageDiv.appendChild(contentWrap);

        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
      })
      .catch((e) => console.log(`Error -> ${e}`))
      .finally(() => {
        btnSubmit.disabled = false;
        btnSubmit.style.cursor = "pointer";
        messageInput.disabled = false;
        messageInput.value = "";
      });
  }
}
