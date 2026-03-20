(function () {
  const CHAT_API_URL = "https://web-production-74dd.up.railway.app/chat";

  // --- Inject HTML ---
  const container = document.createElement("div");
  container.innerHTML = `
    <div id="pf-chat-bubble" aria-label="Open chat">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z"/>
      </svg>
    </div>

    <div id="pf-chat-window" class="pf-hidden">
      <div id="pf-chat-header">
        <div id="pf-header-info">
          <div id="pf-avatar">PF</div>
          <div>
            <div id="pf-chat-title">PeakForm Assistant</div>
            <div id="pf-chat-subtitle">Usually replies instantly</div>
          </div>
        </div>
        <button id="pf-close-btn" aria-label="Close chat">✕</button>
      </div>

      <div id="pf-chat-messages">
        <div class="pf-message pf-bot">
          👋 Hi! I'm the PeakForm assistant. Ask me anything about memberships, classes, pricing, or opening times.
        </div>
      </div>

      <div id="pf-chat-input-area">
        <input
          id="pf-chat-input"
          type="text"
          placeholder="Type your question..."
          autocomplete="off"
        />
        <button id="pf-send-btn" aria-label="Send">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
            <path d="M2 21L23 12L2 3V10L17 12L2 14V21Z"/>
          </svg>
        </button>
      </div>
    </div>
  `;
  document.body.appendChild(container);

  // --- State ---
  const sessionId = crypto.randomUUID();
  let isOpen = false;
  let isLoading = false;

  // --- Elements ---
  const bubble = document.getElementById("pf-chat-bubble");
  const chatWindow = document.getElementById("pf-chat-window");
  const closeBtn = document.getElementById("pf-close-btn");
  const input = document.getElementById("pf-chat-input");
  const sendBtn = document.getElementById("pf-send-btn");
  const messages = document.getElementById("pf-chat-messages");

  // --- Toggle ---
  bubble.addEventListener("click", () => {
    isOpen = !isOpen;
    chatWindow.classList.toggle("pf-hidden", !isOpen);
    bubble.style.display = isOpen ? "none" : "flex";
    document.body.style.overflow = isOpen ? "hidden" : "";
  });

  closeBtn.addEventListener("click", () => {
    isOpen = false;
    chatWindow.classList.add("pf-hidden");
    bubble.style.display = "flex";
    document.body.style.overflow = "";
  });

  // --- Send on Enter ---
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !isLoading) sendMessage();
  });

  sendBtn.addEventListener("click", () => {
    if (!isLoading) sendMessage();
  });

  // --- Core ---
  async function sendMessage() {
    const question = input.value.trim();
    if (!question) return;

    appendMessage(question, "user");
    input.value = "";
    isLoading = true;
    sendBtn.disabled = true;

    const typingEl = appendTyping();

    try {
      const res = await fetch(CHAT_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
      });

      const data = await res.json();
      typingEl.remove();
      appendMessage(data.answer, "bot");
    } catch (err) {
      typingEl.remove();
      appendMessage("Sorry, I'm having trouble connecting. Please try again.", "bot");
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      const isMobile = window.innerWidth <= 480;
      if (!isMobile) {
        input.focus();
      }
    }
  }

    function appendMessage(text, sender) {
        const el = document.createElement("div");
        el.classList.add("pf-message", `pf-${sender}`);
        if (sender === "bot") {
          el.innerHTML = marked.parse(text);
        } else {
          el.textContent = text;
        }
        messages.appendChild(el);
        messages.scrollTop = messages.scrollHeight;
        return el;
      }

  function appendTyping() {
    const el = document.createElement("div");
    el.classList.add("pf-message", "pf-bot", "pf-typing");
    el.innerHTML = `<span></span><span></span><span></span>`;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }
})();