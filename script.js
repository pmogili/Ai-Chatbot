/**
 * script.js
 * ----------
 * Vanilla JS logic for the chatbot frontend:
 *   - Sends messages to POST /chat
 *   - Renders bot/user chat bubbles with timestamps
 *   - Shows a typing indicator + loading spinner while waiting
 *   - Persists session_id in localStorage across page reloads
 *   - Dark mode toggle
 *   - "Clear" button wipes history via DELETE /history/{session_id}
 */

const chatWindow = document.getElementById("chatWindow");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const sendSpinner = document.getElementById("sendSpinner");
const sendLabel = document.getElementById("sendLabel");
const typingIndicator = document.getElementById("typingIndicator");
const darkModeToggle = document.getElementById("darkModeToggle");
const clearBtn = document.getElementById("clearBtn");

const SESSION_KEY = "chatbot_session_id";
const THEME_KEY = "chatbot_theme";

let sessionId = localStorage.getItem(SESSION_KEY) || null;
let isSending = false;

/** Format the current time as HH:MM. */
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Append a chat bubble to the window and scroll to the bottom. */
function appendMessage(text, sender) {
  const messageEl = document.createElement("div");
  messageEl.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  const timestamp = document.createElement("div");
  timestamp.className = "timestamp";
  timestamp.textContent = formatTime(new Date());
  bubble.appendChild(timestamp);

  messageEl.appendChild(bubble);
  chatWindow.appendChild(messageEl);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/** Toggle the visual "typing..." indicator and send-button loading state. */
function setLoading(loading) {
  isSending = loading;
  typingIndicator.hidden = !loading;
  sendBtn.disabled = loading;
  sendSpinner.hidden = !loading;
  sendLabel.textContent = loading ? "Sending" : "Send";
  if (loading) {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
}

/** Send the current input value to the backend and render the reply. */
async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isSending) return;

  appendMessage(text, "user");
  messageInput.value = "";
  setLoading(true);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);

    appendMessage(data.response, "bot");
  } catch (err) {
    console.error("Chat error:", err);
    appendMessage(
      "Sorry, something went wrong while reaching the server. Please try again.",
      "bot"
    );
  } finally {
    setLoading(false);
  }
}

/** Clear the current session's history, both server-side and in the UI. */
async function clearConversation() {
  if (!sessionId) {
    chatWindow.querySelectorAll(".message").forEach((el, i) => (i > 0 ? el.remove() : null));
    return;
  }
  try {
    await fetch(`/history/${sessionId}`, { method: "DELETE" });
  } catch (err) {
    console.error("Failed to clear history:", err);
  }
  chatWindow.innerHTML = "";
  appendMessage("Conversation cleared. How can I help you now?", "bot");
}

/** Apply and persist the chosen theme ('light' or 'dark'). */
function applyTheme(theme) {
  document.body.setAttribute("data-theme", theme);
  darkModeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
  localStorage.setItem(THEME_KEY, theme);
}

// --- Event listeners --------------------------------------------------
sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
});

darkModeToggle.addEventListener("click", () => {
  const current = document.body.getAttribute("data-theme") === "dark" ? "dark" : "light";
  applyTheme(current === "dark" ? "light" : "dark");
});

clearBtn.addEventListener("click", clearConversation);

// --- Initialization -----------------------------------------------------
applyTheme(localStorage.getItem(THEME_KEY) || "light");
messageInput.focus();
