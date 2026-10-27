const form = document.querySelector("#chat-form");
const input = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-button");
const messagesElement = document.querySelector("#messages");
const examples = document.querySelectorAll(".example");

const messages = [];

function appendMessage(role, content = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const roleElement = document.createElement("div");
  roleElement.className = "role";
  roleElement.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  wrapper.append(roleElement, bubble);
  messagesElement.append(wrapper);
  messagesElement.scrollTop = messagesElement.scrollHeight;
  return bubble;
}

async function sendMessage(content) {
  messages.push({ role: "user", content });
  appendMessage("user", content);

  const assistantBubble = appendMessage("assistant", "");
  sendButton.disabled = true;
  input.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let assistantText = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const event of events) {
        const line = event.split("\n").find((item) => item.startsWith("data: "));
        if (!line) continue;

        const payload = JSON.parse(line.slice(6));
        if (payload.type === "content") {
          assistantText += payload.content;
          assistantBubble.innerHTML = marked.parse(assistantText);
          messagesElement.scrollTop = messagesElement.scrollHeight;
        }
      }
    }

    messages.push({ role: "assistant", content: assistantText });
  } catch (error) {
    assistantBubble.textContent = `Error: ${error.message}`;
  } finally {
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content) return;

  input.value = "";
  await sendMessage(content);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

for (const example of examples) {
  example.addEventListener("click", () => {
    input.value = example.textContent.trim();
    input.focus();
  });
}
