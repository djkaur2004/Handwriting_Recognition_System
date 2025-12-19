function sendMessage() {
  const userInput = document.getElementById("userInput");
  const message = userInput.value.trim();
  if (message === "") return;
  appendMessage("You", message);
  userInput.value = "";
  showTyping();
  setTimeout(() => {
    const response = getBotResponse(message);
    removeTyping();
    appendMessage("Bot", response);
  }, 800);
}
function appendMessage(sender, message) {
  const chatlog = document.getElementById("chatlog");
  const msgDiv = document.createElement("div");
  msgDiv.className = sender === "You" ? "msg user" : "msg bot";
  const avatar = sender === "You" ? "👤" : "🤖";
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  msgDiv.innerHTML = `
      <div class="bubble">
        <span class="avatar">${avatar}</span>
        <div class="text">
          <p>${message}</p>
          <span class="timestamp">${time}</span>
        </div>
      </div>`;

  chatlog.appendChild(msgDiv);
  chatlog.scrollTop = chatlog.scrollHeight;
}
function showTyping() {
  const chatlog = document.getElementById("chatlog");
  const typingDiv = document.createElement("div");
  typingDiv.id = "typing";
  typingDiv.className = "msg bot";
  typingDiv.innerHTML = `
      <div class="bubble">
        <span class="avatar">🤖</span>
        <div class="text"><p><em>Typing...</em></p></div>
      </div>`;
  chatlog.appendChild(typingDiv);
  chatlog.scrollTop = chatlog.scrollHeight;
}
function removeTyping() {
  const typing = document.getElementById("typing");
  if (typing) typing.remove();
}
function getBotResponse(input) {
  const lower = input.toLowerCase();
  if (/^(hello|hi|hey)\b/.test(lower)) {
    return "Hello! 👋 I can help you recognize handwritten text. How can I assist you today?";
  }

  if (lower.includes("what") && lower.includes("handwriting")) {
    return "Handwriting recognition is a technology that converts handwritten text (from images or scanned documents) into digital text using AI and machine learning.";
  }

  if (lower.includes("how") && lower.includes("work")) {
    return "✍️ The system works by preprocessing the image, extracting features, and using trained machine learning models to recognize characters and words.";
  }

  if (lower.includes("upload") || lower.includes("image")) {
    return "📸 Please upload an image of handwritten text. Make sure it is clear and well-lit for better accuracy.";
  }

  if (lower.includes("supported") || lower.includes("language")) {
    return "🌐 Currently, the system supports handwritten English text. More languages will be added soon!";
  }

  if (lower.includes("accuracy") || lower.includes("correct")) {
    return "✅ Accuracy depends on handwriting clarity, image quality, and spacing. Neat handwriting gives the best results.";
  }

  if (lower.includes("convert") || lower.includes("text")) {
    return "📝 Once processed, the handwritten content will be converted into editable digital text.";
  }

  if (lower.includes("download") || lower.includes("save")) {
    return "⬇️ You can download the recognized text as a .txt or .pdf file after processing.";
  }

  else {
    return "I didn’t understand that 🤔. You can ask about uploading images, supported languages, or how handwriting recognition works.";
  }

}
// Enable Enter key to send message
document.getElementById("userInput").addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});
