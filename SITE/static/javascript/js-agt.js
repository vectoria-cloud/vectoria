function addMessage(message, isUserMessage) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("mt-3", "p-3", "rounded");

    if (isUserMessage) {
        messageDiv.classList.add("user-message");

    } else {
        messageDiv.classList.add("bot-message");
    }

    messageDiv.innerHTML = `
        <img src="https://cdn-icons-png.flaticon.com/512/17/17004.png" class="user-icon"><p>${message}</p>
        `;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}


function sendMessage() {
    const chatBox = document.getElementById("chat-box");
    const messageInput = document.getElementById("message-input");
    const mensagem = messageInput.value.trim();
    const message = mensagem;
    message.value = '';
    const btnSubmit = document.getElementById('send-btn')

    
    if (message !== "") {
        btnSubmit.disabled = true
        btnSubmit.style.cursor = 'not-allowed'
        messageInput.disabled= true;
        messageInput.value = "Carregando..."


        addMessage(message, true);
        fetch("/api", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        })
            .then(response => response.json())
            .then(data => {
                messageInput.value = "";
                const messageDiv = document.createElement("div");
                messageDiv.classList.add("mt-3", "p-3", "rounded");
                messageDiv.classList.add("bot-message","bot-div");

                const content = data.content;

                const hasCodeBlock = content.includes("```");
                if (hasCodeBlock) {
                    const codeContent = content.replace(/```([\s\S]+?)```/g, '</p><pre><code>$1</code></pre><p>');
                    messageDiv.innerHTML = `<img src="https://play-lh.googleusercontent.com/8XCwpfWc9YkehwhrhoID6PGhs5SaSJoocS0oTBA8EsGFGLrj32oIYu5UKsIO7wdU1PQZ" class="bot-icon"><p>${codeContent}</p>`

                }
                else {
                    messageDiv.innerHTML = `<img src="https://play-lh.googleusercontent.com/8XCwpfWc9YkehwhrhoID6PGhs5SaSJoocS0oTBA8EsGFGLrj32oIYu5UKsIO7wdU1PQZ" class="bot-icon"><p>${content}</p>`
                }
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

            })
            .catch((e) => {
                console.log(`Error -> ${e}`)
            })
            .finally(() => {
                btnSubmit.disabled = false;
                btnSubmit.style.cursor = 'pointer';
                messageInput.disabled= false;
                message.value = '';
            })
            
            
    }
    
    
}