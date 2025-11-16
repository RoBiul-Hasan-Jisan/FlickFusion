let isProcessing = false;

// Initialize the chat when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
});

function initializeChat() {
    // Add event listeners
    document.getElementById('sendButton').addEventListener('click', sendMessage);
    
    document.getElementById('userInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Add quick action button listeners
    const quickButtons = document.querySelectorAll('.quick-btn');
    quickButtons.forEach(button => {
        button.addEventListener('click', function() {
            const message = this.getAttribute('data-message');
            sendQuickMessage(message);
        });
    });

    // Initial bot message after a short delay
    setTimeout(() => {
        addMessage("Hello! I'm your Movie Expert AI \n\nI can help you find perfect movies for any mood, genre, or occasion! Try:\n• Clicking the quick action buttons\n• Typing like 'action movies' or 'I'm feeling sad'\n• Asking for specific genres like comedy, romance, horror\n• Or just chat naturally!\n\nWhat would you like to watch today? ");
    }, 1000);
}

function addMessage(message, isUser = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.innerHTML = formatMessage(message);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(text) {
    // Convert markdown-like formatting to HTML
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>')
        .replace(/(🎬|🍿|😊|😢|💖|🎯|😴|👻|🧙|💥|👋|🔍|❤️|😂|🚀|👽|🎃|🌹|🎭|🏠|📚|🧘|⚡|🌿|💑|👥|🏆|📖|⏰|📅)/g, '<span class="emoji">$1</span>');
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = 'Thinking<span class="typing-dots"></span>';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

async function sendMessage() {
    if (isProcessing) return;
    
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, true);
    userInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    // Disable input while processing
    setInputState(true);
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        hideTypingIndicator();
        addMessage(data.response);
    } catch (error) {
        hideTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.');
        console.error('Error:', error);
    } finally {
        setInputState(false);
    }
}

function sendQuickMessage(message) {
    if (isProcessing) return;
    
    // Add user message immediately for quick feedback
    addMessage(message, true);
    
    // Show typing indicator
    showTypingIndicator();
    
    // Disable input while processing
    setInputState(true);
    
    // Send to server
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        hideTypingIndicator();
        addMessage(data.response);
    })
    .catch(error => {
        hideTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.');
        console.error('Error:', error);
    })
    .finally(() => {
        setInputState(false);
    });
}

function setInputState(disabled) {
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const quickButtons = document.querySelectorAll('.quick-btn');
    
    isProcessing = disabled;
    userInput.disabled = disabled;
    sendButton.disabled = disabled;
    
    quickButtons.forEach(button => {
        button.disabled = disabled;
    });
    
    if (!disabled) {
        userInput.focus();
    }
}