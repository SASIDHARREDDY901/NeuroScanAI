const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const fileSelect = document.getElementById('fileSelect');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultCard = document.getElementById('resultCard');
const resetBtn = document.getElementById('resetBtn');
const predClass = document.getElementById('predClass');
const predConf = document.getElementById('predConf');
const confidenceBar = document.getElementById('confidenceBar');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

// Handle browse button click
fileSelect.addEventListener('click', () => {
    fileElem.click();
});

fileElem.addEventListener('change', function () {
    handleFiles(this.files);
});

resetBtn.addEventListener('click', resetUI);

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    dropArea.classList.add('dragover');
}

function unhighlight(e) {
    dropArea.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            uploadFile(file);
            previewFile(file);
        } else {
            alert('Please upload an image file (JPG, PNG).');
        }
    }
}

function previewFile(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function () {
        imagePreview.src = reader.result;
        previewArea.classList.remove('hidden');
        dropArea.classList.add('hidden');
    }
}

function uploadFile(file) {
    loadingOverlay.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loadingOverlay.classList.add('hidden');
            if (data.error) {
                alert('Error: ' + data.error);
                resetUI();
            } else {
                showResult(data);
            }
        })
        .catch(() => {
            loadingOverlay.classList.add('hidden');
            alert('An error occurred during prediction.');
            resetUI();
        });
}

function showResult(data) {
    resultCard.classList.remove('hidden');
    predClass.textContent = data.class;
    predConf.textContent = data.confidence;

    // Animate confidence bar
    const confidenceValue = parseFloat(data.confidence);
    setTimeout(() => {
        confidenceBar.style.width = confidenceValue + '%';
    }, 100);
}

function resetUI() {
    resultCard.classList.add('hidden');
    previewArea.classList.add('hidden');
    dropArea.classList.remove('hidden');
    imagePreview.src = '';
    confidenceBar.style.width = '0%';
    fileElem.value = ''; // Reset file input
}

// Chatbot Logic
const chatFab = document.getElementById('chatFab');
const chatWindow = document.getElementById('chatWindow');
const chatClose = document.getElementById('chatClose');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');
const chatMessages = document.getElementById('chatMessages');

chatFab.addEventListener('click', () => {
    chatWindow.classList.remove('hidden');
    chatFab.classList.add('hidden');
});

chatClose.addEventListener('click', () => {
    chatWindow.classList.add('hidden');
    chatFab.classList.remove('hidden');
});

const chatFileInput = document.getElementById('chatFileInput');

chatFileInput.addEventListener('change', function () {
    if (this.files.length > 0) {
        const file = this.files[0];

        // Show preview immediately
        const reader = new FileReader();
        reader.onload = function (e) {
            addImageMessage(e.target.result, 'user-message');
            // Send automatically after selection
            sendMessage(file);
        };
        reader.readAsDataURL(file);
    }
});

chatSendBtn.addEventListener('click', () => sendMessage());

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage(file = null) {
    const text = chatInput.value.trim();

    if (!text && !file) return;

    const formData = new FormData();

    if (text) {
        if (!file) addMessage(text, 'user-message'); // Only add text bubble if not uploading image (image bubble added separately)
        formData.append('message', text);
        chatInput.value = '';
    }

    if (file) {
        formData.append('file', file);
        // Clear input after sending
        chatFileInput.value = '';
    }

    // Send to backend
    fetch('/chat', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // Format response to handle newlines
            const formattedResponse = data.response.replace(/\n/g, '<br>');
            addHtmlMessage(formattedResponse, 'bot-message');
        })
        .catch(() => {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot-message');
        });
}

function addMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addHtmlMessage(html, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addImageMessage(src, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);

    const img = document.createElement('img');
    img.src = src;
    img.classList.add('chat-uploaded-image');

    messageDiv.appendChild(img);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
