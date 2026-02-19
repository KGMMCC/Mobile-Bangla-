// app.js - Professional AI Text Editor Frontend

class ProfessionalTextEditor {
    constructor() {
        this.canvas = null;
        this.textBlocks = [];
        this.selectedText = null;
        this.originalImage = null;
        this.history = [];
        this.historyIndex = -1;
        this.apiUrl = 'http://localhost:8000/api';
        this.isProcessing = false;
        
        this.init();
    }

    async init() {
        this.initCanvas();
        this.initEventListeners();
        this.initDragAndDrop();
        this.loadFromLocalStorage();
        this.showToast('🎨 এডিটর প্রস্তুত', 'success');
    }

    initCanvas() {
        this.canvas = new fabric.Canvas('canvas', {
            width: 800,
            height: 600,
            backgroundColor: '#1a1a2e',
            preserveObjectStacking: true,
            allowTouchScrolling: true
        });

        // Canvas events
        this.canvas.on('object:selected', (e) => this.onObjectSelected(e));
        this.canvas.on('selection:cleared', () => this.onSelectionCleared());
        this.canvas.on('object:modified', () => this.saveToHistory());
    }

    initEventListeners() {
        // Upload buttons
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (e) => {
            this.uploadImage(e.target.files[0]);
        });

        // Style controls
        document.getElementById('fontSelect').addEventListener('change', () => this.updateTextStyle());
        document.getElementById('colorPicker').addEventListener('input', () => this.updateTextStyle());
        document.getElementById('fontSize').addEventListener('input', (e) => {
            document.getElementById('fontSizeValue').textContent = e.target.value + 'px';
            this.updateTextStyle();
        });

        // Effect buttons
        document.getElementById('boldBtn').addEventListener('click', () => this.toggleTextStyle('bold'));
        document.getElementById('italicBtn').addEventListener('click', () => this.toggleTextStyle('italic'));
        document.getElementById('underlineBtn').addEventListener('click', () => this.toggleTextStyle('underline'));

        // Edit panel
        document.getElementById('closeEditPanel').addEventListener('click', () => this.hideEditPanel());
        document.getElementById('cancelEdit').addEventListener('click', () => this.hideEditPanel());
        document.getElementById('saveEdit').addEventListener('click', () => this.saveTextEdit());

        // Action buttons
        document.getElementById('autoEnhanceBtn').addEventListener('click', () => this.autoEnhance());
        document.getElementById('batchEditBtn').addEventListener('click', () => this.openBatchEdit());
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadImage());
        
        // Timeline
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('redoBtn').addEventListener('click', () => this.redo());
        document.getElementById('historySlider').addEventListener('input', (e) => this.jumpToHistory(e.target.value));

        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => this.openSettings());
        document.getElementById('helpBtn').addEventListener('click', () => this.showHelp());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }

    initDragAndDrop() {
        const dropZone = document.getElementById('canvas').parentElement;

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.border = '3px dashed #6C5CE7';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.border = 'none';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.border = 'none';
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.uploadImage(file);
            } else {
                this.showToast('⚠️ শুধু ইমেজ ফাইল গ্রহণযোগ্য', 'error');
            }
        });
    }

    async uploadImage(file) {
        if (!file) return;

        this.showLoading();
        this.hideUploadOverlay();

        try {
            // Read image file
            const reader = new FileReader();
            reader.onload = async (e) => {
                // Load image to canvas
                fabric.Image.fromURL(e.target.result, async (img) => {
                    // Scale image to fit canvas
                    const scale = Math.min(
                        this.canvas.width / img.width,
                        this.canvas.height / img.height,
                        1
                    );

                    img.scale(scale);
                    img.set({
                        left: (this.canvas.width - img.width * scale) / 2,
                        top: (this.canvas.height - img.height * scale) / 2,
                        selectable: false,
                        evented: false,
                        hasControls: false
                    });

                    this.canvas.clear();
                    this.canvas.add(img);
                    this.originalImage = img;
                    
                    // Detect text using API
                    await this.detectText(file);
                    
                    this.saveToHistory();
                    this.hideLoading();
                    this.showToast('✅ টেক্সট ডিটেক্ট করা হয়েছে', 'success');
                });
            };
            
            reader.readAsDataURL(file);
        } catch (error) {
            console.error('Upload error:', error);
            this.hideLoading();
            this.showToast('❌ আপলোড ব্যর্থ হয়েছে', 'error');
        }
    }

    async detectText(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiUrl}/detect-text`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.textBlocks = data.text_blocks;
                this.displayTextBlocks();
                this.highlightTextBlocks();
            }
        } catch (error) {
            console.error('Text detection error:', error);
            this.showToast('❌ টেক্সট ডিটেক্ট করতে সমস্যা হয়েছে', 'error');
        }
    }

    displayTextBlocks() {
        const textList = document.getElementById('textList');
        textList.innerHTML = '';

        this.textBlocks.forEach((block, index) => {
            const item = document.createElement('div');
            item.className = 'text-item';
            item.dataset.index = index;

            // Detect language
            const isBengali = /[\u0980-\u09FF]/.test(block.text);
            const language = isBengali ? '🇧🇩' : '🇬🇧';

            item.innerHTML = `
                <span class="text-preview">${language} ${block.text.substring(0, 30)}${block.text.length > 30 ? '...' : ''}</span>
                <span class="text-confidence">${Math.round(block.confidence * 100)}%</span>
            `;

            item.addEventListener('click', () => this.selectTextBlock(index));

            textList.appendChild(item);
        });
    }

    highlightTextBlocks() {
        // Remove existing highlights
        this.canvas.getObjects().forEach(obj => {
            if (obj.type === 'rect') {
                this.canvas.remove(obj);
            }
        });

        // Add new highlights
        this.textBlocks.forEach(block => {
            const [x1, y1, x2, y2] = block.bbox;
            
            const rect = new fabric.Rect({
                left: x1,
                top: y1,
                width: x2 - x1,
                height: y2 - y1,
                fill: 'transparent',
                stroke: '#6C5CE7',
                strokeWidth: 2,
                strokeDashArray: [5, 5],
                selectable: false,
                evented: false
            });

            this.canvas.add(rect);
        });

        this.canvas.renderAll();
    }

    selectTextBlock(index) {
        const block = this.textBlocks[index];
        
        // Remove previous selection
        document.querySelectorAll('.text-item').forEach(item => {
            item.classList.remove('selected');
        });

        // Add selection
        document.querySelector(`.text-item[data-index="${index}"]`).classList.add('selected');

        // Show edit panel
        this.showEditPanel(block);
    }

    showEditPanel(block) {
        const panel = document.getElementById('editPanel');
        const textarea = document.getElementById('editText');
        
        textarea.value = block.text;
        panel.classList.remove('hidden');
        
        // Store current block
        this.currentEditBlock = block;
    }

    hideEditPanel() {
        document.getElementById('editPanel').classList.add('hidden');
        this.currentEditBlock = null;
    }

    async saveTextEdit() {
        if (!this.currentEditBlock) return;

        const newText = document.getElementById('editText').value;
        if (!newText) return;

        this.showLoading();
        this.hideEditPanel();

        try {
            // Get current canvas image
            const imageData = this.canvas.toDataURL('image/png');

            const response = await fetch(`${this.apiUrl}/replace-text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    text_edit: {
                        bbox: this.currentEditBlock.bbox,
                        old_text: this.currentEditBlock.text,
                        new_text: newText,
                        style: this.currentEditBlock.style
                    }
                })
            });

            const data = await response.json();

            if (data.success) {
                // Load edited image
                fabric.Image.fromURL(data.edited_image, (img) => {
                    this.canvas.clear();
                    
                    img.set({
                        left: 0,
                        top: 0,
                        scaleX: 1,
                        scaleY: 1,
                        selectable: false,
                        evented: false
                    });

                    this.canvas.add(img);
                    this.originalImage = img;
                    
                    // Update text blocks
                    this.currentEditBlock.text = newText;
                    this.displayTextBlocks();
                    
                    this.saveToHistory();
                    this.hideLoading();
                    this.showToast('✅ টেক্সট সফলভাবে এডিট করা হয়েছে', 'success');
                });
            }
        } catch (error) {
            console.error('Text edit error:', error);
            this.hideLoading();
            this.showToast('❌ টেক্সট এডিট করতে সমস্যা হয়েছে', 'error');
        }
    }

    updateTextStyle() {
        if (!this.selectedText) return;

        const fontFamily = document.getElementById('fontSelect').value;
        const fontSize = parseInt(document.getElementById('fontSize').value);
        const color = document.getElementById('colorPicker').value;

        this.selectedText.set({
            fontFamily: fontFamily,
            fontSize: fontSize,
            fill: color
        });

        this.canvas.renderAll();
        this.saveToHistory();
    }

    toggleTextStyle(style) {
        if (!this.selectedText) return;

        const button = document.getElementById(style + 'Btn');
        
        switch(style) {
            case 'bold':
                const isBold = this.selectedText.fontWeight === 'bold';
                this.selectedText.set('fontWeight', isBold ? 'normal' : 'bold');
                button.style.background = isBold ? 'rgba(255,255,255,0.2)' : '#6C5CE7';
                break;
                
            case 'italic':
                const isItalic = this.selectedText.fontStyle === 'italic';
                this.selectedText.set('fontStyle', isItalic ? 'normal' : 'italic');
                button.style.background = isItalic ? 'rgba(255,255,255,0.2)' : '#6C5CE7';
                break;
                
            case 'underline':
                const isUnderline = this.selectedText.underline === true;
                this.selectedText.set('underline', !isUnderline);
                button.style.background = isUnderline ? 'rgba(255,255,255,0.2)' : '#6C5CE7';
                break;
        }

        this.canvas.renderAll();
        this.saveToHistory();
    }

    async autoEnhance() {
        if (!this.originalImage) {
            this.showToast('⚠️ আগে একটি ছবি আপলোড করুন', 'warning');
            return;
        }

        this.showLoading();

        try {
            const imageData = this.canvas.toDataURL('image/png');
            
            // Convert base64 to blob
            const blob = this.dataURLToBlob(imageData);
            const formData = new FormData();
            formData.append('file', blob, 'image.png');

            const response = await fetch(`${this.apiUrl}/auto-enhance`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                fabric.Image.fromURL(data.enhanced_image, (img) => {
                    this.canvas.clear();
                    
                    img.set({
                        left: 0,
                        top: 0,
                        scaleX: 1,
                        scaleY: 1,
                        selectable: false,
                        evented: false
                    });

                    this.canvas.add(img);
                    this.originalImage = img;
                    
                    this.saveToHistory();
                    this.hideLoading();
                    this.showToast('✨ ইমেজ এনহ্যান্স করা হয়েছে', 'success');
                });
            }
        } catch (error) {
            console.error('Auto enhance error:', error);
            this.hideLoading();
            this.showToast('❌ এনহ্যান্স করতে সমস্যা হয়েছে', 'error');
        }
    }

    async openBatchEdit() {
        if (this.textBlocks.length === 0) {
            this.showToast('⚠️ কোনো টেক্সট পাওয়া যায়নি', 'warning');
            return;
        }

        // Create batch edit modal
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content glass-effect">
                <h3>ব্যাচ এডিট</h3>
                <div class="batch-edit-list">
                    ${this.textBlocks.map((block, index) => `
                        <div class="batch-edit-item">
                            <span>${block.text}</span>
                            <input type="text" class="batch-input" data-index="${index}" placeholder="নতুন টেক্সট..." value="${block.text}">
                        </div>
                    `).join('')}
                </div>
                <div class="modal-actions">
                    <button class="btn-secondary" onclick="this.closest('.modal').remove()">বাতিল</button>
                    <button class="btn-primary" id="applyBatchEdit">প্রয়োগ করুন</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        document.getElementById('applyBatchEdit').addEventListener('click', async () => {
            const edits = [];
            document.querySelectorAll('.batch-input').forEach(input => {
                const index = input.dataset.index;
                const newText = input.value;
                
                if (newText && newText !== this.textBlocks[index].text) {
                    edits.push({
                        bbox: this.textBlocks[index].bbox,
                        old_text: this.textBlocks[index].text,
                        new_text: newText,
                        style: this.textBlocks[index].style
                    });
                }
            });

            if (edits.length > 0) {
                await this.applyBatchEdits(edits);
            }

            modal.remove();
        });
    }

    async applyBatchEdits(edits) {
        this.showLoading();

        try {
            const imageData = this.canvas.toDataURL('image/png');
            const blob = this.dataURLToBlob(imageData);
            const formData = new FormData();
            formData.append('file', blob, 'image.png');
            formData.append('edits', JSON.stringify(edits));

            const response = await fetch(`${this.apiUrl}/batch-edit`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                fabric.Image.fromURL(data.edited_image, (img) => {
                    this.canvas.clear();
                    
                    img.set({
                        left: 0,
                        top: 0,
                        scaleX: 1,
                        scaleY: 1,
                        selectable: false,
                        evented: false
                    });

                    this.canvas.add(img);
                    this.originalImage = img;
                    
                    // Update text blocks
                    edits.forEach(edit => {
                        const block = this.textBlocks.find(b => b.text === edit.old_text);
                        if (block) block.text = edit.new_text;
                    });
                    
                    this.displayTextBlocks();
                    this.saveToHistory();
                    this.hideLoading();
                    this.showToast('✅ ব্যাচ এডিট সম্পন্ন হয়েছে', 'success');
                });
            }
        } catch (error) {
            console.error('Batch edit error:', error);
            this.hideLoading();
            this.showToast('❌ ব্যাচ এডিট ব্যর্থ হয়েছে', 'error');
        }
    }

    downloadImage() {
        if (!this.originalImage) {
            this.showToast('⚠️ কোনো ছবি নেই', 'warning');
            return;
        }

        // Show download options
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content glass-effect">
                <h3>ডাউনলোড অপশন</h3>
                <div class="download-options">
                    <button class="btn-primary" id="downloadPNG">
                        <span class="icon">🖼️</span> PNG (High Quality)
                    </button>
                    <button class="btn-primary" id="downloadJPG">
                        <span class="icon">📸</span> JPG (Small Size)
                    </button>
                    <button class="btn-primary" id="downloadJSON">
                        <span class="icon">📋</span> JSON (Edit Data)
                    </button>
                </div>
                <div class="quality-control">
                    <label>কোয়ালিটি:</label>
                    <input type="range" id="downloadQuality" min="0.1" max="1" step="0.1" value="0.95">
                    <span id="qualityValue">95%</span>
                </div>
                <button class="btn-secondary" onclick="this.closest('.modal').remove()">বাতিল</button>
            </div>
        `;

        document.body.appendChild(modal);

        document.getElementById('downloadQuality').addEventListener('input', (e) => {
            document.getElementById('qualityValue').textContent = Math.round(e.target.value * 100) + '%';
        });

        document.getElementById('downloadPNG').addEventListener('click', () => {
            const quality = document.getElementById('downloadQuality').value;
            this.exportImage('png', quality);
            modal.remove();
        });

        document.getElementById('downloadJPG').addEventListener('click', () => {
            const quality = document.getElementById('downloadQuality').value;
            this.exportImage('jpeg', quality);
            modal.remove();
        });

        document.getElementById('downloadJSON').addEventListener('click', () => {
            this.exportJSON();
            modal.remove();
        });
    }

    exportImage(format, quality = 0.95) {
        const dataURL = this.canvas.toDataURL({
            format: format,
            quality: quality
        });

        const link = document.createElement('a');
        link.download = `edited-image.${format}`;
        link.href = dataURL;
        link.click();

        this.showToast('✅ ইমেজ ডাউনলোড হচ্ছে', 'success');
    }

    exportJSON() {
        const data = {
            image: this.canvas.toJSON(),
            textBlocks: this.textBlocks,
            history: this.history,
            timestamp: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.download = 'project-data.json';
        link.href = url;
        link.click();

        URL.revokeObjectURL(url);
        this.showToast('✅ JSON ডাউনলোড হচ্ছে', 'success');
    }

    // History Management
    saveToHistory() {
        const state = this.canvas.toJSON();
        
        // Remove future states if we're not at the end
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }

        this.history.push(state);
        this.historyIndex = this.history.length - 1;

        // Limit history size
        if (this.history.length > 50) {
            this.history.shift();
            this.historyIndex--;
        }

        this.updateTimeline();
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.loadHistoryState();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.loadHistoryState();
        }
    }

    jumpToHistory(index) {
        this.historyIndex = parseInt(index);
        this.loadHistoryState();
    }

    loadHistoryState() {
        if (this.historyIndex >= 0 && this.historyIndex < this.history.length) {
            this.canvas.loadFromJSON(this.history[this.historyIndex], () => {
                this.canvas.renderAll();
                this.updateTimeline();
            });
        }
    }

    updateTimeline() {
        const undoBtn = document.getElementById('undoBtn');
        const redoBtn = document.getElementById('redoBtn');
        const slider = document.getElementById('historySlider');

        undoBtn.disabled = this.historyIndex <= 0;
        redoBtn.disabled = this.historyIndex >= this.history.length - 1;

        slider.max = this.history.length - 1;
        slider.value = this.historyIndex;
    }

    // Utility Functions
    showLoading() {
        this.isProcessing = true;
        document.getElementById('loadingOverlay').classList.remove('hidden');
    }

    hideLoading() {
        this.isProcessing = false;
        document.getElementById('loadingOverlay').classList.add('hidden');
    }

    showUploadOverlay() {
        document.getElementById('uploadOverlay').classList.remove('hidden');
    }

    hideUploadOverlay() {
        document.getElementById('uploadOverlay').classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.className = `toast ${type}`;
        toast.classList.remove('hidden');

        setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }

    dataURLToBlob(dataURL) {
        const parts = dataURL.split(';base64,');
        const contentType = parts[0].split(':')[1];
        const raw = window.atob(parts[1]);
        const rawLength = raw.length;
        const uInt8Array = new Uint8Array(rawLength);

        for (let i = 0; i < rawLength; ++i) {
            uInt8Array[i] = raw.charCodeAt(i);
        }

        return new Blob([uInt8Array], { type: contentType });
    }

    loadFromLocalStorage() {
        try {
            const saved = localStorage.getItem('textEditorState');
            if (saved) {
                const state = JSON.parse(saved);
                // Load saved state
                this.showToast('📁 পূর্বের কাজ লোড করা হয়েছে', 'info');
            }
        } catch (error) {
            console.error('LocalStorage error:', error);
        }
    }

    saveToLocalStorage() {
        try {
            const state = {
                history: this.history,
                historyIndex: this.historyIndex,
                textBlocks: this.textBlocks
            };
            localStorage.setItem('textEditorState', JSON.stringify(state));
        } catch (error) {
            console.error('LocalStorage error:', error);
        }
    }

    // Event Handlers
    onObjectSelected(e) {
        const obj = e.selected[0];
        if (obj.type === 'text') {
            this.selectedText = obj;
            
            // Update style controls
            document.getElementById('fontSelect').value = obj.fontFamily || 'Arial';
            document.getElementById('fontSize').value = Math.round(obj.fontSize || 32);
            document.getElementById('fontSizeValue').textContent = Math.round(obj.fontSize || 32) + 'px';
            document.getElementById('colorPicker').value = obj.fill || '#000000';
            
            // Update effect buttons
            document.getElementById('boldBtn').style.background = obj.fontWeight === 'bold' ? '#6C5CE7' : 'rgba(255,255,255,0.2)';
            document.getElementById('italicBtn').style.background = obj.fontStyle === 'italic' ? '#6C5CE7' : 'rgba(255,255,255,0.2)';
            document.getElementById('underlineBtn').style.background = obj.underline ? '#6C5CE7' : 'rgba(255,255,255,0.2)';
        }
    }

    onSelectionCleared() {
        this.selectedText = null;
    }

    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + Z for undo
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            this.undo();
        }
        
        // Ctrl/Cmd + Shift + Z for redo
        if ((e.ctrlKey || e.metaKey) && e.key === 'Z' && e.shiftKey) {
            e.preventDefault();
            this.redo();
        }
        
        // Ctrl/Cmd + S for save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            this.downloadImage();
        }
        
        // Delete key for selected object
        if (e.key === 'Delete' && this.selectedText) {
            e.preventDefault();
            this.canvas.remove(this.selectedText);
            this.selectedText = null;
            this.saveToHistory();
        }
        
        // Escape key to close panels
        if (e.key === 'Escape') {
            this.hideEditPanel();
        }
    }

    openSettings() {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content glass-effect">
                <h3>⚙️ সেটিংস</h3>
                <div class="settings-list">
                    <label class="setting-item">
                        <span>অটো-সেভ</span>
                        <input type="checkbox" id="autoSave" checked>
                    </label>
                    <label class="setting-item">
                        <span>হাই কোয়ালিটি প্রিভিউ</span>
                        <input type="checkbox" id="highQuality" checked>
                    </label>
                    <label class="setting-item">
                        <span>টাচジェスチャー</span>
                        <input type="checkbox" id="touchGestures" checked>
                    </label>
                    <label class="setting-item">
                        <span>ডার্ক মোড</span>
                        <input type="checkbox" id="darkMode" checked>
                    </label>
                </div>
                <button class="btn-primary" onclick="this.closest('.modal').remove()">সংরক্ষণ</button>
            </div>
        `;

        document.body.appendChild(modal);
    }

    showHelp() {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content glass-effect">
                <h3>❓ কিভাবে ব্যবহার করবেন</h3>
                <div class="help-content">
                    <h4>কীবোর্ড শর্টকাট:</h4>
                    <ul>
                        <li><kbd>Ctrl + Z</kbd> - Undo</li>
                        <li><kbd>Ctrl + Shift + Z</kbd> - Redo</li>
                        <li><kbd>Ctrl + S</kbd> - ডাউনলোড</li>
                        <li><kbd>Delete</kbd> - টেক্সট মুছুন</li>
                        <li><kbd>Esc</kbd> - প্যানেল বন্ধ করুন</li>
                    </ul>
                    
                    <h4>টাচジェスチャー:</h4>
                    <ul>
                        <li>👆 সিঙ্গেল ট্যাপ - সিলেক্ট</li>
                        <li>✌️ ডাবল ট্যাপ - এডিট</li>
                        <li>🤌 পিঞ্চ - জুম</li>
                        <li>👆 ড্র্যাগ - মুভ</li>
                    </ul>
                    
                    <h4>টিপস:</h4>
                    <ul>
                        <li>📸 হাই কোয়ালিটি ছবি ব্যবহার করুন</li>
                        <li>🎯 প্রতিটি টেক্সট ব্লক আলাদাভাবে এডিট করুন</li>
                        <li>💾 অটো-সেভ চালু রাখুন</li>
                        <li>🔄 ব্যাচ এডিট ব্যবহার করে একসাথে অনেক টেক্সট এডিট করুন</li>
                    </ul>
                </div>
                <button class="btn-primary" onclick="this.closest('.modal').remove()">বুঝেছি</button>
            </div>
        `;

        document.body.appendChild(modal);
    }
}

// Initialize the editor when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.editor = new ProfessionalTextEditor();
});

// Handle visibility change
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && window.editor) {
        window.editor.saveToLocalStorage();
    }
});

// Handle online/offline status
window.addEventListener('online', () => {
    window.editor?.showToast('🌐 ইন্টারনেট সংযোগ পুনরুদ্ধার হয়েছে', 'success');
});

window.addEventListener('offline', () => {
    window.editor?.showToast('⚠️ ইন্টারনেট সংযোগ বিচ্ছিন্ন', 'warning');
});
