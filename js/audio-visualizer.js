/**
 * Audio Visualizer Module
 * Creates retro-style real-time audio waveform visualization
 */

class AudioVisualizer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.animationId = null;
        this.isActive = false;
        
        // Configuration
        this.config = {
            backgroundColor: '#001100',
            waveColor: '#00ff00',
            gridColor: '#003300',
            textColor: '#00aa00',
            pixelSize: 2,
            waveHeight: 0.7,
            gridSpacing: 20,
            showGrid: true,
            showText: true,
            retroGlow: true,
            ...options
        };
        
        // Audio data
        this.audioData = null;
        this.audioLevel = 0;
        this.peakLevel = 0;
        this.peakDecay = 0.95;
        
        // Visualization state
        this.waveHistory = [];
        this.maxHistoryLength = 200;
        this.scanLine = 0;
        this.lastUpdateTime = 0;
        
        // Initialize canvas
        this.initializeCanvas();
        this.setupEventListeners();
    }
    
    /**
     * Initialize canvas settings
     */
    initializeCanvas() {
        // Set high DPI support
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        
        this.ctx.scale(dpr, dpr);
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        // Set pixelated rendering
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.webkitImageSmoothingEnabled = false;
        this.ctx.mozImageSmoothingEnabled = false;
        this.ctx.msImageSmoothingEnabled = false;
        
        // Initial draw
        this.drawBackground();
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        window.addEventListener('resize', () => {
            this.initializeCanvas();
        });
    }
    
    /**
     * Start visualization with audio recorder
     */
    start(audioRecorder) {
        this.audioRecorder = audioRecorder;
        this.isActive = true;
        this.animate();
        console.log('Audio visualizer started');
    }
    
    /**
     * Stop visualization
     */
    stop() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.audioRecorder = null;
        console.log('Audio visualizer stopped');
    }
    
    /**
     * Main animation loop
     */
    animate() {
        if (!this.isActive) return;
        
        const currentTime = Date.now();
        
        // Update audio data if available
        if (this.audioRecorder) {
            this.audioLevel = this.audioRecorder.getCurrentAudioLevel() || 0;
            this.audioData = this.audioRecorder.getTimeDomainData();
            
            // Update peak level
            if (this.audioLevel > this.peakLevel) {
                this.peakLevel = this.audioLevel;
            } else {
                this.peakLevel *= this.peakDecay;
            }
        }
        
        // Add to wave history for scrolling effect
        if (currentTime - this.lastUpdateTime > 50) { // Update every 50ms
            this.waveHistory.push({
                level: this.audioLevel,
                peak: this.peakLevel,
                timestamp: currentTime
            });
            
            // Limit history length
            if (this.waveHistory.length > this.maxHistoryLength) {
                this.waveHistory.shift();
            }
            
            this.lastUpdateTime = currentTime;
        }
        
        // Update scan line
        this.scanLine = (this.scanLine + 2) % this.canvas.clientWidth;
        
        // Draw visualization
        this.draw();
        
        // Continue animation
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    /**
     * Main drawing function
     */
    draw() {
        // Clear canvas
        this.drawBackground();
        
        // Draw grid if enabled
        if (this.config.showGrid) {
            this.drawGrid();
        }
        
        // Draw waveform
        if (this.audioData) {
            this.drawWaveform();
        } else {
            this.drawScrollingWave();
        }
        
        // Draw level meters
        this.drawLevelMeters();
        
        // Draw scan line
        this.drawScanLine();
        
        // Draw text overlays
        if (this.config.showText) {
            this.drawTextOverlay();
        }
    }
    
    /**
     * Draw background with retro pattern
     */
    drawBackground() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        
        // Create gradient background
        const gradient = this.ctx.createRadialGradient(
            width / 2, height / 2, 0,
            width / 2, height / 2, Math.max(width, height) / 2
        );
        gradient.addColorStop(0, this.config.backgroundColor);
        gradient.addColorStop(1, '#000800');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, width, height);
    }
    
    /**
     * Draw retro grid pattern
     */
    drawGrid() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        const spacing = this.config.gridSpacing;
        
        this.ctx.strokeStyle = this.config.gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.3;
        
        // Vertical lines
        for (let x = 0; x <= width; x += spacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y <= height; y += spacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    /**
     * Draw real-time waveform
     */
    drawWaveform() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        const centerY = height / 2;
        const waveHeight = height * this.config.waveHeight;
        
        this.ctx.strokeStyle = this.config.waveColor;
        this.ctx.lineWidth = 2;
        
        // Add glow effect
        if (this.config.retroGlow) {
            this.ctx.shadowColor = this.config.waveColor;
            this.ctx.shadowBlur = 5;
        }
        
        this.ctx.beginPath();
        
        // Draw waveform from audio data
        for (let i = 0; i < this.audioData.length; i++) {
            const x = (i / this.audioData.length) * width;
            const sample = (this.audioData[i] - 128) / 128.0; // Normalize to -1 to 1
            const y = centerY + sample * waveHeight / 2;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Reset shadow
        if (this.config.retroGlow) {
            this.ctx.shadowBlur = 0;
        }
    }
    
    /**
     * Draw scrolling wave visualization
     */
    drawScrollingWave() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        const centerY = height / 2;
        const waveHeight = height * this.config.waveHeight;
        
        if (this.waveHistory.length < 2) return;
        
        this.ctx.strokeStyle = this.config.waveColor;
        this.ctx.lineWidth = 2;
        
        // Add glow effect
        if (this.config.retroGlow) {
            this.ctx.shadowColor = this.config.waveColor;
            this.ctx.shadowBlur = 3;
        }
        
        this.ctx.beginPath();
        
        // Draw wave from history
        for (let i = 0; i < this.waveHistory.length; i++) {
            const x = (i / this.maxHistoryLength) * width;
            const level = this.waveHistory[i].level / 255; // Normalize
            const y = centerY - (level * waveHeight / 2);
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Draw frequency bars if available
        if (this.audioRecorder) {
            const freqData = this.audioRecorder.getFrequencyData();
            if (freqData) {
                this.drawFrequencyBars(freqData);
            }
        }
        
        // Reset shadow
        if (this.config.retroGlow) {
            this.ctx.shadowBlur = 0;
        }
    }
    
    /**
     * Draw frequency spectrum as bars
     */
    drawFrequencyBars(freqData) {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        const barCount = 32; // Reduced for retro look
        const barWidth = width / barCount;
        
        this.ctx.fillStyle = this.config.waveColor;
        this.ctx.globalAlpha = 0.6;
        
        // Draw bars
        for (let i = 0; i < barCount; i++) {
            const dataIndex = Math.floor((i / barCount) * freqData.length);
            const barHeight = (freqData[dataIndex] / 255) * height * 0.8;
            
            const x = i * barWidth;
            const y = height - barHeight;
            
            // Pixelated bars
            this.ctx.fillRect(
                Math.floor(x), 
                Math.floor(y), 
                Math.floor(barWidth - 2), 
                Math.floor(barHeight)
            );
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    /**
     * Draw level meters
     */
    drawLevelMeters() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        const meterWidth = 20;
        const meterHeight = height - 40;
        const meterX = width - 30;
        const meterY = 20;
        
        // Background
        this.ctx.fillStyle = '#003300';
        this.ctx.fillRect(meterX, meterY, meterWidth, meterHeight);
        
        // Level bar
        const levelHeight = (this.audioLevel / 255) * meterHeight;
        const levelY = meterY + meterHeight - levelHeight;
        
        // Color based on level
        let levelColor = this.config.waveColor;
        if (this.audioLevel > 200) levelColor = '#ffff00';
        if (this.audioLevel > 240) levelColor = '#ff0000';
        
        this.ctx.fillStyle = levelColor;
        this.ctx.fillRect(meterX, levelY, meterWidth, levelHeight);
        
        // Peak indicator
        const peakY = meterY + meterHeight - (this.peakLevel / 255) * meterHeight;
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(meterX, peakY - 1, meterWidth, 2);
        
        // Border
        this.ctx.strokeStyle = this.config.textColor;
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(meterX, meterY, meterWidth, meterHeight);
    }
    
    /**
     * Draw scanning line effect
     */
    drawScanLine() {
        const height = this.canvas.clientHeight;
        
        // Vertical scan line
        this.ctx.strokeStyle = this.config.waveColor;
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.3;
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.scanLine, 0);
        this.ctx.lineTo(this.scanLine, height);
        this.ctx.stroke();
        
        this.ctx.globalAlpha = 1;
    }
    
    /**
     * Draw text overlay with info
     */
    drawTextOverlay() {
        const width = this.canvas.clientWidth;
        
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '10px monospace';
        this.ctx.textAlign = 'left';
        
        // Status text
        const status = this.audioRecorder && this.audioRecorder.recording ? 'RECORDING' : 'STANDBY';
        this.ctx.fillText(`STATUS: ${status}`, 10, 20);
        
        // Level info
        const level = Math.round(this.audioLevel);
        this.ctx.fillText(`LEVEL: ${level}`, 10, 35);
        
        // Peak info  
        const peak = Math.round(this.peakLevel);
        this.ctx.fillText(`PEAK: ${peak}`, 10, 50);
        
        // Recording time if active
        if (this.audioRecorder && this.audioRecorder.recording) {
            const duration = (this.audioRecorder.duration / 1000).toFixed(1);
            this.ctx.fillText(`TIME: ${duration}s`, 10, 65);
            
            // Blinking REC indicator
            if (Math.floor(Date.now() / 500) % 2) {
                this.ctx.fillStyle = '#ff0000';
                this.ctx.fillText('● REC', width - 60, 20);
            }
        }
    }
    
    /**
     * Clear the visualization
     */
    clear() {
        this.waveHistory = [];
        this.audioLevel = 0;
        this.peakLevel = 0;
        this.drawBackground();
        
        if (this.config.showGrid) {
            this.drawGrid();
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }
    
    /**
     * Set wave color (for dynamic theming)
     */
    setWaveColor(color) {
        this.config.waveColor = color;
        this.config.textColor = color;
    }
    
    /**
     * Toggle grid display
     */
    toggleGrid() {
        this.config.showGrid = !this.config.showGrid;
    }
    
    /**
     * Toggle text overlay
     */
    toggleText() {
        this.config.showText = !this.config.showText;
    }
    
    /**
     * Get current audio statistics
     */
    getStats() {
        return {
            audioLevel: this.audioLevel,
            peakLevel: this.peakLevel,
            historyLength: this.waveHistory.length,
            isActive: this.isActive
        };
    }
}

// Export for use in main application
window.AudioVisualizer = AudioVisualizer;