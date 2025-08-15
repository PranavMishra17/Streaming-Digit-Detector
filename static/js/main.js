/**
 * Main Application Controller
 * Coordinates all components and manages the UI state
 */

class AudioDigitApp {
    constructor() {
        // Core components
        this.audioRecorder = null;
        this.audioVisualizer = null;
        this.noiseGenerator = null;
        
        // UI elements
        this.elements = {
            startRecording: document.getElementById('startRecording'),
            stopRecording: document.getElementById('stopRecording'),
            clearCanvas: document.getElementById('clearCanvas'),
            processAudio: document.getElementById('processAudio'),
            getStats: document.getElementById('getStats'),
            testConnection: document.getElementById('testConnection'),
            
            // Status displays
            recordingStatus: document.getElementById('recordingStatus'),
            audioInfo: document.getElementById('audioInfo'),
            predictedDigit: document.getElementById('predictedDigit'),
            methodUsed: document.getElementById('methodUsed'),
            inferenceTime: document.getElementById('inferenceTime'),
            audioDuration: document.getElementById('audioDuration'),
            averageTime: document.getElementById('averageTime'),
            
            // Performance stats
            totalPredictions: document.getElementById('totalPredictions'),
            fastestMethod: document.getElementById('fastestMethod'),
            successRate: document.getElementById('successRate'),
            
            // Noise controls
            noiseType: document.getElementById('noiseType'),
            noiseLevel: document.getElementById('noiseLevel'),
            noiseLevelValue: document.getElementById('noiseLevelValue'),
            
            // Log
            activityLog: document.getElementById('activityLog'),
            
            // Canvas
            audioCanvas: document.getElementById('audioCanvas')
        };
        
        // Application state
        this.state = {
            isRecording: false,
            hasRecordedAudio: false,
            currentAudioBlob: null,
            selectedMethod: 'external_api',
            totalPredictions: 0,
            methodStats: {},
            sessionStartTime: Date.now(),
            streamingErrors: 0,
            maxStreamingErrors: 5,
            lastErrorTime: 0
        };
        
        // Initialize
        this.initialize();
    }
    
    /**
     * Initialize the application
     */
    async initialize() {
        try {
            addLogEntry('[INFO] Initializing Audio Digit Classifier...', 'info');
            
            // Initialize components
            await this.initializeComponents();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize UI state
            this.updateUIState();
            
            addLogEntry('[SUCCESS] Application initialized successfully', 'success');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
            addLogEntry(`[ERROR] Initialization failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Initialize core components
     */
    async initializeComponents() {
        // Initialize audio recorder
        this.audioRecorder = new AudioRecorder();
        
        // Set up audio recorder callbacks
        this.audioRecorder.onRecordingStart = () => {
            this.state.isRecording = true;
            this.updateRecordingState();
            addLogEntry('[INFO] Recording started', 'info');
        };
        
        this.audioRecorder.onRecordingStop = (duration) => {
            this.state.isRecording = false;
            this.updateRecordingState();
            addLogEntry(`[INFO] Recording stopped - Duration: ${(duration/1000).toFixed(1)}s`, 'info');
        };
        
        this.audioRecorder.onDataAvailable = (audioBlob, duration) => {
            this.state.hasRecordedAudio = true;
            this.state.currentAudioBlob = audioBlob;
            this.updateAudioInfo(duration);
            this.updateUIState();
            addLogEntry(`[SUCCESS] Audio captured - ${(audioBlob.size/1024).toFixed(1)}KB`, 'success');
        };
        
        this.audioRecorder.onError = (error) => {
            addLogEntry(`[ERROR] Recording error: ${error.message}`, 'error');
            this.state.isRecording = false;
            this.updateRecordingState();
        };
        
        this.audioRecorder.onChunkReady = (audioBlob, duration) => {
            this.state.currentAudioBlob = audioBlob;
            this.processAudioChunk(audioBlob, duration);
            addLogEntry(`[INFO] Streaming chunk ready - ${(duration/1000).toFixed(1)}s`, 'info');
        };
        
        // Initialize audio visualizer
        this.audioVisualizer = new AudioVisualizer(this.elements.audioCanvas, {
            waveColor: '#00ff00',
            backgroundColor: '#001100',
            showGrid: true,
            showText: true,
            retroGlow: true
        });
        
        // Initialize noise generator
        this.noiseGenerator = new NoiseGenerator();
        await this.noiseGenerator.initialize();
        
        addLogEntry('[INFO] Core components initialized', 'info');
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Recording controls
        this.elements.startRecording.addEventListener('click', () => this.startRecording());
        this.elements.stopRecording.addEventListener('click', () => this.stopRecording());
        this.elements.clearCanvas.addEventListener('click', () => this.clearVisualization());
        
        // Processing controls
        this.elements.processAudio.addEventListener('click', () => this.processAudio());
        this.elements.getStats.addEventListener('click', () => this.showStats());
        this.elements.testConnection.addEventListener('click', () => this.testAPIConnection());
        
        // Method selection with lazy loading
        const methodRadios = document.querySelectorAll('input[name=\"method\"]');
        methodRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.state.selectedMethod = e.target.value;
                    this.updateMethodSelection();
                    this.initializeSelectedMethod(e.target.value);
                    addLogEntry(`[INFO] Selected method: ${this.getMethodName(e.target.value)}`, 'info');
                }
            });
        });
        
        // Noise controls
        this.elements.noiseLevel.addEventListener('input', (e) => {
            this.elements.noiseLevelValue.textContent = e.target.value;
        });
        
        this.elements.noiseType.addEventListener('change', (e) => {
            const noiseType = e.target.value;
            if (noiseType !== 'none') {
                addLogEntry(`[INFO] Noise injection enabled: ${noiseType}`, 'info');
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
                e.preventDefault();
                if (!this.state.isRecording) {
                    this.startRecording();
                } else {
                    this.stopRecording();
                }
            }
        });
        
        // Cabinet visual feedback
        const cabinets = document.querySelectorAll('.radio-cabinet');
        cabinets.forEach(cabinet => {
            cabinet.addEventListener('click', () => {
                const radio = cabinet.querySelector('input[type=\"radio\"]');
                radio.checked = true;
                radio.dispatchEvent(new Event('change'));
            });
        });
        
        addLogEntry('[INFO] Event listeners registered', 'info');
    }
    
    /**
     * Initialize selected method (lazy loading)
     */
    async initializeSelectedMethod(method) {
        try {
            addLogEntry(`[INFO] Initializing ${this.getMethodName(method)}...`, 'info');
            
            // Show loading indicator
            const cabinet = document.querySelector(`[data-method="${method}"]`);
            if (cabinet) {
                cabinet.classList.add('loading');
            }
            
            // Pre-warm the model by making a test request
            const testAudio = new ArrayBuffer(1000); // Minimal audio data
            const testBlob = new Blob([testAudio], { type: 'audio/wav' });
            
            const formData = new FormData();
            formData.append('audio', testBlob, 'init_test.wav');
            formData.append('method', method);
            
            const response = await fetch('/process_audio', {
                method: 'POST',
                body: formData
            });
            
            // Remove loading indicator
            if (cabinet) {
                cabinet.classList.remove('loading');
            }
            
            if (response.ok) {
                addLogEntry(`[SUCCESS] ${this.getMethodName(method)} ready`, 'success');
                this.updateCabinetStatus(method, 'ready');
            } else {
                addLogEntry(`[WARNING] ${this.getMethodName(method)} may have issues`, 'warning');
                this.updateCabinetStatus(method, 'error');
            }
            
        } catch (error) {
            console.error(`Failed to initialize ${method}:`, error);
            addLogEntry(`[ERROR] Failed to initialize ${this.getMethodName(method)}`, 'error');
            
            const cabinet = document.querySelector(`[data-method="${method}"]`);
            if (cabinet) {
                cabinet.classList.remove('loading');
            }
            this.updateCabinetStatus(method, 'error');
        }
    }
    
    /**
     * Start audio recording
     */
    async startRecording() {
        try {
            if (this.state.isRecording) return;
            
            // Check if selected method is initialized
            if (!this.isMethodReady(this.state.selectedMethod)) {
                addLogEntry(`[INFO] Initializing ${this.getMethodName(this.state.selectedMethod)} first...`, 'info');
                await this.initializeSelectedMethod(this.state.selectedMethod);
            }
            
            // Start recording
            await this.audioRecorder.startRecording();
            
            // Start visualization
            this.audioVisualizer.start(this.audioRecorder);
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            addLogEntry(`[ERROR] Failed to start recording: ${error.message}`, 'error');
            
            // Handle permission denied
            if (error.message.includes('Permission denied') || error.message.includes('NotAllowedError')) {
                this.showMicrophonePermissionHelp();
            }
        }
    }
    
    /**
     * Check if method is ready for use
     */
    isMethodReady(method) {
        const cabinet = document.querySelector(`[data-method="${method}"]`);
        if (!cabinet) return false;
        
        const indicator = cabinet.querySelector('.status-indicator');
        return indicator && (indicator.classList.contains('ready') || indicator.classList.contains('working'));
    }
    
    /**
     * Stop audio recording
     */
    stopRecording() {
        if (!this.state.isRecording) return;
        
        this.audioRecorder.stopRecording();
        this.audioVisualizer.stop();
    }
    
    /**
     * Clear visualization
     */
    clearVisualization() {
        this.audioVisualizer.clear();
        addLogEntry('[INFO] Visualization cleared', 'info');
    }
    
    /**
     * Process audio chunk automatically (streaming mode)
     */
    async processAudioChunk(audioBlob, duration) {
        // Check for error backoff
        const now = Date.now();
        if (this.state.streamingErrors >= this.state.maxStreamingErrors) {
            const timeSinceLastError = now - this.state.lastErrorTime;
            if (timeSinceLastError < 10000) { // 10 second backoff
                console.log('Skipping chunk due to error backoff');
                return;
            } else {
                // Reset error count after backoff period
                this.state.streamingErrors = 0;
                addLogEntry('[INFO] Resuming streaming after error backoff', 'info');
            }
        }
        
        try {
            // Apply noise if configured
            const noiseType = this.elements.noiseType.value;
            const noiseLevel = parseFloat(this.elements.noiseLevel.value);
            
            let processedBlob = audioBlob;
            if (noiseType !== 'none' && noiseLevel > 0) {
                processedBlob = await this.noiseGenerator.mixNoiseWithAudio(audioBlob, noiseType, noiseLevel);
            }
            
            // Use the selected method from UI
            let selectedMethod = this.state.selectedMethod;

            // Prepare form data
            const formData = new FormData();
            formData.append('audio', processedBlob, 'streaming_chunk.wav');
            formData.append('method', selectedMethod);
            formData.append('noise_type', noiseType);
            formData.append('noise_level', noiseLevel.toString());
            
            // Send to server
            const response = await fetch('/process_audio', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok && result.success !== false) {
                // Reset error count on success
                this.state.streamingErrors = 0;
                
                // Display streaming result
                this.displayStreamingResult(result);
                this.updateStats(result);
                
                addLogEntry(`[SUCCESS] Streaming: ${result.predicted_digit} (${result.inference_time}s)`, 'success');
            } else {
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
        } catch (error) {
            this.state.streamingErrors++;
            this.state.lastErrorTime = now;
            
            console.error('Streaming chunk processing failed:', error);
            addLogEntry(`[ERROR] Streaming error (${this.state.streamingErrors}/${this.state.maxStreamingErrors}): ${error.message}`, 'error');
            
            if (this.state.streamingErrors >= this.state.maxStreamingErrors) {
                addLogEntry('[WARNING] Too many streaming errors - entering backoff mode', 'warning');
            }
        }
    }
    
    /**
     * Process recorded audio with selected method (manual mode)
     */
    async processAudio() {
        if (!this.state.hasRecordedAudio || !this.state.currentAudioBlob) {
            addLogEntry('[WARNING] No audio to process', 'warning');
            return;
        }
        
        try {
            // Update UI for processing state
            this.elements.processAudio.textContent = 'Processing...';
            this.elements.processAudio.disabled = true;
            
            // Apply noise if configured
            let audioBlob = this.state.currentAudioBlob;
            const noiseType = this.elements.noiseType.value;
            const noiseLevel = parseFloat(this.elements.noiseLevel.value);
            
            if (noiseType !== 'none' && noiseLevel > 0) {
                addLogEntry(`[INFO] Applying ${noiseType} noise (level: ${noiseLevel})`, 'info');
                audioBlob = await this.noiseGenerator.mixNoiseWithAudio(audioBlob, noiseType, noiseLevel);
            }
            
            // Prepare form data
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            formData.append('method', this.state.selectedMethod);
            formData.append('noise_type', noiseType);
            formData.append('noise_level', noiseLevel.toString());
            
            addLogEntry(`[INFO] Processing with ${this.getMethodName(this.state.selectedMethod)}...`, 'info');
            
            // Send to server
            const response = await fetch('/process_audio', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok && result.success !== false) {
                // Update UI with results
                this.displayResults(result);
                this.updateStats(result);
                
                addLogEntry(`[SUCCESS] Predicted digit: ${result.predicted_digit} (${result.inference_time}s)`, 'success');
            } else {
                throw new Error(result.error || 'Processing failed');
            }
            
        } catch (error) {
            console.error('Audio processing failed:', error);
            addLogEntry(`[ERROR] Processing failed: ${error.message}`, 'error');
            
            // Show error in UI
            this.elements.predictedDigit.textContent = 'ERROR';
            this.elements.predictedDigit.style.color = '#ff0000';
            
        } finally {
            // Reset processing button
            this.elements.processAudio.textContent = 'Analyze Audio';
            this.elements.processAudio.disabled = false;
        }
    }
    
    /**
     * Display streaming result (non-intrusive)
     */
    displayStreamingResult(result) {
        // Update prediction with streaming indicator
        this.elements.predictedDigit.textContent = result.predicted_digit;
        this.elements.predictedDigit.style.color = result.predicted_digit === 'ERROR' ? '#ff0000' : '#ffe66d';
        
        // Update method and timing info
        this.elements.methodUsed.textContent = this.getMethodName(result.method);
        this.elements.inferenceTime.textContent = `${result.inference_time}s`;
        if (result.audio_duration) {
            this.elements.audioDuration.textContent = `${result.audio_duration}s`;
        }
        if (result.average_time) {
            this.elements.averageTime.textContent = `${result.average_time}s`;
        }
        
        // Brief visual feedback for method cabinet
        this.updateCabinetStatus(result.method, 'working');
        setTimeout(() => {
            this.updateCabinetStatus(result.method, 'ready');
        }, 1000);
    }
    
    /**
     * Display processing results in UI
     */
    displayResults(result) {
        // Main prediction
        this.elements.predictedDigit.textContent = result.predicted_digit;
        this.elements.predictedDigit.style.color = result.predicted_digit === 'ERROR' ? '#ff0000' : '#ffe66d';
        
        // Stats
        this.elements.methodUsed.textContent = this.getMethodName(result.method);
        this.elements.inferenceTime.textContent = `${result.inference_time}s`;
        this.elements.audioDuration.textContent = `${result.audio_duration}s`;
        this.elements.averageTime.textContent = `${result.average_time || result.inference_time}s`;
        
        // Visual feedback for method cabinet
        this.updateCabinetStatus(result.method, result.success !== false ? 'working' : 'error');
        
        setTimeout(() => {
            this.updateCabinetStatus(result.method, 'ready');
        }, 2000);
    }
    
    /**
     * Update application statistics
     */
    updateStats(result) {
        this.state.totalPredictions++;
        this.elements.totalPredictions.textContent = this.state.totalPredictions;
        
        // Update method stats
        if (!this.state.methodStats[result.method]) {
            this.state.methodStats[result.method] = {
                predictions: 0,
                totalTime: 0,
                errors: 0
            };
        }
        
        const methodStats = this.state.methodStats[result.method];
        methodStats.predictions++;
        methodStats.totalTime += result.inference_time;
        
        if (result.success === false) {
            methodStats.errors++;
        }
        
        // Find fastest method
        let fastestMethod = null;
        let fastestTime = Infinity;
        
        for (const [method, stats] of Object.entries(this.state.methodStats)) {
            const avgTime = stats.totalTime / stats.predictions;
            if (avgTime < fastestTime) {
                fastestTime = avgTime;
                fastestMethod = method;
            }
        }
        
        if (fastestMethod) {
            this.elements.fastestMethod.textContent = this.getMethodName(fastestMethod);
        }
        
        // Calculate success rate
        const totalErrors = Object.values(this.state.methodStats).reduce((sum, stats) => sum + stats.errors, 0);
        const successRate = ((this.state.totalPredictions - totalErrors) / this.state.totalPredictions * 100).toFixed(1);
        this.elements.successRate.textContent = `${successRate}%`;
    }
    
    /**
     * Show detailed statistics
     */
    async showStats() {
        try {
            const response = await fetch('/stats');
            const stats = await response.json();
            
            console.log('Detailed Statistics:', stats);
            addLogEntry('[INFO] Statistics retrieved - Check console for details', 'info');
            
            // Create stats popup (simple alert for now)
            let statsText = 'Performance Statistics:\n\n';
            for (const [method, methodStats] of Object.entries(stats)) {
                statsText += `${this.getMethodName(method)}:\n`;
                statsText += `  Predictions: ${methodStats.total_calls}\n`;
                statsText += `  Success Rate: ${(100 - methodStats.error_rate).toFixed(1)}%\n`;
                statsText += `  Avg Time: ${methodStats.avg_inference_time}s\n\n`;
            }
            
            alert(statsText);
            
        } catch (error) {
            console.error('Failed to get stats:', error);
            addLogEntry('[ERROR] Failed to retrieve statistics', 'error');
        }
    }
    
    /**
     * Test API connection
     */
    async testAPIConnection() {
        try {
            addLogEntry('[INFO] Testing API connection...', 'info');
            
            const response = await fetch('/health');
            const health = await response.json();
            
            if (response.ok) {
                addLogEntry('[SUCCESS] API connection test passed', 'success');
                
                // Test specific processors
                for (const [method, status] of Object.entries(health.processors)) {
                    const statusText = status.configured ? 'Ready' : 'Not configured';
                    const logLevel = status.configured ? 'success' : 'warning';
                    addLogEntry(`[${logLevel.toUpperCase()}] ${this.getMethodName(method)}: ${statusText}`, logLevel);
                }
            } else {
                throw new Error(health.error || 'Connection test failed');
            }
            
        } catch (error) {
            console.error('API connection test failed:', error);
            addLogEntry(`[ERROR] API connection test failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Update recording state UI
     */
    updateRecordingState() {
        if (this.state.isRecording) {
            this.elements.startRecording.disabled = true;
            this.elements.stopRecording.disabled = false;
            this.elements.recordingStatus.textContent = 'Streaming... (Press SPACE or click stop)';
            this.elements.recordingStatus.style.color = '#ff0000';
            
            // Add recording class for visual effects
            document.body.classList.add('recording');
        } else {
            this.elements.startRecording.disabled = false;
            this.elements.stopRecording.disabled = true;
            this.elements.recordingStatus.textContent = 'Ready to stream... (Press SPACE or click start)';
            this.elements.recordingStatus.style.color = '#00ff00';
            
            // Remove recording class
            document.body.classList.remove('recording');
        }
    }
    
    /**
     * Update audio information display
     */
    updateAudioInfo(duration) {
        this.elements.audioInfo.textContent = `Duration: ${(duration / 1000).toFixed(1)}s`;
    }
    
    /**
     * Update UI state based on application state
     */
    updateUIState() {
        // Enable/disable process button
        this.elements.processAudio.disabled = !this.state.hasRecordedAudio;
        
        // Update button text based on state
        if (this.state.hasRecordedAudio) {
            this.elements.processAudio.classList.remove('btn-disabled');
        } else {
            this.elements.processAudio.classList.add('btn-disabled');
        }
    }
    
    /**
     * Update method selection visual feedback
     */
    updateMethodSelection() {
        const cabinets = document.querySelectorAll('.radio-cabinet');
        cabinets.forEach(cabinet => {
            const method = cabinet.dataset.method;
            if (method === this.state.selectedMethod) {
                cabinet.classList.add('selected');
            } else {
                cabinet.classList.remove('selected');
            }
        });
    }
    
    /**
     * Update cabinet status indicators
     */
    updateCabinetStatus(method, status) {
        const cabinet = document.querySelector(`[data-method=\"${method}\"]`);
        if (cabinet) {
            const indicator = cabinet.querySelector('.status-indicator');
            if (indicator) {
                indicator.className = `status-indicator ${status}`;
            }
        }
    }
    
    /**
     * Show microphone permission help
     */
    showMicrophonePermissionHelp() {
        const helpText = `
Microphone Access Required

To use the audio digit classifier, please:

1. Click on the microphone icon in your browser's address bar
2. Select "Allow" for microphone access
3. Refresh the page and try again

Note: HTTPS is required for microphone access in most browsers.
        `;
        
        alert(helpText);
        addLogEntry('[INFO] Microphone permission help displayed', 'info');
    }
    
    /**
     * Get friendly method name
     */
    getMethodName(method) {
        const names = {
            external_api: 'External API (Whisper)',
            raw_spectrogram: 'Raw Spectrogram',
            mel_spectrogram: 'Mel Spectrogram',
            mfcc: 'MFCC Features'
        };
        return names[method] || method;
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        if (this.audioRecorder) {
            this.audioRecorder.cleanup();
        }
        
        if (this.audioVisualizer) {
            this.audioVisualizer.stop();
        }
        
        if (this.noiseGenerator) {
            this.noiseGenerator.cleanup();
        }
    }
}

/**
 * Utility function to add entries to the activity log
 */
function addLogEntry(message, level = 'info') {
    const logContainer = document.getElementById('activityLog');
    if (!logContainer) return;
    
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    
    logContainer.appendChild(entry);
    
    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
    
    // Limit log entries (keep last 50)
    const entries = logContainer.querySelectorAll('.log-entry');
    if (entries.length > 50) {
        entries[0].remove();
    }
}

/**
 * Global error handler
 */
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    addLogEntry(`[ERROR] ${event.error.message}`, 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    addLogEntry(`[ERROR] Promise rejection: ${event.reason}`, 'error');
    event.preventDefault();
});

// Initialize application when DOM is loaded
let app = null;

document.addEventListener('DOMContentLoaded', () => {
    app = new AudioDigitApp();
    
    // Make app globally available for debugging
    window.audioDigitApp = app;
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (app) {
        app.cleanup();
    }
});

// Export app class for testing
window.AudioDigitApp = AudioDigitApp;