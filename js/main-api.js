/**
 * API Integration for HF Spaces Backend
 * Handles communication with the Flask API backend
 */

class APIClient {
    constructor(baseUrl = '') {
        // Use the API base URL from global config
        this.baseUrl = window.APP_CONFIG.apiBaseUrl;
        this.endpoints = window.APP_CONFIG.endpoints;
    }

    /**
     * Process audio file with selected method
     */
    async processAudio(audioBlob, method, options = {}) {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, options.filename || 'recording.wav');
            formData.append('method', method);
            
            if (options.noiseType && options.noiseType !== 'none') {
                formData.append('noise_type', options.noiseType);
                formData.append('noise_level', options.noiseLevel || '0.0');
            }

            const response = await fetch(`${this.baseUrl}${this.endpoints.processAudio}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
            
        } catch (error) {
            console.error('API processAudio failed:', error);
            throw error;
        }
    }

    /**
     * Process streaming audio chunk
     */
    async processAudioChunk(audioBlob, method, options = {}) {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, options.filename || 'chunk.wav');
            formData.append('method', method);
            
            if (options.noiseType && options.noiseType !== 'none') {
                formData.append('noise_type', options.noiseType);
                formData.append('noise_level', options.noiseLevel || '0.0');
            }

            const response = await fetch(`${this.baseUrl}${this.endpoints.processChunk}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
            
        } catch (error) {
            console.error('API processAudioChunk failed:', error);
            throw error;
        }
    }

    /**
     * Check API health and get processor status
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.health}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
            
        } catch (error) {
            console.error('API health check failed:', error);
            throw error;
        }
    }

    /**
     * Get available processors
     */
    async getProcessors() {
        try {
            const response = await fetch(`${this.baseUrl}${this.endpoints.processors}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
            
        } catch (error) {
            console.error('API getProcessors failed:', error);
            throw error;
        }
    }
}

// Update the main app to use API client
class AudioDigitApp {
    constructor() {
        // Initialize API client
        this.apiClient = new APIClient();
        
        // Core components (existing code structure maintained)
        this.audioRecorder = null;
        this.audioVisualizer = null;
        this.noiseGenerator = null;
        
        // UI elements
        this.elements = {
            startRecording: document.getElementById('startRecording'),
            stopRecording: document.getElementById('stopRecording'),
            clearCanvas: document.getElementById('clearCanvas'),
            recordingStatus: document.getElementById('recordingStatus'),
            audioInfo: document.getElementById('audioInfo'),
            totalPredictions: document.getElementById('totalPredictions'),
            fastestMethod: document.getElementById('fastestMethod'),
            sessionTime: document.getElementById('sessionTime'),
            audioCanvas: document.getElementById('audioCanvas'),
            apiLatency: document.getElementById('apiLatency')
        };
        
        // Application state
        this.state = {
            isRecording: false,
            hasRecordedAudio: false,
            currentAudioBlob: null,
            selectedMethod: 'ml_mfcc',
            totalPredictions: 0,
            methodStats: {},
            sessionStartTime: Date.now(),
            streamingErrors: 0,
            maxStreamingErrors: 5,
            lastErrorTime: 0,
            apiConnected: false
        };
        
        this.initialize();
    }

    async initialize() {
        try {
            console.log('[INFO] Initializing Audio Digit Classifier for HF Spaces...');
            
            // Check API connectivity first
            await this.testAPIConnection();
            
            // Initialize components (simplified for frontend-only)
            await this.initializeComponents();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize UI state
            this.updateUIState();
            
            console.log('[SUCCESS] Application initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize application:', error);
        }
    }

    async testAPIConnection() {
        try {
            console.log('[INFO] Testing API connection...');
            
            const health = await this.apiClient.checkHealth();
            
            if (health.status === 'healthy') {
                this.state.apiConnected = true;
                console.log('[SUCCESS] API connection established');
                
                // Update API status in UI
                const statusElement = document.getElementById('apiStatus');
                if (statusElement) {
                    statusElement.textContent = 'Connected';
                    statusElement.className = 'status-indicator ready';
                }
                
                // Load and display processor status
                await this.loadProcessorStatus();
                
            } else {
                throw new Error('API unhealthy');
            }
            
        } catch (error) {
            console.error('API connection failed:', error);
            this.state.apiConnected = false;
            
            const statusElement = document.getElementById('apiStatus');
            if (statusElement) {
                statusElement.textContent = 'Failed';
                statusElement.className = 'status-indicator error';
            }
            
            throw error;
        }
    }

    async loadProcessorStatus() {
        try {
            const processors = await this.apiClient.getProcessors();
            
            // Update status indicators for each processor
            Object.keys(processors).forEach(processorKey => {
                const processor = processors[processorKey];
                const statusElement = document.getElementById(`status_${processorKey}`);
                
                if (statusElement) {
                    if (processor.configured) {
                        statusElement.className = 'status-indicator ready';
                    } else {
                        statusElement.className = 'status-indicator error';
                    }
                }
            });
            
            console.log('[INFO] Processor status loaded:', processors);
            
        } catch (error) {
            console.error('Failed to load processor status:', error);
        }
    }

    async initializeComponents() {
        // Initialize basic audio recorder (using existing audio-recorder.js)
        if (typeof AudioRecorder !== 'undefined') {
            this.audioRecorder = new AudioRecorder();
            this.setupAudioRecorderCallbacks();
        }
        
        // Initialize audio visualizer (using existing audio-visualizer.js)
        if (typeof AudioVisualizer !== 'undefined') {
            this.audioVisualizer = new AudioVisualizer(this.elements.audioCanvas, {
                waveColor: '#00ff00',
                backgroundColor: '#001100',
                showGrid: true,
                retroGlow: true
            });
        }
        
        // Initialize noise generator (using existing noise-generator.js)
        if (typeof NoiseGenerator !== 'undefined') {
            this.noiseGenerator = new NoiseGenerator();
            await this.noiseGenerator.initialize();
        }
        
        console.log('[INFO] Core components initialized');
    }

    setupAudioRecorderCallbacks() {
        if (!this.audioRecorder) return;
        
        this.audioRecorder.onStart = () => {
            this.state.isRecording = true;
            this.updateRecordingState();
            console.log('[INFO] Recording started');
        };
        
        this.audioRecorder.onStop = (audioBlob, duration) => {
            this.state.isRecording = false;
            this.state.hasRecordedAudio = true;
            this.state.currentAudioBlob = audioBlob;
            this.updateRecordingState();
            this.updateAudioInfo(duration);
            
            // Auto-process the recorded audio
            this.processRecordedAudio();
            console.log('[INFO] Recording stopped');
        };
        
        this.audioRecorder.onError = (error) => {
            console.error('[ERROR] Recording error:', error);
            this.state.isRecording = false;
            this.updateRecordingState();
        };
    }

    setupEventListeners() {
        // Recording controls
        this.elements.startRecording.addEventListener('click', () => this.startRecording());
        this.elements.stopRecording.addEventListener('click', () => this.stopRecording());
        this.elements.clearCanvas.addEventListener('click', () => this.clearVisualization());
        
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
        
        console.log('[INFO] Event listeners registered');
    }

    async startRecording() {
        try {
            if (this.state.isRecording || !this.state.apiConnected) return;
            
            // Check if recorder is available
            if (!this.audioRecorder || typeof this.audioRecorder.start !== 'function') {
                console.warn('[WARN] Audio recorder not available, using simplified recording');
                return;
            }
            
            await this.audioRecorder.start();
            
            // Start visualization if available
            if (this.audioVisualizer && typeof this.audioVisualizer.start === 'function') {
                this.audioVisualizer.start(this.audioRecorder);
            }
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            if (error.message.includes('Permission denied') || error.message.includes('NotAllowedError')) {
                this.showMicrophonePermissionHelp();
            }
        }
    }

    stopRecording() {
        if (!this.state.isRecording || !this.audioRecorder) return;
        
        this.audioRecorder.stop();
        
        if (this.audioVisualizer && typeof this.audioVisualizer.stop === 'function') {
            this.audioVisualizer.stop();
        }
        
        console.log('[INFO] Recording stopped manually');
    }

    clearVisualization() {
        if (this.audioVisualizer && typeof this.audioVisualizer.clear === 'function') {
            this.audioVisualizer.clear();
        }
        console.log('[INFO] Visualization cleared');
    }

    async processRecordedAudio() {
        if (!this.state.hasRecordedAudio || !this.state.currentAudioBlob || !this.state.apiConnected) {
            console.warn('[WARN] Cannot process audio - missing audio or API connection');
            return;
        }

        try {
            console.log('[INFO] Processing recorded audio...');
            
            const startTime = Date.now();
            
            // Get noise settings if configured
            const noiseType = document.getElementById('noiseType')?.value || 'none';
            const noiseLevel = document.getElementById('noiseLevel')?.value || '0';
            
            const result = await this.apiClient.processAudio(
                this.state.currentAudioBlob, 
                this.state.selectedMethod,
                {
                    noiseType,
                    noiseLevel,
                    filename: 'recording.wav'
                }
            );
            
            const apiLatency = Date.now() - startTime;
            this.updateAPILatency(apiLatency);
            
            if (result.success !== false) {
                this.displayResults(result);
                this.updateStats(result);
                console.log(`[SUCCESS] Predicted digit: ${result.predicted_digit} (${result.inference_time}s)`);
            } else {
                throw new Error(result.error || 'Processing failed');
            }
            
        } catch (error) {
            console.error('Audio processing failed:', error);
        }
    }

    displayResults(result) {
        // Use the global updatePredictionDisplay function from the HTML
        if (typeof window.updatePredictionDisplay === 'function') {
            window.updatePredictionDisplay(this.state.selectedMethod, result);
        }
        
        // Update cabinet status
        this.updateCabinetStatus(this.state.selectedMethod, 'working');
        setTimeout(() => {
            this.updateCabinetStatus(this.state.selectedMethod, 'ready');
        }, 1000);
    }

    updateStats(result) {
        this.state.totalPredictions++;
        if (this.elements.totalPredictions) {
            this.elements.totalPredictions.textContent = this.state.totalPredictions;
        }
        
        // Update method stats
        if (!this.state.methodStats[result.method || this.state.selectedMethod]) {
            this.state.methodStats[result.method || this.state.selectedMethod] = {
                predictions: 0,
                totalTime: 0,
                errors: 0
            };
        }
        
        const methodStats = this.state.methodStats[result.method || this.state.selectedMethod];
        methodStats.predictions++;
        methodStats.totalTime += result.inference_time || 0;
        
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
        
        if (fastestMethod && this.elements.fastestMethod) {
            this.elements.fastestMethod.textContent = this.getMethodName(fastestMethod);
        }
    }

    updateAPILatency(latency) {
        if (this.elements.apiLatency) {
            this.elements.apiLatency.textContent = `${latency} ms`;
        }
    }

    updateRecordingState() {
        if (this.state.isRecording) {
            this.elements.startRecording.disabled = true;
            this.elements.stopRecording.disabled = false;
            this.elements.recordingStatus.textContent = 'Recording... (Press SPACE or click stop)';
            this.elements.recordingStatus.style.color = '#ff0000';
        } else {
            this.elements.startRecording.disabled = false;
            this.elements.stopRecording.disabled = true;
            this.elements.recordingStatus.textContent = 'Ready to record... (Press SPACE or click start)';
            this.elements.recordingStatus.style.color = '#00ff00';
        }
    }

    updateAudioInfo(duration) {
        if (this.elements.audioInfo) {
            this.elements.audioInfo.textContent = `Duration: ${(duration / 1000).toFixed(1)}s`;
        }
    }

    updateUIState() {
        console.log('UI state updated:', {
            hasRecordedAudio: this.state.hasRecordedAudio,
            isRecording: this.state.isRecording,
            selectedMethod: this.state.selectedMethod,
            apiConnected: this.state.apiConnected
        });
    }

    updateCabinetStatus(method, status) {
        const cabinet = document.querySelector(`[data-method="${method}"]`);
        if (cabinet) {
            const indicator = cabinet.querySelector('.status-indicator');
            if (indicator) {
                indicator.className = `status-indicator ${status}`;
            }
        }
    }

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
        console.log('[INFO] Microphone permission help displayed');
    }

    getMethodName(method) {
        const names = {
            ml_mfcc: 'MFCC + Dense NN',
            ml_mel_cnn: 'Mel CNN (2D)',
            ml_raw_cnn: 'Raw CNN (1D)', 
            external_api: 'External API',
            whisper_digit: 'Whisper Digit'
        };
        return names[method] || method;
    }

    cleanup() {
        if (this.audioRecorder && typeof this.audioRecorder.cleanup === 'function') {
            this.audioRecorder.cleanup();
        }
        
        if (this.audioVisualizer && typeof this.audioVisualizer.stop === 'function') {
            this.audioVisualizer.stop();
        }
        
        if (this.noiseGenerator && typeof this.noiseGenerator.cleanup === 'function') {
            this.noiseGenerator.cleanup();
        }
    }
}

// Make APIClient available globally
window.APIClient = APIClient;

// Initialize application when DOM is loaded
let app = null;

document.addEventListener('DOMContentLoaded', () => {
    app = new AudioDigitApp();
    
    // Make app globally available for debugging
    window.audioDigitApp = app;
    
    // Override the global selectMethod function to work with our app
    const originalSelectMethod = window.selectMethod;
    window.selectMethod = function(methodName) {
        // Call original UI function
        if (originalSelectMethod) {
            originalSelectMethod(methodName);
        }
        
        // Update application state
        if (app) {
            app.state.selectedMethod = methodName;
            console.log(`[INFO] Selected method: ${app.getMethodName(methodName)}`);
        }
    };
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (app) {
        app.cleanup();
    }
});

// Export for testing
window.AudioDigitApp = AudioDigitApp;