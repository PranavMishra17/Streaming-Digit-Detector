/**
 * VAD-Based Audio Recorder using @ricky0123/vad-web
 * Provides superior voice activity detection with Silero VAD model
 */

class VADAudioRecorder {
    constructor() {
        this.vad = null;
        this.isListening = false;
        this.isInitialized = false;
        this.sessionId = null;
        
        // Callback functions
        this.onSpeechStart = null;
        this.onSpeechEnd = null;
        this.onChunkReady = null;
        this.onError = null;
        this.onDigitDetected = null;
        
        // Configuration
        this.vadConfig = {
            preSpeechPadFrames: 10,     // Padding before speech (frames)
            redemptionFrames: 8,        // Frames to wait before ending speech
            frameSamples: 1536,         // Samples per frame
            minSpeechFrames: 3,         // Minimum frames for valid speech
            positiveSpeechThreshold: 0.5, // VAD confidence threshold
            negativeSpeechThreshold: 0.35
        };
        
        // Audio processing
        this.sampleRate = 16000;
        this.processingActive = false;
        
        // Audio analysis for visualization
        this.audioContext = null;
        this.analyser = null;
        this.stream = null;
        this.recordingStartTime = null;
        
        // Bind methods
        this.handleSpeechStart = this.handleSpeechStart.bind(this);
        this.handleSpeechEnd = this.handleSpeechEnd.bind(this);
        this.handleError = this.handleError.bind(this);
    }
    
    /**
     * Initialize the VAD system
     */
    async initialize() {
        try {
            console.log('Initializing VAD-based audio recorder...');
            
            // Check if VAD library is available
            if (typeof vad === 'undefined') {
                throw new Error('VAD library not loaded. Make sure @ricky0123/vad-web is included.');
            }
            
            // Set up audio context for visualization
            await this.setupAudioContext();
            
            // Initialize the VAD
            this.vad = await vad.MicVAD.new({
                onSpeechStart: this.handleSpeechStart,
                onSpeechEnd: this.handleSpeechEnd,
                onVADMisfire: () => {
                    console.log('VAD misfire detected - ignoring short noise');
                },
                ...this.vadConfig
            });
            
            this.isInitialized = true;
            console.log('VAD audio recorder initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize VAD:', error);
            this.handleError(error);
            throw error;
        }
    }
    
    /**
     * Start listening for speech
     */
    async startListening() {
        try {
            if (!this.isInitialized) {
                await this.initialize();
            }
            
            if (this.isListening) {
                console.warn('Already listening');
                return;
            }
            
            console.log('Starting VAD listening...');
            
            // If VAD was previously paused, restart it
            if (this.vad) {
                await this.vad.start();
            }
            
            this.isListening = true;
            this.recordingStartTime = Date.now();
            
            console.log('VAD listening started successfully');
            
        } catch (error) {
            console.error('Failed to start listening:', error);
            this.handleError(error);
            throw error;
        }
    }
    
    /**
     * Stop listening for speech
     */
    async stopListening() {
        try {
            if (!this.isListening) {
                console.warn('Not currently listening');
                return;
            }
            
            console.log('Stopping VAD listening...');
            
            if (this.vad) {
                await this.vad.pause();
            }
            
            this.isListening = false;
            this.recordingStartTime = null;
            console.log('VAD listening stopped successfully');
            
        } catch (error) {
            console.error('Failed to stop listening:', error);
            this.handleError(error);
        }
    }
    
    /**
     * Destroy the VAD instance and cleanup
     */
    async destroy() {
        try {
            if (this.isListening) {
                await this.stopListening();
            }
            
            if (this.vad) {
                // VAD cleanup if available
                this.vad = null;
            }
            
            // Clean up audio context and stream
            if (this.audioContext && this.audioContext.state !== 'closed') {
                this.audioContext.close();
                this.audioContext = null;
            }
            
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
            
            this.isInitialized = false;
            console.log('VAD audio recorder destroyed');
            
        } catch (error) {
            console.error('Error during VAD cleanup:', error);
        }
    }
    
    /**
     * Handle speech start event
     */
    handleSpeechStart() {
        console.log('🎤 Speech detected - recording started');
        
        if (this.onSpeechStart) {
            this.onSpeechStart();
        }
    }
    
    /**
     * Handle speech end event with audio data
     */
    async handleSpeechEnd(audioData) {
        try {
            console.log(`🔇 Speech ended - processing ${audioData.length} samples`);
            
            if (this.processingActive) {
                console.log('Previous audio still processing, skipping...');
                return;
            }
            
            this.processingActive = true;
            
            // Validate audio data
            if (!audioData || audioData.length === 0) {
                console.warn('Empty audio data received');
                return;
            }
            
            // Check minimum duration (e.g., 200ms minimum)
            const minSamples = this.sampleRate * 0.2; // 200ms
            if (audioData.length < minSamples) {
                console.log(`Audio too short: ${audioData.length} samples (min: ${minSamples})`);
                return;
            }
            
            // Convert Float32Array to WAV format
            const wavBuffer = this.encodeWAV(audioData, this.sampleRate);
            const audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });
            
            console.log(`Audio converted to WAV: ${audioBlob.size} bytes`);
            
            // Calculate duration
            const duration = (audioData.length / this.sampleRate) * 1000; // milliseconds
            
            // Trigger callbacks
            if (this.onSpeechEnd) {
                this.onSpeechEnd(audioBlob, duration);
            }
            
            if (this.onChunkReady) {
                this.onChunkReady(audioBlob, duration);
            }
            
        } catch (error) {
            console.error('Error processing speech end:', error);
            this.handleError(error);
        } finally {
            this.processingActive = false;
        }
    }
    
    /**
     * Convert Float32Array audio data to WAV format
     */
    encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        // RIFF chunk descriptor
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true); // File size - 8
        writeString(8, 'WAVE');
        
        // fmt sub-chunk
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // Subchunk1Size for PCM
        view.setUint16(20, 1, true);  // AudioFormat (PCM)
        view.setUint16(22, 1, true);  // NumChannels (mono)
        view.setUint32(24, sampleRate, true); // SampleRate
        view.setUint32(28, sampleRate * 2, true); // ByteRate
        view.setUint16(32, 2, true);  // BlockAlign
        view.setUint16(34, 16, true); // BitsPerSample
        
        // data sub-chunk
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true); // Subchunk2Size
        
        // Convert float samples to 16-bit PCM
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const sample = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
        
        return buffer;
    }
    
    /**
     * Convert ArrayBuffer to base64 string
     */
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    /**
     * Handle errors
     */
    handleError(error) {
        console.error('VAD Audio Recorder Error:', error);
        
        if (this.onError) {
            this.onError(error);
        }
    }
    
    /**
     * Get current status
     */
    getStatus() {
        return {
            isInitialized: this.isInitialized,
            isListening: this.isListening,
            processingActive: this.processingActive,
            vadAvailable: typeof vad !== 'undefined',
            sampleRate: this.sampleRate
        };
    }
    
    /**
     * Update VAD configuration
     */
    updateConfig(newConfig) {
        this.vadConfig = { ...this.vadConfig, ...newConfig };
        console.log('VAD configuration updated:', this.vadConfig);
    }
    
    /**
     * Set session ID for this recording session
     */
    setSessionId(sessionId) {
        this.sessionId = sessionId;
        console.log('VAD recorder session ID set:', sessionId);
    }
    
    /**
     * Get current audio level for visualization
     */
    getCurrentAudioLevel() {
        if (!this.analyser) {
            return 0;
        }
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        
        // Calculate RMS (Root Mean Square) for audio level
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sum / bufferLength);
        
        // Normalize to 0-1 range
        return rms / 255;
    }
    
    /**
     * Get time domain data for waveform visualization
     */
    getTimeDomainData() {
        if (!this.analyser) {
            return new Uint8Array(0);
        }
        
        const bufferLength = this.analyser.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);
        return dataArray;
    }
    
    /**
     * Get frequency data for spectrum visualization
     */
    getFrequencyData() {
        if (!this.analyser) {
            return new Uint8Array(0);
        }
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        return dataArray;
    }
    
    /**
     * Get recording property for compatibility
     */
    get recording() {
        return this.isListening;
    }
    
    /**
     * Get duration property for compatibility
     */
    get duration() {
        // Return approximate duration since listening started
        if (!this.isListening || !this.recordingStartTime) {
            return 0;
        }
        return Date.now() - this.recordingStartTime;
    }
    
    /**
     * Set up audio context for visualization
     */
    async setupAudioContext() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Set up audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            const source = this.audioContext.createMediaStreamSource(this.stream);
            
            // Create analyser for visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            source.connect(this.analyser);
            
            console.log('Audio context setup complete for visualization');
            
        } catch (error) {
            console.error('Failed to setup audio context:', error);
            throw error;
        }
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.VADAudioRecorder = VADAudioRecorder;
}