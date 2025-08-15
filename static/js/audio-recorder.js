/**
 * Audio Recorder Module
 * Handles microphone access, audio recording, and pause detection
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioContext = null;
        this.stream = null;
        this.isRecording = false;
        this.audioChunks = [];
        this.recordingStartTime = null;
        this.recordingDuration = 0;
        
        // Audio analysis for pause detection
        this.analyser = null;
        this.dataArray = null;
        this.silenceTimeout = null;
        this.minRecordingTime = 500; // 0.5 seconds minimum
        this.maxRecordingTime = 10000; // 10 seconds maximum
        this.silenceThreshold = 30; // Threshold for silence detection
        this.silenceDuration = 1500; // 1.5 seconds of silence to auto-stop
        
        this.onDataAvailable = null;
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onError = null;
        this.onSilenceDetected = null;
        
        // Bind methods
        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);
        this.handleDataAvailable = this.handleDataAvailable.bind(this);
        this.checkAudioLevel = this.checkAudioLevel.bind(this);
    }
    
    /**
     * Initialize audio recording with microphone access
     */
    async startRecording() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio context for analysis
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(this.stream);
            
            // Set up analyser for real-time audio analysis
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.3;
            source.connect(this.analyser);
            
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            // Set up MediaRecorder
            const options = {
                mimeType: this.getSupportedMimeType(),
                audioBitsPerSecond: 128000
            };
            
            this.mediaRecorder = new MediaRecorder(this.stream, options);
            this.audioChunks = [];
            
            // Set up event handlers
            this.mediaRecorder.addEventListener('dataavailable', this.handleDataAvailable);
            this.mediaRecorder.addEventListener('start', () => {
                this.isRecording = true;
                this.recordingStartTime = Date.now();
                this.startSilenceDetection();
                if (this.onRecordingStart) this.onRecordingStart();
                console.log('Recording started');
            });
            
            this.mediaRecorder.addEventListener('stop', () => {
                this.isRecording = false;
                this.recordingDuration = Date.now() - this.recordingStartTime;
                this.stopSilenceDetection();
                this.processRecordedAudio();
                if (this.onRecordingStop) this.onRecordingStop(this.recordingDuration);
                console.log(`Recording stopped after ${this.recordingDuration}ms`);
            });
            
            this.mediaRecorder.addEventListener('error', (event) => {
                console.error('MediaRecorder error:', event.error);
                if (this.onError) this.onError(event.error);
            });
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            
            // Auto-stop after maximum duration
            setTimeout(() => {
                if (this.isRecording) {
                    console.log('Auto-stopping recording after maximum duration');
                    this.stopRecording();
                }
            }, this.maxRecordingTime);
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            if (this.onError) {
                this.onError(new Error(`Microphone access failed: ${error.message}`));
            }
        }
    }
    
    /**
     * Stop audio recording
     */
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
        }
        
        this.cleanup();
    }
    
    /**
     * Handle recorded audio data chunks
     */
    handleDataAvailable(event) {
        if (event.data.size > 0) {
            this.audioChunks.push(event.data);
        }
    }
    
    /**
     * Process recorded audio and convert to appropriate format
     */
    async processRecordedAudio() {
        if (this.audioChunks.length === 0) {
            console.warn('No audio data recorded');
            return;
        }
        
        try {
            // Create blob from recorded chunks
            const audioBlob = new Blob(this.audioChunks, { 
                type: this.getSupportedMimeType() 
            });
            
            // Convert to WAV if needed (for better compatibility)
            let processedBlob = audioBlob;
            
            if (!this.getSupportedMimeType().includes('wav')) {
                // Convert to WAV using Web Audio API
                processedBlob = await this.convertToWAV(audioBlob);
            }
            
            if (this.onDataAvailable) {
                this.onDataAvailable(processedBlob, this.recordingDuration);
            }
            
        } catch (error) {
            console.error('Error processing recorded audio:', error);
            if (this.onError) this.onError(error);
        }
    }
    
    /**
     * Convert audio blob to WAV format
     */
    async convertToWAV(audioBlob) {
        try {
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to WAV
            const wavBuffer = this.audioBufferToWAV(audioBuffer);
            return new Blob([wavBuffer], { type: 'audio/wav' });
            
        } catch (error) {
            console.warn('WAV conversion failed, using original format:', error);
            return audioBlob;
        }
    }
    
    /**
     * Convert AudioBuffer to WAV format
     */
    audioBufferToWAV(audioBuffer) {
        const length = audioBuffer.length;
        const sampleRate = audioBuffer.sampleRate;
        const channels = audioBuffer.numberOfChannels;
        
        // Create WAV header
        const buffer = new ArrayBuffer(44 + length * channels * 2);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * channels * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // PCM format
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, channels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * channels * 2, true);
        view.setUint16(32, channels * 2, true);
        view.setUint16(34, 16, true); // 16-bit
        writeString(36, 'data');
        view.setUint32(40, length * channels * 2, true);
        
        // Convert audio data to 16-bit PCM
        let offset = 44;
        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < channels; channel++) {
                const sample = audioBuffer.getChannelData(channel)[i];
                const intSample = Math.max(-1, Math.min(1, sample)) * 0x7FFF;
                view.setInt16(offset, intSample, true);
                offset += 2;
            }
        }
        
        return buffer;
    }
    
    /**
     * Start silence detection for auto-stop functionality
     */
    startSilenceDetection() {
        if (!this.analyser) return;
        
        this.checkAudioLevel();
    }
    
    /**
     * Check audio level for silence detection
     */
    checkAudioLevel() {
        if (!this.isRecording || !this.analyser) return;
        
        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const average = sum / this.dataArray.length;
        
        // Check if audio is below silence threshold
        if (average < this.silenceThreshold) {
            if (!this.silenceTimeout) {
                // Start silence timer
                this.silenceTimeout = setTimeout(() => {
                    const recordingTime = Date.now() - this.recordingStartTime;
                    if (recordingTime > this.minRecordingTime && this.isRecording) {
                        console.log('Auto-stopping due to silence');
                        if (this.onSilenceDetected) this.onSilenceDetected();
                        this.stopRecording();
                    }
                }, this.silenceDuration);
            }
        } else {
            // Reset silence timer if audio detected
            if (this.silenceTimeout) {
                clearTimeout(this.silenceTimeout);
                this.silenceTimeout = null;
            }
        }
        
        // Continue checking
        if (this.isRecording) {
            requestAnimationFrame(this.checkAudioLevel);
        }
    }
    
    /**
     * Stop silence detection
     */
    stopSilenceDetection() {
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
        }
    }
    
    /**
     * Get supported MIME type for recording
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/wav'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        return 'audio/webm'; // Fallback
    }
    
    /**
     * Get current audio level for visualization
     */
    getCurrentAudioLevel() {
        if (!this.analyser || !this.dataArray) return 0;
        
        this.analyser.getByteFrequencyData(this.dataArray);
        
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        
        return sum / this.dataArray.length;
    }
    
    /**
     * Get frequency data for visualization
     */
    getFrequencyData() {
        if (!this.analyser) return null;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);
        
        return dataArray;
    }
    
    /**
     * Get time domain data for waveform visualization
     */
    getTimeDomainData() {
        if (!this.analyser) return null;
        
        const bufferLength = this.analyser.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);
        
        return dataArray;
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        this.stopSilenceDetection();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.mediaRecorder = null;
        this.analyser = null;
        this.dataArray = null;
    }
    
    /**
     * Check if recording is currently active
     */
    get recording() {
        return this.isRecording;
    }
    
    /**
     * Get recording duration in milliseconds
     */
    get duration() {
        if (this.isRecording && this.recordingStartTime) {
            return Date.now() - this.recordingStartTime;
        }
        return this.recordingDuration;
    }
}

// Export for use in main application
window.AudioRecorder = AudioRecorder;