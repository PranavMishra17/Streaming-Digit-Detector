/**
 * Audio Recorder Module
 * Handles microphone access, audio recording, and pause detection
 * Now using RecordRTC for more robust audio recording
 */

// RecordRTC should be loaded via HTML script tag

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.recordRTC = null;
        this.audioContext = null;
        this.stream = null;
        this.isRecording = false;
        this.audioChunks = [];
        this.recordingStartTime = null;
        this.recordingDuration = 0;
        
        // Audio analysis for streaming with VAD
        this.analyser = null;
        this.dataArray = null;
        this.streamingTimeout = null;
        this.isStreaming = false;
        this.audioChunksBuffer = [];
        
        // Voice Activity Detection - Optimized for digit recognition
        this.vadActive = false;
        this.speechStartTime = null;
        this.silenceFrames = 0;
        this.speechFrames = 0;
        this.minSpeechFrames = 2;      // 200ms minimum speech
        this.minSilenceFrames = 8;     // 800ms silence to end
        this.energyThreshold = 0.01;   // Lower threshold for better detection
        this.vadCheckInterval = 100;   // Check every 100ms
        
        // Audio collection settings for better digit recognition
        this.minChunkDuration = 300;   // Minimum 300ms chunks
        this.maxChunkDuration = 3000;  // Maximum 3s chunks
        
        this.onDataAvailable = null;
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onError = null;
        this.onChunkReady = null; // New callback for streaming chunks
        this.onVADResult = null; // Callback for VAD processing results
        this.onDigitDetected = null; // Callback for digit detection results
        
        // WebRTC VAD integration
        this.useWebRTCVAD = true; // Use backend WebRTC VAD instead of simple energy VAD
        this.streamingMethod = 'faster_whisper'; // Default method for streaming processing (try faster-whisper first)
        this.vadProcessingActive = false;
        
        // Session management
        this.sessionId = null;
        
        // Connection retry tracking
        this.connectionFailures = 0;
        this.maxConnectionFailures = 3;
        this.lastFailureTime = 0;
        this.backoffDelay = 5000; // 5 second backoff
        
        // Bind methods
        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);
        this.startVADMonitoring = this.startVADMonitoring.bind(this);
        this.checkVoiceActivity = this.checkVoiceActivity.bind(this);
        this.processSpeechSegment = this.processSpeechSegment.bind(this);
    }
    
    /**
     * Initialize audio recording with microphone access
     */
    async startRecording() {
        try {
            // Wait for RecordRTC to load if needed
            if (typeof RecordRTC === 'undefined') {
                console.log('Waiting for RecordRTC to load...');
                await this.waitForRecordRTC();
            }
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Set up audio context for analysis
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            const source = this.audioContext.createMediaStreamSource(this.stream);
            
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            source.connect(this.analyser);
            this.dataArray = new Uint8Array(this.analyser.fftSize);
            
            // Initialize RecordRTC with better options for chunked recording
            const options = {
                type: 'audio',
                mimeType: 'audio/webm', // Use WebM for proper chunk concatenation
                recorderType: RecordRTC.MediaStreamRecorder,
                sampleRate: 16000,
                numberOfAudioChannels: 1,
                disableLogs: false,
                timeSlice: 2000, // 2 second chunks instead of 1 second
                ondataavailable: (blob) => {
                    if (blob && blob.size > 0) {
                        console.log(`RecordRTC chunk received: ${blob.size} bytes`);
                        this.audioChunks.push(blob);
                        if (this.vadActive) {
                            this.audioChunksBuffer.push(blob);
                        }
                    }
                }
            };
            
            this.recordRTC = new RecordRTC(this.stream, options);
            
            // Initialize chunk arrays
            this.audioChunks = [];
            this.audioChunksBuffer = [];
            
            // Start recording
            this.recordRTC.startRecording();
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            this.startStreamingMode();
            
            if (this.onRecordingStart) this.onRecordingStart();
            console.log('RecordRTC streaming recording started');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            if (this.onError) {
                this.onError(new Error(`Microphone access failed: ${error.message}`));
            }
        }
    }
    
    /**
     * Wait for RecordRTC library to load
     */
    async waitForRecordRTC(maxWait = 5000) {
        const startTime = Date.now();
        while (typeof RecordRTC === 'undefined' && (Date.now() - startTime) < maxWait) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (typeof RecordRTC === 'undefined') {
            throw new Error('RecordRTC failed to load within timeout');
        }
    }
    
    /**
     * Stop audio recording
     */
    stopRecording() {
        if (this.recordRTC && this.isRecording) {
            this.recordRTC.stopRecording(() => {
                this.isRecording = false;
                this.recordingDuration = Date.now() - this.recordingStartTime;
                this.stopStreamingMode();
                if (this.onRecordingStop) this.onRecordingStop(this.recordingDuration);
                console.log(`RecordRTC recording stopped after ${this.recordingDuration}ms`);
            });
        }
        
        this.cleanup();
    }
    
    /**
     * Process recorded audio and convert to appropriate format
     */
    async processRecordedAudio() {
        if (!this.recordRTC) {
            console.warn('No RecordRTC instance available');
            return;
        }
        
        try {
            // Get the recorded blob from RecordRTC
            const audioBlob = this.recordRTC.getBlob();
            
            if (!audioBlob || audioBlob.size === 0) {
                console.warn('No audio data recorded');
                return;
            }
            
            console.log(`Processed audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);
            
            if (this.onDataAvailable) {
                this.onDataAvailable(audioBlob, this.recordingDuration);
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
            // Ensure we have a valid AudioContext
            if (!this.audioContext || this.audioContext.state === 'closed') {
                console.warn('AudioContext not available for WAV conversion');
                // Try to create a new one if needed
                if (!this.audioContext) {
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                        sampleRate: 16000
                    });
                }
                if (this.audioContext.state === 'suspended') {
                    await this.audioContext.resume();
                }
            }
            
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
     * Create minimal WAV file from audio blob
     */
    createMinimalWAV(audioBlob) {
        try {
            // Create a basic WAV header for the blob
            const sampleRate = 16000;
            const numChannels = 1;
            const bitsPerSample = 16;
            const dataSize = audioBlob.size;
            const totalSize = 36 + dataSize;
            
            const buffer = new ArrayBuffer(44);
            const view = new DataView(buffer);
            
            // WAV header
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            writeString(0, 'RIFF');
            view.setUint32(4, totalSize, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true);
            view.setUint16(32, numChannels * bitsPerSample / 8, true);
            view.setUint16(34, bitsPerSample, true);
            writeString(36, 'data');
            view.setUint32(40, dataSize, true);
            
            // Combine header with audio data
            return new Blob([buffer, audioBlob], { type: 'audio/wav' });
            
        } catch (error) {
            console.error('Failed to create minimal WAV:', error);
            return audioBlob; // Return original if all else fails
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
     * Start streaming mode with voice activity detection
     */
    startStreamingMode() {
        if (!this.analyser) return;
        
        this.isStreaming = true;
        this.vadActive = false;
        this.speechStartTime = null;
        this.silenceFrames = 0;
        this.speechFrames = 0;
        this.audioChunksBuffer = [];
        
        console.log('Started VAD-based streaming mode');
        this.startVADMonitoring();
    }
    
    /**
     * Start voice activity detection monitoring
     */
    startVADMonitoring() {
        if (!this.isStreaming) return;
        
        this.streamingTimeout = setTimeout(() => {
            if (this.isStreaming) {
                this.checkVoiceActivity();
                this.startVADMonitoring(); // Continue monitoring
            }
        }, this.vadCheckInterval);
    }
    
    /**
     * Check for voice activity and manage speech segments
     */
    checkVoiceActivity() {
        if (!this.analyser || !this.dataArray) return;
        
        // Get TIME DOMAIN data for proper VAD
        const bufferLength = this.analyser.fftSize;
        const timeData = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(timeData);
        
        // Calculate RMS energy properly
        let sumSquares = 0;
        for (let i = 0; i < bufferLength; i++) {
            const normalized = (timeData[i] - 128) / 128.0;
            sumSquares += normalized * normalized;
        }
        const rms = Math.sqrt(sumSquares / bufferLength);
        
        // More sensitive threshold for speech
        const isSpeech = rms > 0.01; // Lower threshold
        
        if (isSpeech) {
            this.speechFrames++;
            this.silenceFrames = 0;
            
            if (!this.vadActive && this.speechFrames >= 2) { // Faster activation
                this.vadActive = true;
                this.speechStartTime = Date.now();
                this.audioChunksBuffer = []; // Start fresh
                console.log(`Speech detected! RMS: ${rms.toFixed(4)}`);
            }
        } else {
            this.silenceFrames++;
            
            if (this.vadActive && this.silenceFrames >= 8) { // Faster deactivation
                const duration = Date.now() - this.speechStartTime;
                if (duration >= 300 && this.audioChunksBuffer.length > 0) { // Minimum 300ms
                    console.log(`Speech ended after ${duration}ms`);
                    this.processSpeechSegment();
                }
                this.vadActive = false;
                this.speechFrames = 0;
            }
        }
    }
    
    /**
     * Process detected speech segment with WebRTC VAD
     */
    async processSpeechSegment() {
        if (!this.audioChunksBuffer || this.audioChunksBuffer.length === 0) {
            console.log('No audio chunks to process');
            return;
        }
        
        try {
            // For RecordRTC chunks, they should already be proper WAV blobs
            let speechBlob;
            
            if (this.audioChunksBuffer.length === 1) {
                // Single chunk - use directly
                speechBlob = this.audioChunksBuffer[0];
            } else {
                // Multiple chunks - combine them
                // WebM chunks can be concatenated directly
                speechBlob = new Blob(this.audioChunksBuffer, { type: 'audio/webm' });
            }
            
            if (speechBlob.size < 1000) { // Minimum 1KB
                console.log('Speech segment too small:', speechBlob.size);
                return;
            }
            
            const duration = Date.now() - this.speechStartTime;
            console.log(`Processing ${this.audioChunksBuffer.length} RecordRTC chunks, ${speechBlob.size} bytes, ${duration}ms`);
            
            // Clear buffer after creating blob
            this.audioChunksBuffer = [];
            
            // Process the blob
            if (this.useWebRTCVAD) {
                await this.processWithBackendVAD(speechBlob, duration);
            } else {
                await this.processWithFrontendVAD(speechBlob, duration);
            }
            
        } catch (error) {
            console.error('Error processing speech segment:', error);
        }
    }
    
    /**
     * Process audio chunk using backend WebRTC VAD
     */
    async processWithBackendVAD(audioBlob, duration) {
        try {
            // Prepare form data for backend processing
            const formData = new FormData();
            formData.append('audio', audioBlob, 'audio_chunk.webm');
            formData.append('method', this.streamingMethod);
            
            // Add session ID if available
            if (this.sessionId) {
                formData.append('session_id', this.sessionId);
            }
            
            this.vadProcessingActive = true;
            
            // Send to backend VAD processing endpoint
            const response = await fetch('/process_audio_chunk', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Backend VAD processing failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Reset connection failure counter on success
            if (this.connectionFailures > 0) {
                this.connectionFailures = 0;
                console.log('Backend connection recovered, resetting failure counter');
            }
            
            console.log('Backend VAD processing result:', result);
            
            // Log detailed processing info
            if (result.has_fallback) {
                console.log('🔄 Fallback processing was used (VAD detected no speech)');
            }
            if (result.segments_detected > 0) {
                console.log(`📊 VAD detected ${result.segments_detected} speech segments`);
            }
            if (result.total_results > 0) {
                console.log(`✅ Generated ${result.total_results} prediction results`);
            }
            
            // Handle VAD results
            if (this.onVADResult) {
                this.onVADResult(result);
            }
            
            // Handle digit detection results
            if (result.results && result.results.length > 0) {
                for (const segmentResult of result.results) {
                    if (segmentResult.predicted_digit && segmentResult.success) {
                        console.log(`Digit detected: ${segmentResult.predicted_digit}`);
                        
                        // Add session information to the result
                        const enhancedResult = {
                            ...segmentResult,
                            chunks_saved: result.chunks_saved || 0,
                            session_id: result.session_id || this.sessionId
                        };
                        
                        if (this.onDigitDetected) {
                            this.onDigitDetected(enhancedResult);
                        }
                    }
                }
            }
            
        } catch (error) {
            console.error('Backend VAD processing error:', error);
            
            // Track connection failures
            this.connectionFailures++;
            this.lastFailureTime = Date.now();
            
            // If too many failures, disable backend VAD temporarily
            if (this.connectionFailures >= this.maxConnectionFailures) {
                const timeSinceLastFailure = Date.now() - this.lastFailureTime;
                if (timeSinceLastFailure < this.backoffDelay) {
                    console.log(`Too many connection failures (${this.connectionFailures}), backing off for ${this.backoffDelay/1000}s`);
                    this.useWebRTCVAD = false;
                    // Re-enable after backoff period
                    setTimeout(() => {
                        this.connectionFailures = 0;
                        this.useWebRTCVAD = true;
                        console.log('Re-enabling backend VAD after backoff period');
                    }, this.backoffDelay);
                }
            }
            
            // Fallback to frontend processing
            console.log('Falling back to frontend processing...');
            await this.processWithFrontendVAD(audioBlob, duration);
            
        } finally {
            this.vadProcessingActive = false;
        }
    }
    
    /**
     * Process audio chunk using original frontend method
     */
    async processWithFrontendVAD(audioBlob, duration) {
        try {
            // Try to convert to proper WAV format
            let processedBlob;
            try {
                // First try proper audio decoding and WAV conversion
                processedBlob = await this.convertToWAV(audioBlob);
                console.log('Speech segment WAV conversion successful');
            } catch (error) {
                console.warn('Speech segment WAV conversion failed, sending original format:', error.message);
                // Send original blob and let server handle conversion
                processedBlob = audioBlob;
            }
            
            // Notify about new speech segment
            if (this.onChunkReady) {
                this.onChunkReady(processedBlob, duration);
            }
            
        } catch (error) {
            console.error('Error processing speech segment with frontend VAD:', error);
        }
    }
    
    /**
     * Stop streaming mode
     */
    stopStreamingMode() {
        this.isStreaming = false;
        this.vadActive = false;
        
        if (this.streamingTimeout) {
            clearTimeout(this.streamingTimeout);
            this.streamingTimeout = null;
        }
        
        // Process any remaining speech segment
        if (this.audioChunksBuffer.length > 0) {
            console.log('Processing final speech segment on stop');
            this.processSpeechSegment();
        }
        
        console.log('VAD-based streaming mode stopped');
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
        this.stopStreamingMode();
        
        if (this.recordRTC) {
            if (this.recordRTC.state === 'recording') {
                this.recordRTC.stopRecording();
            }
            this.recordRTC.destroy();
            this.recordRTC = null;
        }
        
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
    
    /**
     * Configure streaming processing method
     */
    setStreamingMethod(method) {
        this.streamingMethod = method;
        console.log(`Streaming method set to: ${method}`);
    }
    
    /**
     * Toggle between WebRTC VAD and frontend VAD
     */
    setVADMode(useWebRTC) {
        this.useWebRTCVAD = useWebRTC;
        console.log(`VAD mode set to: ${useWebRTC ? 'Backend WebRTC' : 'Frontend Energy'}`);
    }
    
    /**
     * Get current VAD status from backend
     */
    async getVADStatus() {
        try {
            const response = await fetch('/vad_status');
            if (response.ok) {
                const status = await response.json();
                return status;
            }
        } catch (error) {
            console.error('Error getting VAD status:', error);
        }
        return null;
    }
    
    /**
     * Reset VAD state on backend
     */
    async resetVADState() {
        try {
            const response = await fetch('/reset_vad', { method: 'POST' });
            if (response.ok) {
                const result = await response.json();
                console.log('VAD state reset:', result.message);
                return true;
            }
        } catch (error) {
            console.error('Error resetting VAD state:', error);
        }
        return false;
    }
    
    /**
     * Check if VAD processing is currently active
     */
    get isVADProcessing() {
        return this.vadProcessingActive;
    }
    
    /**
     * Set session ID for audio chunk saving
     */
    setSessionId(sessionId) {
        this.sessionId = sessionId;
        console.log(`AudioRecorder session ID set to: ${sessionId}`);
    }
    
    /**
     * Clear session ID
     */
    clearSessionId() {
        this.sessionId = null;
        console.log('AudioRecorder session ID cleared');
    }
}

// Export for use in main application
window.AudioRecorder = AudioRecorder;