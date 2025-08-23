/**
 * VAD-enabled Audio Recorder using @ricky0123/vad-web
 * Provides automatic voice activity detection for hands-free recording
 */

class VADAudioRecorder {
    constructor() {
        this.stream = null;
        this.isListening = false;
        this.isRecording = false;
        this.vadModel = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        
        // Callbacks
        this.onSpeechStart = null;
        this.onSpeechEnd = null;
        this.onError = null;
        this.onChunkReady = null;
        
        // VAD settings
        this.vadOptions = {
            positiveSpeechThreshold: 0.5,
            negativeSpeechThreshold: 0.35,
            preSpeechPadFrames: 1,
            redemptionFrames: 8,
            frameSamples: 1536,
            minSpeechFrames: 4
        };
        
        // Recording state
        this.speechStartTime = null;
        this.lastAudioTime = Date.now();
        this.silenceTimeout = null;
        this.maxSilenceDuration = 2000; // 2 seconds of silence to stop
        
        console.log('[VAD] VAD Audio Recorder initialized');
    }
    
    async initialize() {
        try {
            // Initialize VAD model
            if (typeof vad !== 'undefined') {
                console.log('[VAD] Loading VAD model...');
                this.vadModel = await vad.MicVAD.new(this.vadOptions);
                console.log('[VAD] VAD model loaded successfully');
                return true;
            } else {
                console.warn('[VAD] VAD library not available');
                return false;
            }
        } catch (error) {
            console.error('[VAD] Failed to initialize VAD:', error);
            if (this.onError) this.onError(error);
            return false;
        }
    }
    
    async startListening() {
        try {
            if (this.isListening) return;
            
            console.log('[VAD] Starting VAD listening...');
            
            // Get microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Initialize audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            // Initialize VAD if not already done
            if (!this.vadModel) {
                const initialized = await this.initialize();
                if (!initialized) {
                    throw new Error('VAD initialization failed');
                }
            }
            
            // Start VAD processing
            await this.vadModel.start(this.stream);
            
            // Set up VAD callbacks
            this.vadModel.onSpeechStart = () => {
                console.log('[VAD] Speech detected - starting recording');
                this.startRecording();
            };
            
            this.vadModel.onSpeechEnd = (audio) => {
                console.log('[VAD] Speech ended - processing audio');
                this.stopRecording(audio);
            };
            
            this.vadModel.onVADMisfire = () => {
                console.log('[VAD] VAD misfire - ignoring');
            };
            
            this.isListening = true;
            console.log('[VAD] VAD listening started');
            
        } catch (error) {
            console.error('[VAD] Failed to start listening:', error);
            if (this.onError) this.onError(error);
        }
    }
    
    async stopListening() {
        try {
            if (!this.isListening) return;
            
            console.log('[VAD] Stopping VAD listening...');
            
            // Stop VAD
            if (this.vadModel) {
                await this.vadModel.pause();
            }
            
            // Stop any ongoing recording
            if (this.isRecording) {
                this.forceStopRecording();
            }
            
            // Clean up media stream
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
            
            // Clean up audio context
            if (this.audioContext && this.audioContext.state !== 'closed') {
                await this.audioContext.close();
                this.audioContext = null;
            }
            
            this.isListening = false;
            console.log('[VAD] VAD listening stopped');
            
        } catch (error) {
            console.error('[VAD] Error stopping VAD:', error);
        }
    }
    
    startRecording() {
        if (this.isRecording) return;
        
        this.isRecording = true;
        this.speechStartTime = Date.now();
        this.recordedChunks = [];
        
        // Set up MediaRecorder for backup audio capture
        try {
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.start(100); // Collect data every 100ms
            
        } catch (error) {
            console.warn('[VAD] MediaRecorder setup failed:', error);
        }
        
        // Start silence detection
        this.lastAudioTime = Date.now();
        this.startSilenceDetection();
        
        if (this.onSpeechStart) {
            this.onSpeechStart();
        }
        
        console.log('[VAD] Recording started');
    }
    
    stopRecording(vadAudio) {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        // Stop silence detection
        if (this.silenceTimeout) {
            clearTimeout(this.silenceTimeout);
            this.silenceTimeout = null;
        }
        
        // Stop MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        // Calculate duration
        const duration = Date.now() - this.speechStartTime;
        
        // Convert VAD audio to blob
        let audioBlob = null;
        if (vadAudio && vadAudio.length > 0) {
            // VAD audio is Float32Array, convert to WAV
            audioBlob = this.float32ArrayToWAVBlob(vadAudio, 16000);
        } else if (this.recordedChunks.length > 0) {
            // Fallback to MediaRecorder audio
            audioBlob = new Blob(this.recordedChunks, { type: 'audio/webm' });
        }
        
        if (audioBlob && this.onSpeechEnd) {
            this.onSpeechEnd(audioBlob, duration);
        }
        
        // Also trigger chunk callback for streaming
        if (audioBlob && this.onChunkReady) {
            this.onChunkReady(audioBlob, duration);
        }
        
        console.log(`[VAD] Recording stopped - duration: ${duration}ms`);
    }
    
    forceStopRecording() {
        if (this.isRecording) {
            this.stopRecording(null);
        }
    }
    
    startSilenceDetection() {
        const checkSilence = () => {
            if (!this.isRecording) return;
            
            const now = Date.now();
            const silenceDuration = now - this.lastAudioTime;
            
            if (silenceDuration > this.maxSilenceDuration) {
                console.log('[VAD] Max silence duration reached, stopping recording');
                this.forceStopRecording();
                return;
            }
            
            // Continue checking
            this.silenceTimeout = setTimeout(checkSilence, 500);
        };
        
        // Start the silence check
        this.silenceTimeout = setTimeout(checkSilence, this.maxSilenceDuration);
    }
    
    float32ArrayToWAVBlob(float32Array, sampleRate) {
        // Convert Float32Array to 16-bit PCM
        const length = float32Array.length;
        const arrayBuffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(arrayBuffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert float samples to 16-bit PCM
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
        
        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }
    
    cleanup() {
        this.stopListening();
    }
    
    // Compatibility methods for existing code
    async start() {
        return this.startListening();
    }
    
    stop() {
        this.forceStopRecording();
        return this.stopListening();
    }
}

// Make it globally available
window.VADAudioRecorder = VADAudioRecorder;