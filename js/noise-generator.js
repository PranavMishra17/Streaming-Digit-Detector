/**
 * Noise Generator Module
 * Client-side noise injection for testing audio robustness
 */

class NoiseGenerator {
    constructor() {
        this.audioContext = null;
        this.noiseBuffer = null;
        this.noiseTypes = {
            white: this.generateWhiteNoise.bind(this),
            pink: this.generatePinkNoise.bind(this),
            brown: this.generateBrownNoise.bind(this),
            background: this.generateBackgroundNoise.bind(this)
        };
        
        this.initialized = false;
    }
    
    /**
     * Initialize audio context
     */
    async initialize() {
        if (this.initialized) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.initialized = true;
            console.log('Noise generator initialized');
        } catch (error) {
            console.error('Failed to initialize noise generator:', error);
            throw error;
        }
    }
    
    /**
     * Generate white noise
     */
    generateWhiteNoise(duration, sampleRate = 16000, amplitude = 0.1) {
        const samples = Math.floor(duration * sampleRate);
        const buffer = new Float32Array(samples);
        
        for (let i = 0; i < samples; i++) {
            buffer[i] = (Math.random() * 2 - 1) * amplitude;
        }
        
        return buffer;
    }
    
    /**
     * Generate pink noise (1/f noise)
     */
    generatePinkNoise(duration, sampleRate = 16000, amplitude = 0.1) {
        const samples = Math.floor(duration * sampleRate);
        const buffer = new Float32Array(samples);
        
        // Simple pink noise approximation using multiple octaves
        let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
        
        for (let i = 0; i < samples; i++) {
            const white = Math.random() * 2 - 1;
            
            b0 = 0.99886 * b0 + white * 0.0555179;
            b1 = 0.99332 * b1 + white * 0.0750759;
            b2 = 0.96900 * b2 + white * 0.1538520;
            b3 = 0.86650 * b3 + white * 0.3104856;
            b4 = 0.55000 * b4 + white * 0.5329522;
            b5 = -0.7616 * b5 - white * 0.0168980;
            
            const pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
            b6 = white * 0.115926;
            
            buffer[i] = pink * amplitude * 0.11;
        }
        
        return buffer;
    }
    
    /**
     * Generate brown noise (1/f² noise)
     */
    generateBrownNoise(duration, sampleRate = 16000, amplitude = 0.1) {
        const samples = Math.floor(duration * sampleRate);
        const buffer = new Float32Array(samples);
        
        let lastOut = 0.0;
        
        for (let i = 0; i < samples; i++) {
            const white = Math.random() * 2 - 1;
            const brown = (lastOut + (0.02 * white)) / 1.02;
            lastOut = brown;
            
            buffer[i] = brown * amplitude * 3.5;
        }
        
        return buffer;
    }
    
    /**
     * Generate realistic background noise
     */
    generateBackgroundNoise(duration, sampleRate = 16000, amplitude = 0.05) {
        const samples = Math.floor(duration * sampleRate);
        const buffer = new Float32Array(samples);
        
        // Mix different noise types
        const white = this.generateWhiteNoise(duration, sampleRate, amplitude * 0.3);
        const pink = this.generatePinkNoise(duration, sampleRate, amplitude * 0.5);
        
        // Add some low-frequency rumble
        for (let i = 0; i < samples; i++) {
            const t = i / sampleRate;
            const rumble = Math.sin(2 * Math.PI * 60 * t) * amplitude * 0.2; // 60 Hz hum
            
            buffer[i] = white[i] + pink[i] + rumble;
        }
        
        return buffer;
    }
    
    /**
     * Mix noise with audio blob
     */
    async mixNoiseWithAudio(audioBlob, noiseType, noiseLevel) {
        if (!this.initialized) {
            await this.initialize();
        }
        
        if (noiseLevel <= 0 || noiseType === 'none') {
            return audioBlob;
        }
        
        try {
            // Decode original audio
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            const duration = audioBuffer.duration;
            const sampleRate = audioBuffer.sampleRate;
            const channels = audioBuffer.numberOfChannels;
            
            // Generate noise
            const noiseGenerator = this.noiseTypes[noiseType];
            if (!noiseGenerator) {
                throw new Error(`Unknown noise type: ${noiseType}`);
            }
            
            const noiseData = noiseGenerator(duration, sampleRate, noiseLevel);
            
            // Create new buffer with mixed audio
            const mixedBuffer = this.audioContext.createBuffer(
                channels,
                audioBuffer.length,
                sampleRate
            );
            
            // Mix audio with noise for each channel
            for (let channel = 0; channel < channels; channel++) {
                const originalData = audioBuffer.getChannelData(channel);
                const mixedData = mixedBuffer.getChannelData(channel);
                
                for (let i = 0; i < originalData.length; i++) {
                    const noiseIndex = Math.min(i, noiseData.length - 1);
                    mixedData[i] = originalData[i] + noiseData[noiseIndex];
                    
                    // Clip to prevent distortion
                    mixedData[i] = Math.max(-1, Math.min(1, mixedData[i]));
                }
            }
            
            // Convert back to blob
            return this.audioBufferToBlob(mixedBuffer);
            
        } catch (error) {
            console.error('Failed to mix noise with audio:', error);
            return audioBlob; // Return original if mixing fails
        }
    }
    
    /**
     * Convert AudioBuffer to Blob
     */
    audioBufferToBlob(audioBuffer) {
        const length = audioBuffer.length;
        const channels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        
        // Create interleaved array
        const interleavedArray = new Float32Array(length * channels);
        
        // Interleave channels
        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < channels; channel++) {
                const channelData = audioBuffer.getChannelData(channel);
                interleavedArray[i * channels + channel] = channelData[i];
            }
        }
        
        // Convert to WAV
        const wavBuffer = this.encodeWAV(interleavedArray, channels, sampleRate);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }
    
    /**
     * Encode PCM data as WAV
     */
    encodeWAV(samples, channels, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        const floatTo16BitPCM = (output, offset, input) => {
            for (let i = 0; i < input.length; i++, offset += 2) {
                const s = Math.max(-1, Math.min(1, input[i]));
                output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, channels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 4, true);
        view.setUint16(32, channels * 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);
        
        floatTo16BitPCM(view, 44, samples);
        
        return buffer;
    }
    
    /**
     * Create pure noise blob for testing
     */
    async createPureNoise(noiseType, duration = 1.0, amplitude = 0.3) {
        if (!this.initialized) {
            await this.initialize();
        }
        
        const sampleRate = 16000;
        const noiseGenerator = this.noiseTypes[noiseType];
        
        if (!noiseGenerator) {
            throw new Error(`Unknown noise type: ${noiseType}`);
        }
        
        // Generate noise data
        const noiseData = noiseGenerator(duration, sampleRate, amplitude);
        
        // Create audio buffer
        const audioBuffer = this.audioContext.createBuffer(1, noiseData.length, sampleRate);
        audioBuffer.copyToChannel(noiseData, 0);
        
        // Convert to blob
        return this.audioBufferToBlob(audioBuffer);
    }
    
    /**
     * Get available noise types
     */
    getAvailableNoiseTypes() {
        return Object.keys(this.noiseTypes);
    }
    
    /**
     * Test noise generation
     */
    async testNoise(noiseType, duration = 0.5) {
        try {
            const noiseBlob = await this.createPureNoise(noiseType, duration, 0.2);
            console.log(`Generated ${noiseType} noise: ${noiseBlob.size} bytes`);
            return noiseBlob;
        } catch (error) {
            console.error(`Failed to generate ${noiseType} noise:`, error);
            throw error;
        }
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }
        this.initialized = false;
    }
}

// Export for use in main application
window.NoiseGenerator = NoiseGenerator;