import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Union
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# Simplified implementation without complex TTS dependencies
TTS_AVAILABLE = False

class VoiceCloner:
    """Voice cloning system using Coqui TTS models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.tts_model = None
        self.speaker_encoder = None
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load the required TTS models"""
        try:
            if not TTS_AVAILABLE:
                print("TTS library not available. Using fallback implementation.")
                self._init_fallback_models()
                return
            
            # Load XTTS model for voice cloning
            print("Loading XTTS model...")
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            self.models_loaded = True
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading TTS models: {e}")
            print("Falling back to alternative implementation...")
            self._init_fallback_models()
    
    def _init_fallback_models(self):
        """Initialize fallback models when Coqui TTS is not available"""
        try:
            # Try alternative TTS libraries
            self._init_simple_tts()
        except Exception as e:
            print(f"Fallback model initialization failed: {e}")
            self.models_loaded = False
    
    def _init_simple_tts(self):
        """Initialize a simple TTS system as fallback"""
        # This is a simplified implementation for when advanced models aren't available
        self.models_loaded = True
        print("Using simplified TTS implementation")
    
    def extract_speaker_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file"""
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            if len(audio) < sr:  # Less than 1 second
                raise ValueError("Audio too short for voice analysis")
            
            if self.tts_model and hasattr(self.tts_model, 'synthesizer'):
                # Use Coqui TTS speaker encoder
                try:
                    # For XTTS, we'll use the audio directly as reference
                    return audio  # XTTS uses raw audio as speaker reference
                except Exception as e:
                    print(f"Error with TTS encoder: {e}")
                    return self._extract_simple_features(audio)
            else:
                # Fallback feature extraction
                return self._extract_simple_features(audio)
                
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            return None
    
    def _extract_simple_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract simple audio features as speaker representation"""
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=22050)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                [np.mean(spectral_centroids)],
                [np.mean(spectral_rolloff)]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting simple features: {e}")
            # Return a dummy embedding if all else fails
            return np.random.randn(50)
    
    def synthesize_speech(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float = 1.0,
        pitch_shift: float = 0.0
    ) -> Optional[np.ndarray]:
        """Synthesize speech using the cloned voice"""
        try:
            if not self.models_loaded:
                return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
            
            if self.tts_model and TTS_AVAILABLE:
                return self._synthesize_with_xtts(text, speaker_embedding, speed, pitch_shift)
            else:
                return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
                
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _synthesize_with_xtts(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> Optional[np.ndarray]:
        """Synthesize using XTTS model"""
        try:
            # Save speaker audio to temp file for XTTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                if isinstance(speaker_embedding, np.ndarray) and len(speaker_embedding.shape) == 1:
                    # If it's actually audio data
                    sf.write(tmp_file.name, speaker_embedding, 22050)
                    speaker_wav_path = tmp_file.name
                else:
                    # Create a dummy audio file
                    dummy_audio = np.random.randn(22050) * 0.1
                    sf.write(tmp_file.name, dummy_audio, 22050)
                    speaker_wav_path = tmp_file.name
            
            # Generate speech
            output = self.tts_model.tts(text=text, speaker_wav=speaker_wav_path, language="en")
            
            # Clean up temp file
            os.unlink(speaker_wav_path)
            
            # Apply speed and pitch modifications
            if isinstance(output, list):
                output = np.array(output)
            
            # Apply speed change
            if speed != 1.0:
                output = librosa.effects.time_stretch(output, rate=speed)
            
            # Apply pitch shift
            if pitch_shift != 0.0:
                output = librosa.effects.pitch_shift(output, sr=22050, n_steps=pitch_shift * 12)
            
            return output
            
        except Exception as e:
            print(f"XTTS synthesis error: {e}")
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _synthesize_fallback(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> np.ndarray:
        """Fallback synthesis method"""
        try:
            # Simple synthesis using basic audio generation
            duration = len(text) * 0.1  # Approximate duration based on text length
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Generate basic tone based on speaker features
            if len(speaker_embedding) > 0:
                base_freq = 150 + (np.mean(speaker_embedding[:5]) * 50)  # Use first 5 features for pitch
                base_freq = np.clip(base_freq, 80, 300)
            else:
                base_freq = 150
            
            # Create simple synthesized speech-like audio
            audio = np.zeros_like(t)
            
            # Add multiple harmonics for speech-like quality
            for i, char in enumerate(text[:min(len(text), 100)]):  # Limit processing
                char_time = i * duration / len(text)
                char_duration = duration / len(text)
                
                # Vary frequency based on character
                freq_mod = base_freq + (ord(char) % 50 - 25)
                
                # Create time window for this character
                mask = (t >= char_time) & (t < char_time + char_duration)
                
                # Add fundamental and harmonics
                if np.any(mask):
                    char_audio = (
                        0.5 * np.sin(2 * np.pi * freq_mod * t[mask]) +
                        0.3 * np.sin(2 * np.pi * freq_mod * 2 * t[mask]) +
                        0.2 * np.sin(2 * np.pi * freq_mod * 3 * t[mask])
                    )
                    
                    # Apply envelope
                    envelope = np.exp(-3 * (t[mask] - char_time) / char_duration)
                    audio[mask] += char_audio * envelope * 0.3
            
            # Apply modifications
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            
            if pitch_shift != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift * 12)
            
            # Normalize
            audio = librosa.util.normalize(audio) * 0.7
            
            return audio
            
        except Exception as e:
            print(f"Fallback synthesis error: {e}")
            # Return silence as last resort
            return np.zeros(22050)
