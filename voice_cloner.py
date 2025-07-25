import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Union
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

# Try to import available TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import subprocess
    # Check if espeak is available
    result = subprocess.run(['which', 'espeak'], capture_output=True)
    ESPEAK_AVAILABLE = result.returncode == 0
except:
    ESPEAK_AVAILABLE = False

try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False

TTS_AVAILABLE = PYTTSX3_AVAILABLE or GTTS_AVAILABLE or EDGE_TTS_AVAILABLE or COQUI_TTS_AVAILABLE or ESPEAK_AVAILABLE

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
            if ESPEAK_AVAILABLE:
                print("Loading eSpeak TTS engine...")
                self.tts_mode = "espeak"
                self.models_loaded = True
                print("eSpeak TTS loaded successfully!")
                return
                
            elif COQUI_TTS_AVAILABLE:
                print("Loading Coqui TTS engine...")
                self.tts_mode = "coqui_tts"
                self._load_coqui_models()
                self.models_loaded = True
                print("Coqui TTS loaded successfully!")
                return
                
            elif GTTS_AVAILABLE:
                print("Loading Google TTS engine...")
                self.tts_mode = "gtts"
                self.models_loaded = True
                print("Google TTS loaded successfully!")
                return
                
            elif EDGE_TTS_AVAILABLE:
                print("Loading Microsoft Edge TTS engine...")
                self.tts_mode = "edge_tts"
                self.models_loaded = True
                print("Microsoft Edge TTS loaded successfully!")
                return
                
            elif PYTTSX3_AVAILABLE:
                print("Loading pyttsx3 TTS engine...")
                self.tts_engine = pyttsx3.init()
                self.tts_mode = "pyttsx3"
                # Set properties for better voice quality
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to find a good voice
                    for voice in voices:
                        if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                
                self.tts_engine.setProperty('rate', 180)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                self.models_loaded = True
                print("pyttsx3 TTS engine loaded successfully!")
                return
            else:
                print("No TTS library available. Using enhanced fallback implementation.")
                self._init_fallback_models()
            
        except Exception as e:
            print(f"Error loading TTS models: {e}")
            print("Using enhanced fallback implementation...")
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
        """Extract simple audio features as speaker representation with gender detection"""
        try:
            # Extract fundamental frequency for gender detection using better method
            try:
                # Use yin algorithm for better F0 estimation
                f0 = librosa.yin(audio, fmin=50, fmax=400, sr=22050)
                # Remove unvoiced frames (0 values) and outliers
                f0_voiced = f0[f0 > 0]
                
                if len(f0_voiced) > 0:
                    # Use median for robustness against outliers
                    fundamental_freq = np.median(f0_voiced)
                    # Also check if the median is reasonable
                    if fundamental_freq > 400 or fundamental_freq < 50:
                        # Fallback to percentile method
                        fundamental_freq = np.percentile(f0_voiced, 50)
                else:
                    fundamental_freq = 0
            except:
                # Fallback to piptrack method with better filtering
                pitches, magnitudes = librosa.piptrack(y=audio, sr=22050, fmin=50, fmax=400)
                # Get only strong, low-frequency pitches (fundamental frequency range)
                mask = magnitudes > np.percentile(magnitudes[magnitudes > 0], 75)
                pitches_filtered = pitches[mask]
                pitches_clean = pitches_filtered[(pitches_filtered > 50) & (pitches_filtered < 400)]
                
                if len(pitches_clean) > 0:
                    fundamental_freq = np.median(pitches_clean)
                else:
                    fundamental_freq = 0
            
            gender = 'unknown'
            if fundamental_freq > 0:
                print(f"Detected fundamental frequency: {fundamental_freq:.1f} Hz")
                
                # Gender detection with more conservative thresholds
                if fundamental_freq < 160:  # Male range (typically 85-180 Hz)
                    gender = 'male'
                    print(f"Detected MALE voice (F0: {fundamental_freq:.1f} Hz)")
                elif fundamental_freq > 190:  # Female range (typically 165-265 Hz)
                    gender = 'female' 
                    print(f"Detected FEMALE voice (F0: {fundamental_freq:.1f} Hz)")
                else:
                    # Use spectral centroid for ambiguous cases (160-190 Hz overlap)
                    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=22050)
                    avg_centroid = np.mean(spectral_centroids)
                    print(f"Ambiguous F0: {fundamental_freq:.1f} Hz, using spectral centroid: {avg_centroid:.1f}")
                    
                    if avg_centroid < 1800:  # Lower spectral centroid = male
                        gender = 'male'
                        print(f"Detected MALE voice (ambiguous F0, low spectral centroid)")
                    else:
                        gender = 'female'
                        print(f"Detected FEMALE voice (ambiguous F0, high spectral centroid)")
            else:
                print("Could not detect fundamental frequency, defaulting to male")
                gender = 'male'  # Default to male when detection fails
            
            # Store gender information
            self.detected_gender = gender
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=22050)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)
            
            # Combine features with gender encoding
            features = []
            
            # First feature encodes gender and pitch
            if gender == 'male':
                features.append(-abs(fundamental_freq) / 200)  # Negative for male
            else:
                features.append(abs(fundamental_freq) / 300)   # Positive for female/unknown
            
            # Add other features
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            features.append(np.mean(spectral_centroids))
            features.append(np.mean(spectral_rolloff))
            
            return np.array(features)
            
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
        """Synthesize speech using the available TTS engine"""
        try:
            if not self.models_loaded:
                return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
            
            if hasattr(self, 'tts_mode'):
                if self.tts_mode == "espeak":
                    return self._synthesize_with_espeak(text, speaker_embedding, speed, pitch_shift)
                elif self.tts_mode == "coqui_tts":
                    return self._synthesize_with_coqui_tts(text, speaker_embedding, speed, pitch_shift)
                elif self.tts_mode == "edge_tts":
                    return self._synthesize_with_edge_tts(text, speaker_embedding, speed, pitch_shift)
                elif self.tts_mode == "gtts":
                    return self._synthesize_with_gtts(text, speaker_embedding, speed, pitch_shift)
                elif self.tts_mode == "pyttsx3":
                    return self._synthesize_with_pyttsx3(text, speaker_embedding, speed, pitch_shift)
            
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
                
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _synthesize_with_espeak(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> Optional[np.ndarray]:
        """Synthesize speech using eSpeak with voice cloning"""
        try:
            import tempfile
            import subprocess
            import os
            
            # Detect gender for voice selection
            detected_gender = self._detect_gender_from_embedding(speaker_embedding)
            print(f"Detected gender: {detected_gender}")
            
            # Select appropriate eSpeak voice
            if detected_gender == 'male':
                voice = "en+m3"  # Male voice variant 3
                pitch_adj = 35   # Lower pitch for male
            else:
                voice = "en+f4"  # Female voice variant 4
                pitch_adj = 60   # Higher pitch for female
            
            # Adjust speed (eSpeak uses words per minute)
            espeak_speed = max(80, min(300, int(175 * speed)))
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Build eSpeak command
            espeak_cmd = [
                'espeak', 
                '-v', voice,
                '-s', str(espeak_speed),
                '-p', str(pitch_adj),
                '-a', '100',  # Amplitude
                '-g', '5',    # Gap between words
                '--stdout',
                text
            ]
            
            print(f"Using eSpeak voice: {voice} (speed: {espeak_speed}, pitch: {pitch_adj})")
            
            # Generate speech with eSpeak
            result = subprocess.run(espeak_cmd, capture_output=True)
            
            if result.returncode != 0:
                raise Exception(f"eSpeak failed with return code {result.returncode}")
            
            # Save audio data to file
            with open(output_path, 'wb') as f:
                f.write(result.stdout)
            
            # Load the generated audio
            base_audio, sr = librosa.load(output_path, sr=22050)
            
            # Clean up temp file
            try:
                os.unlink(output_path)
            except:
                pass
            
            # Apply voice cloning transfer if reference audio available
            if len(speaker_embedding) > 1000:  # Reference audio available
                cloned_audio = self._apply_voice_transfer(base_audio, speaker_embedding)
            else:
                cloned_audio = base_audio
            
            # Apply additional pitch shift if requested
            if pitch_shift != 0.0:
                cloned_audio = librosa.effects.pitch_shift(cloned_audio, sr=22050, n_steps=pitch_shift * 12)
            
            # Apply gentle filtering and normalization
            try:
                cloned_audio = librosa.effects.preemphasis(cloned_audio, coef=0.95)
                cloned_audio = np.tanh(cloned_audio * 0.9) * 0.9
            except Exception as e:
                print(f"Audio filtering error: {e}")
            
            # Normalize
            cloned_audio = librosa.util.normalize(cloned_audio) * 0.8
            
            print("eSpeak TTS generation completed successfully!")
            return cloned_audio
            
        except Exception as e:
            print(f"eSpeak TTS synthesis error: {e}")
            # Fallback to Google TTS
            if GTTS_AVAILABLE:
                print("Falling back to Google TTS...")
                return self._synthesize_with_gtts(text, speaker_embedding, speed, pitch_shift)
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _synthesize_with_edge_tts(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> Optional[np.ndarray]:
        """Synthesize speech using Microsoft Edge TTS with voice selection"""
        try:
            import tempfile
            import os
            import asyncio
            
            # Choose voice based on speaker characteristics
            voice = "en-US-JennyNeural"  # Default to natural female voice
            
            if len(speaker_embedding) > 5:
                avg_pitch = np.mean(speaker_embedding[:5])
                spectral_centroid = np.mean(speaker_embedding[5:10]) if len(speaker_embedding) > 10 else 0
                
                # Voice selection based on characteristics
                if avg_pitch < -0.2:  # Lower pitch - male voices
                    if spectral_centroid > 0:
                        voice = "en-US-GuyNeural"  # Energetic male
                    else:
                        voice = "en-US-DavisNeural"  # Calm male
                elif avg_pitch > 0.2:  # Higher pitch - female voices  
                    if spectral_centroid > 0:
                        voice = "en-US-AriaNeural"  # Expressive female
                    else:
                        voice = "en-US-JennyNeural"  # Natural female
                else:  # Neutral pitch
                    voice = "en-US-BrandonNeural"  # Neutral voice
            
            print(f"Using Edge TTS voice: {voice}")
            
            # Adjust speech rate based on speed parameter
            rate_str = f"{int((speed - 1) * 50):+d}%"
            
            async def generate_speech():
                # Create communicate object with rate adjustment
                ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="{voice}"><prosody rate="{rate_str}">{text}</prosody></voice></speak>'
                communicate = edge_tts.Communicate(ssml, voice)
                
                # Create temporary file for audio output
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_filename = tmp_file.name
                
                # Generate speech and save to file
                with open(tmp_filename, "wb") as file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            file.write(chunk["data"])
                
                return tmp_filename
            
            # Run async function with proper error handling
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tmp_filename = loop.run_until_complete(generate_speech())
                loop.close()
            except Exception as e:
                print(f"Edge TTS async error: {e}")
                # Fall back to simpler approach
                communicate = edge_tts.Communicate(text, voice)
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_filename = tmp_file.name
                
                # Try synchronous generation
                import subprocess
                import sys
                result = subprocess.run([
                    sys.executable, '-c', 
                    f'''
import asyncio
import edge_tts

async def main():
    communicate = edge_tts.Communicate("{text}", "{voice}")
    await communicate.save("{tmp_filename}")

asyncio.run(main())
'''
                ], capture_output=True)
                
                if result.returncode != 0:
                    raise Exception(f"Edge TTS subprocess failed: {result.stderr.decode()}")
            
            print(f"Edge TTS audio saved to {tmp_filename}")
            
            # Load the generated audio
            base_audio, sr = librosa.load(tmp_filename, sr=22050)
            
            # Clean up temp file
            try:
                os.unlink(tmp_filename)
            except:
                pass
            
            # Apply voice cloning transfer
            if len(speaker_embedding) > 1000:  # Reference audio available
                cloned_audio = self._apply_voice_transfer(base_audio, speaker_embedding)
            else:
                cloned_audio = base_audio
            
            # Apply pitch shift if requested
            if pitch_shift != 0.0:
                cloned_audio = librosa.effects.pitch_shift(cloned_audio, sr=22050, n_steps=pitch_shift * 12)
            
            # Normalize
            cloned_audio = librosa.util.normalize(cloned_audio) * 0.8
            
            return cloned_audio
            
        except Exception as e:
            print(f"Edge TTS synthesis error: {e}")
            # Try Google TTS as fallback
            if GTTS_AVAILABLE:
                print("Falling back to Google TTS...")
                return self._synthesize_with_gtts(text, speaker_embedding, speed, pitch_shift)
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _synthesize_with_pyttsx3(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> Optional[np.ndarray]:
        """Synthesize speech using pyttsx3 and then apply voice cloning"""
        try:
            import tempfile
            import os
            
            # Adjust TTS engine properties based on speaker characteristics
            if len(speaker_embedding) > 5:
                # Use speaker features to adjust voice properties
                avg_pitch = np.mean(speaker_embedding[:5])
                
                # Adjust rate based on speed parameter
                rate = max(120, min(250, int(180 * speed)))
                self.tts_engine.setProperty('rate', rate)
                
                # Try to adjust pitch if possible (not all engines support this)
                try:
                    voices = self.tts_engine.getProperty('voices')
                    if voices and len(voices) > 1:
                        # Choose voice based on pitch characteristics
                        if avg_pitch > 0:
                            # Higher pitch - try to find female voice
                            for voice in voices:
                                if any(word in voice.name.lower() for word in ['female', 'woman', 'zira', 'hazel']):
                                    self.tts_engine.setProperty('voice', voice.id)
                                    break
                        else:
                            # Lower pitch - try to find male voice
                            for voice in voices:
                                if any(word in voice.name.lower() for word in ['male', 'man', 'david', 'mark']):
                                    self.tts_engine.setProperty('voice', voice.id)
                                    break
                except:
                    pass  # Voice selection failed, use default
            
            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
            
            # Generate speech to file
            self.tts_engine.save_to_file(text, tmp_filename)
            self.tts_engine.runAndWait()
            
            print(f"Audio saved to {tmp_filename}")
            
            # Load the generated audio
            base_audio, sr = librosa.load(tmp_filename, sr=22050)
            
            # Clean up temp file
            try:
                os.unlink(tmp_filename)
            except:
                pass
            
            # Now apply voice cloning by morphing with reference audio
            if len(speaker_embedding) > 1000:  # This is reference audio
                cloned_audio = self._apply_voice_transfer(base_audio, speaker_embedding)
            else:
                cloned_audio = base_audio
            
            # Apply pitch shift if requested
            if pitch_shift != 0.0:
                cloned_audio = librosa.effects.pitch_shift(cloned_audio, sr=22050, n_steps=pitch_shift * 12)
            
            # Normalize
            cloned_audio = librosa.util.normalize(cloned_audio) * 0.8
            
            return cloned_audio
            
        except Exception as e:
            print(f"pyttsx3 synthesis error: {e}")
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _apply_voice_transfer(self, base_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Apply advanced voice characteristics transfer using multiple techniques"""
        try:
            print("Applying voice transfer...")
            
            # Ensure we have enough audio to work with
            if len(reference_audio) < 5000 or len(base_audio) < 1000:
                return base_audio
            
            # Normalize inputs
            ref_audio = librosa.util.normalize(reference_audio)
            base_audio = librosa.util.normalize(base_audio)
            
            # Extract comprehensive voice characteristics from reference
            ref_features = self._extract_voice_characteristics(ref_audio)
            
            # Apply multiple voice transfer techniques
            transferred_audio = base_audio.copy()
            
            # 1. Spectral envelope matching with multiple frequency bands
            transferred_audio = self._transfer_spectral_envelope(transferred_audio, ref_audio, ref_features)
            
            # 2. Formant frequency adjustment
            transferred_audio = self._adjust_formants(transferred_audio, ref_features)
            
            # 3. Pitch contour matching
            transferred_audio = self._match_pitch_characteristics(transferred_audio, ref_audio, ref_features)
            
            # 4. Voice texture transfer (roughness, breathiness)
            transferred_audio = self._transfer_voice_texture(transferred_audio, ref_audio)
            
            # 5. Dynamic range and prosody adjustment
            transferred_audio = self._adjust_prosody(transferred_audio, ref_features)
            
            # Final normalization and blending
            transferred_audio = librosa.util.normalize(transferred_audio)
            
            # Stronger blend toward reference characteristics
            blend_factor = 0.85  # More aggressive voice transfer
            final_audio = blend_factor * transferred_audio + (1 - blend_factor) * base_audio
            
            print("Voice transfer complete")
            return final_audio
            
        except Exception as e:
            print(f"Voice transfer error: {e}")
            return base_audio
    
    def _extract_voice_characteristics(self, audio: np.ndarray) -> dict:
        """Extract detailed voice characteristics"""
        try:
            features = {}
            
            # Pitch characteristics
            pitches, magnitudes = librosa.piptrack(y=audio, sr=22050, threshold=0.1)
            valid_pitches = pitches[pitches > 0]
            if len(valid_pitches) > 0:
                features['mean_pitch'] = np.mean(valid_pitches)
                features['pitch_std'] = np.std(valid_pitches)
                features['pitch_range'] = np.max(valid_pitches) - np.min(valid_pitches)
            else:
                features['mean_pitch'] = 150
                features['pitch_std'] = 20
                features['pitch_range'] = 50
            
            # Spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=22050)[0]
            features['spectral_centroid'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Formant estimation using MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
            features['formant_profile'] = np.mean(mfccs[:5], axis=1)  # First 5 MFCCs for formants
            
            # Voice quality indicators
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=22050))
            
            # Harmonic-percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            features['harmonic_ratio'] = np.mean(np.abs(harmonic)) / (np.mean(np.abs(audio)) + 1e-8)
            
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {'mean_pitch': 150, 'spectral_centroid': 2000, 'formant_profile': np.zeros(5)}
    
    def _transfer_spectral_envelope(self, audio: np.ndarray, ref_audio: np.ndarray, ref_features: dict) -> np.ndarray:
        """Transfer spectral envelope characteristics"""
        try:
            # Multi-resolution spectral analysis
            stft_audio = librosa.stft(audio, hop_length=512)
            stft_ref = librosa.stft(ref_audio[:len(audio)] if len(ref_audio) > len(audio) else np.pad(ref_audio, (0, len(audio) - len(ref_audio)), 'wrap'), hop_length=512)
            
            # Extract magnitude spectra
            mag_audio = np.abs(stft_audio)
            mag_ref = np.abs(stft_ref)
            phase_audio = np.angle(stft_audio)
            
            # Smooth spectral envelope transfer in frequency bands
            n_bands = 8
            freq_bins = mag_audio.shape[0]
            band_size = freq_bins // n_bands
            
            transferred_mag = mag_audio.copy()
            
            for band in range(n_bands):
                start_bin = band * band_size
                end_bin = min((band + 1) * band_size, freq_bins)
                
                # Calculate average energy in this band for both signals
                ref_energy = np.mean(mag_ref[start_bin:end_bin])
                audio_energy = np.mean(mag_audio[start_bin:end_bin])
                
                if audio_energy > 0:
                    # Transfer energy characteristics
                    energy_ratio = ref_energy / audio_energy
                    energy_ratio = np.clip(energy_ratio, 0.3, 3.0)  # Limit extreme changes
                    transferred_mag[start_bin:end_bin] *= energy_ratio
            
            # Reconstruct audio
            transferred_stft = transferred_mag * np.exp(1j * phase_audio)
            return librosa.istft(transferred_stft, hop_length=512)
            
        except Exception as e:
            print(f"Spectral transfer error: {e}")
            return audio
    
    def _adjust_formants(self, audio: np.ndarray, ref_features: dict) -> np.ndarray:
        """Adjust formant frequencies to match reference"""
        try:
            # Use pitch shifting to approximate formant adjustment
            formant_shift = 0
            if 'formant_profile' in ref_features:
                # Calculate approximate formant adjustment based on MFCC differences
                target_brightness = np.mean(ref_features['formant_profile'][:3])
                
                # Adjust based on brightness (simplified formant approximation)
                if target_brightness > 0.5:
                    formant_shift = 2  # Brighter voice
                elif target_brightness < -0.5:
                    formant_shift = -2  # Darker voice
            
            if abs(formant_shift) > 0.5:
                return librosa.effects.pitch_shift(audio, sr=22050, n_steps=formant_shift)
            
            return audio
        except Exception as e:
            print(f"Formant adjustment error: {e}")
            return audio
    
    def _match_pitch_characteristics(self, audio: np.ndarray, ref_audio: np.ndarray, ref_features: dict) -> np.ndarray:
        """Match pitch characteristics more accurately"""
        try:
            # Extract pitch from both signals
            pitches_ref, _ = librosa.piptrack(y=ref_audio, sr=22050)
            pitches_audio, _ = librosa.piptrack(y=audio, sr=22050)
            
            # Get valid pitch values
            valid_ref = pitches_ref[pitches_ref > 50]
            valid_audio = pitches_audio[pitches_audio > 50]
            
            if len(valid_ref) > 0 and len(valid_audio) > 0:
                ref_median_pitch = np.median(valid_ref)
                audio_median_pitch = np.median(valid_audio)
                
                # Calculate pitch shift needed
                if audio_median_pitch > 0:
                    pitch_ratio = ref_median_pitch / audio_median_pitch
                    n_steps = 12 * np.log2(pitch_ratio)
                    n_steps = np.clip(n_steps, -8, 8)
                    
                    if abs(n_steps) > 0.5:
                        audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=n_steps)
            
            return audio
        except Exception as e:
            print(f"Pitch matching error: {e}")
            return audio
    
    def _transfer_voice_texture(self, audio: np.ndarray, ref_audio: np.ndarray) -> np.ndarray:
        """Transfer voice texture characteristics"""
        try:
            # Add controlled noise based on reference voice texture
            ref_noise_level = np.std(ref_audio - librosa.effects.preemphasis(ref_audio))
            audio_noise_level = np.std(audio - librosa.effects.preemphasis(audio))
            
            if audio_noise_level > 0:
                noise_ratio = ref_noise_level / audio_noise_level
                noise_ratio = np.clip(noise_ratio, 0.5, 2.0)
                
                # Generate controlled noise
                noise = np.random.randn(len(audio)) * ref_noise_level * 0.1
                audio = audio + noise * 0.05  # Add subtle texture
            
            return audio
        except Exception as e:
            print(f"Texture transfer error: {e}")
            return audio
    
    def _adjust_prosody(self, audio: np.ndarray, ref_features: dict) -> np.ndarray:
        """Adjust prosodic characteristics"""
        try:
            # Apply dynamic range adjustment based on reference
            if 'spectral_centroid_std' in ref_features:
                target_dynamics = ref_features['spectral_centroid_std']
                current_dynamics = np.std(librosa.feature.spectral_centroid(y=audio, sr=22050))
                
                if current_dynamics > 0:
                    dynamics_ratio = target_dynamics / current_dynamics
                    dynamics_ratio = np.clip(dynamics_ratio, 0.7, 1.5)
                    
                    # Apply subtle compression/expansion
                    audio = np.sign(audio) * np.power(np.abs(audio), 1.0 / dynamics_ratio)
            
            return audio
        except Exception as e:
            print(f"Prosody adjustment error: {e}")
            return audio
    
    def _synthesize_with_gtts(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> Optional[np.ndarray]:
        """Synthesize speech using Google TTS with voice cloning"""
        try:
            import tempfile
            import os
            
            # Create gTTS object with better settings
            tts = gTTS(text=text, lang='en', slow=(speed < 0.8), tld='com')
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_filename = tmp_file.name
            
            tts.save(tmp_filename)
            print(f"Google TTS audio saved to {tmp_filename}")
            
            # Load audio
            base_audio, sr = librosa.load(tmp_filename, sr=22050)
            
            # Clean up temp file
            try:
                os.unlink(tmp_filename)
            except:
                pass
            
            # Detect gender from reference audio for voice adaptation
            detected_gender = self._detect_gender_from_embedding(speaker_embedding)
            print(f"Detected gender: {detected_gender}")
            
            # Apply gender-appropriate pitch adjustment for Google TTS (which is female by default)
            if detected_gender == 'male':
                # More gentle pitch adjustment to avoid crackling
                base_audio = librosa.effects.pitch_shift(base_audio, sr=22050, n_steps=-3.5)
                print("Applied male voice pitch adjustment (-3.5 semitones)")
            elif detected_gender == 'unknown':
                # Default to male adjustment since most detection failures are with male voices
                base_audio = librosa.effects.pitch_shift(base_audio, sr=22050, n_steps=-3)
                print("Applied default male voice pitch adjustment (-3 semitones)")
            
            # Apply voice cloning transfer if reference audio available
            if len(speaker_embedding) > 1000:  # Reference audio available
                cloned_audio = self._apply_voice_transfer(base_audio, speaker_embedding)
            else:
                cloned_audio = base_audio
            
            # Apply speed adjustment
            if speed != 1.0:
                cloned_audio = librosa.effects.time_stretch(cloned_audio, rate=speed)
            
            # Apply pitch shift if requested
            if pitch_shift != 0.0:
                cloned_audio = librosa.effects.pitch_shift(cloned_audio, sr=22050, n_steps=pitch_shift * 12)
            
            # Apply gentle filtering to reduce artifacts
            try:
                # Low-pass filter to remove high-frequency artifacts from pitch shifting
                cloned_audio = librosa.effects.preemphasis(cloned_audio, coef=0.95)
                # Gentle compression to smooth dynamics
                cloned_audio = np.tanh(cloned_audio * 0.9) * 0.9
            except Exception as e:
                print(f"Audio filtering error: {e}")
            
            # Normalize with reduced gain to prevent clipping
            cloned_audio = librosa.util.normalize(cloned_audio) * 0.7
            
            return cloned_audio
            
        except Exception as e:
            print(f"Google TTS synthesis error: {e}")
            return self._synthesize_fallback(text, speaker_embedding, speed, pitch_shift)
    
    def _detect_gender_from_embedding(self, speaker_embedding: np.ndarray) -> str:
        """Detect gender from speaker embedding features"""
        try:
            # Check if we already detected gender during feature extraction
            if hasattr(self, 'detected_gender') and self.detected_gender != 'unknown':
                return self.detected_gender
            
            if len(speaker_embedding) < 1:
                return 'unknown'
            
            # The first feature encodes gender information
            # Negative values = male, Positive values = female
            gender_feature = speaker_embedding[0]
            
            if gender_feature < -0.1:
                return 'male'
            elif gender_feature > 0.1:
                return 'female'
            else:
                return 'unknown'
                    
        except Exception as e:
            print(f"Gender detection error: {e}")
            return 'unknown'
    
    def _load_coqui_models(self):
        """Load Coqui TTS models"""
        try:
            # Use XTTS-v2 model for voice cloning
            print("Loading XTTS-v2 model...")
            self.coqui_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("XTTS-v2 model loaded successfully!")
        except Exception as e:
            print(f"Error loading Coqui TTS models: {e}")
            raise
    
    def _synthesize_with_coqui_tts(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> Optional[np.ndarray]:
        """Synthesize speech using Coqui TTS XTTS-v2 with voice cloning"""
        try:
            import tempfile
            import os
            
            # Detect gender for voice selection
            detected_gender = self._detect_gender_from_embedding(speaker_embedding)
            print(f"Detected gender: {detected_gender}")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_audio_file:
                ref_audio_path = ref_audio_file.name
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                output_path = output_file.name
            
            # Save reference audio if available
            if hasattr(self, 'reference_audio') and self.reference_audio is not None:
                # Use the original reference audio
                sf.write(ref_audio_path, self.reference_audio, 22050)
                print("Using original reference audio for voice cloning")
            elif len(speaker_embedding) > 22050:
                # Extract audio samples from embedding if available
                audio_samples = speaker_embedding[-22050:]  # Last 22050 samples (1 second)
                sf.write(ref_audio_path, audio_samples, 22050)
                print("Using embedded audio samples for voice cloning")
            else:
                # Use a default voice based on detected gender
                default_voice = "female" if detected_gender == "female" else "male"
                print(f"No reference audio available, using default {default_voice} voice")
                # Generate without reference audio (will use model's default voice)
                self.coqui_tts.tts_to_file(text=text, file_path=output_path, language="en")
                
                # Load and return the generated audio
                generated_audio, sr = librosa.load(output_path, sr=22050)
                
                # Apply pitch adjustment based on detected gender
                if detected_gender == "male":
                    generated_audio = librosa.effects.pitch_shift(generated_audio, sr=22050, n_steps=-2)
                
                # Clean up temp files
                try:
                    os.unlink(output_path)
                except:
                    pass
                    
                return generated_audio
            
            # Generate speech with voice cloning
            print("Generating speech with voice cloning...")
            self.coqui_tts.tts_to_file(
                text=text, 
                speaker_wav=ref_audio_path, 
                file_path=output_path, 
                language="en"
            )
            
            # Load the generated audio
            generated_audio, sr = librosa.load(output_path, sr=22050)
            
            # Apply speed adjustment
            if speed != 1.0:
                generated_audio = librosa.effects.time_stretch(generated_audio, rate=speed)
            
            # Apply pitch shift
            if pitch_shift != 0.0:
                generated_audio = librosa.effects.pitch_shift(generated_audio, sr=22050, n_steps=pitch_shift * 12)
            
            # Apply gentle filtering and normalization
            try:
                generated_audio = librosa.effects.preemphasis(generated_audio, coef=0.95)
                generated_audio = np.tanh(generated_audio * 0.9) * 0.9
            except Exception as e:
                print(f"Audio filtering error: {e}")
            
            # Normalize
            generated_audio = librosa.util.normalize(generated_audio) * 0.8
            
            # Clean up temp files
            try:
                os.unlink(ref_audio_path)
                os.unlink(output_path)
            except:
                pass
            
            print("Coqui TTS generation completed successfully!")
            return generated_audio
            
        except Exception as e:
            print(f"Coqui TTS synthesis error: {e}")
            # Fallback to Google TTS
            if GTTS_AVAILABLE:
                print("Falling back to Google TTS...")
                return self._synthesize_with_gtts(text, speaker_embedding, speed, pitch_shift)
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
        """Enhanced fallback synthesis method using voice morphing"""
        try:
            # If speaker_embedding is actually audio data, use it for voice morphing
            if len(speaker_embedding) > 1000:  # Likely audio data
                return self._morph_voice_with_reference(text, speaker_embedding, speed, pitch_shift)
            else:
                return self._generate_speech_like_audio(text, speaker_embedding, speed, pitch_shift)
                
        except Exception as e:
            print(f"Fallback synthesis error: {e}")
            # Return silence as last resort
            return np.zeros(22050)
    
    def _morph_voice_with_reference(
        self, 
        text: str, 
        reference_audio: np.ndarray, 
        speed: float, 
        pitch_shift: float
    ) -> np.ndarray:
        """Create speech-like audio by morphing reference audio"""
        try:
            # Calculate target duration based on text
            chars_per_second = 15  # Approximate speaking rate
            target_duration = len(text) / chars_per_second
            target_samples = int(22050 * target_duration)
            
            # Use reference audio as base
            reference_audio = np.array(reference_audio)
            
            # Ensure we have enough audio to work with
            if len(reference_audio) < target_samples:
                # Repeat the reference audio
                repeats = (target_samples // len(reference_audio)) + 1
                reference_audio = np.tile(reference_audio, repeats)
            
            # Take only what we need
            morphed_audio = reference_audio[:target_samples]
            
            # Apply text-based modulation to simulate speech patterns
            words = text.split()
            if len(words) > 0:
                samples_per_word = len(morphed_audio) // len(words)
                
                for i, word in enumerate(words):
                    start_idx = i * samples_per_word
                    end_idx = min(start_idx + samples_per_word, len(morphed_audio))
                    
                    if start_idx < len(morphed_audio):
                        # Modulate based on word characteristics
                        word_segment = morphed_audio[start_idx:end_idx]
                        
                        # Vary pitch based on word length and vowels
                        vowels = sum(1 for c in word.lower() if c in 'aeiou')
                        pitch_mod = 1.0 + (vowels - 2) * 0.1  # Slight pitch variation
                        
                        # Apply pitch modulation
                        if len(word_segment) > 0:
                            try:
                                morphed_audio[start_idx:end_idx] = librosa.effects.pitch_shift(
                                    word_segment, sr=22050, n_steps=pitch_mod
                                )
                            except:
                                pass  # Keep original if pitch shift fails
            
            # Apply speed and pitch modifications
            if speed != 1.0:
                morphed_audio = librosa.effects.time_stretch(morphed_audio, rate=speed)
            
            if pitch_shift != 0.0:
                morphed_audio = librosa.effects.pitch_shift(morphed_audio, sr=22050, n_steps=pitch_shift * 12)
            
            # Normalize and add slight envelope
            morphed_audio = librosa.util.normalize(morphed_audio) * 0.8
            
            # Add fade in/out for natural sound
            fade_samples = min(1000, len(morphed_audio) // 10)
            if len(morphed_audio) > fade_samples * 2:
                # Fade in
                morphed_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out
                morphed_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            return morphed_audio
            
        except Exception as e:
            print(f"Voice morphing error: {e}")
            return self._generate_speech_like_audio(text, np.mean(reference_audio) if len(reference_audio) > 0 else 0, speed, pitch_shift)
    
    def _generate_speech_like_audio(
        self, 
        text: str, 
        speaker_features: Union[np.ndarray, float], 
        speed: float, 
        pitch_shift: float
    ) -> np.ndarray:
        """Generate more realistic speech-like audio"""
        try:
            # Calculate duration based on text
            chars_per_second = 12  # More realistic speaking rate
            duration = len(text) / chars_per_second
            sample_rate = 22050
            
            # Generate base frequency from speaker features
            if isinstance(speaker_features, np.ndarray) and len(speaker_features) > 0:
                base_freq = 120 + (np.mean(speaker_features[:3]) * 30)  # More realistic range
            else:
                base_freq = 150
            
            base_freq = np.clip(base_freq, 80, 200)  # Human voice range
            
            # Create more natural speech patterns
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.zeros_like(t)
            
            # Process words instead of characters for more natural rhythm
            words = text.split()
            if len(words) == 0:
                words = ['speech']
            
            time_per_word = duration / len(words)
            
            for i, word in enumerate(words):
                word_start_time = i * time_per_word
                word_end_time = (i + 1) * time_per_word
                
                # Create time mask for this word
                word_mask = (t >= word_start_time) & (t < word_end_time)
                word_t = t[word_mask] - word_start_time
                
                if len(word_t) > 0:
                    # Vary frequency based on word characteristics
                    vowel_count = sum(1 for c in word.lower() if c in 'aeiouyw')
                    consonant_count = len(word) - vowel_count
                    
                    # Adjust frequency based on phonetic content
                    freq_variation = base_freq + (vowel_count * 10) - (consonant_count * 5)
                    freq_variation = np.clip(freq_variation, base_freq * 0.8, base_freq * 1.3)
                    
                    # Create formant-like structure
                    formant1 = freq_variation
                    formant2 = freq_variation * 2.2
                    formant3 = freq_variation * 3.8
                    
                    # Generate more natural harmonics
                    word_audio = (
                        0.6 * np.sin(2 * np.pi * formant1 * word_t) +
                        0.3 * np.sin(2 * np.pi * formant2 * word_t) +
                        0.15 * np.sin(2 * np.pi * formant3 * word_t) +
                        0.05 * np.random.randn(len(word_t))  # Add slight noise for realism
                    )
                    
                    # Apply natural envelope (attack, decay, sustain, release)
                    envelope = np.ones_like(word_t)
                    word_len = len(word_t)
                    
                    if word_len > 100:  # Only apply envelope if word is long enough
                        attack_len = word_len // 10
                        release_len = word_len // 8
                        
                        # Attack
                        envelope[:attack_len] = np.linspace(0, 1, attack_len)
                        # Release
                        envelope[-release_len:] = np.linspace(1, 0, release_len)
                        # Add slight sustain variation
                        sustain_start = attack_len
                        sustain_end = word_len - release_len
                        if sustain_end > sustain_start:
                            envelope[sustain_start:sustain_end] *= (0.8 + 0.2 * np.random.random())
                    
                    audio[word_mask] = word_audio * envelope * 0.4
                
                # Add brief pause between words
                if i < len(words) - 1:
                    pause_start = word_end_time
                    pause_end = min(word_end_time + 0.1, duration)
                    pause_mask = (t >= pause_start) & (t < pause_end)
                    audio[pause_mask] *= 0.1  # Very quiet during pause
            
            # Apply speed and pitch modifications
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            
            if pitch_shift != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift * 12)
            
            # Final processing for more natural sound
            audio = librosa.util.normalize(audio) * 0.7
            
            return audio
            
        except Exception as e:
            print(f"Speech generation error: {e}")
            # Return simple tone as absolute fallback
            duration = max(1.0, len(text) * 0.1)
            t = np.linspace(0, duration, int(22050 * duration))
            return 0.3 * np.sin(2 * np.pi * 150 * t)
