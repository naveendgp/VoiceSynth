import librosa
import numpy as np
import soundfile as sf
import io
from typing import Union, Tuple, Optional

class AudioProcessor:
    """Utility class for audio file processing and validation"""
    
    def __init__(self):
        self.target_sr = 22050
        self.max_duration = 300  # Maximum duration in seconds (5 minutes)
        self.min_duration = 3    # Minimum duration in seconds
        self.clip_duration = 30  # Clip longer files to this duration for processing
    
    def process_uploaded_audio(self, uploaded_file) -> Optional[np.ndarray]:
        """Process uploaded audio file and return normalized audio array"""
        try:
            # Read file content
            file_content = uploaded_file.read()
            
            # Reset file pointer for potential re-reading
            uploaded_file.seek(0)
            
            # Load audio using librosa
            audio, sr = librosa.load(io.BytesIO(file_content), sr=self.target_sr)
            
            # Validate audio
            if not self._validate_audio(audio, sr):
                return None
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio)
            
            return processed_audio
            
        except Exception as e:
            print(f"Error processing uploaded audio: {e}")
            return None
    
    def _validate_audio(self, audio: np.ndarray, sr: int) -> bool:
        """Validate audio file meets requirements"""
        try:
            duration = len(audio) / sr
            
            # Check duration
            if duration < self.min_duration:
                print(f"Audio too short: {duration:.1f}s (minimum: {self.min_duration}s)")
                return False
            
            if duration > self.max_duration:
                print(f"Audio too long: {duration:.1f}s (maximum: {self.max_duration}s)")
                print("Audio will be clipped to first 30 seconds for processing")
                # Don't return False, just warn and clip later
            
            # Check if audio has content (not just silence)
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.001:
                print("Audio appears to be silent or too quiet")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating audio: {e}")
            return False
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for voice cloning"""
        try:
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Remove leading/trailing silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Apply gentle high-pass filter to remove low-frequency noise
            audio = librosa.effects.preemphasis(audio)
            
            # Clip to reasonable length for voice cloning
            max_samples = self.target_sr * self.clip_duration  # Use clip_duration
            if len(audio) > max_samples:
                print(f"Clipping audio from {len(audio)/self.target_sr:.1f}s to {self.clip_duration}s")
                audio = audio[:max_samples]
            
            return audio
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return audio
    
    def get_audio_info(self, audio: np.ndarray, sr: int) -> dict:
        """Get information about audio file"""
        try:
            duration = len(audio) / sr
            rms = np.sqrt(np.mean(audio**2))
            
            # Calculate basic audio features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1,  # Mono after librosa processing
                'rms_level': rms,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate,
                'samples': len(audio)
            }
            
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return {}
    
    def convert_format(self, audio: np.ndarray, target_format: str = 'wav') -> bytes:
        """Convert audio to specified format and return as bytes"""
        try:
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Write audio to buffer
            sf.write(buffer, audio, self.target_sr, format=target_format.upper())
            
            # Get bytes
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            print(f"Error converting audio format: {e}")
            return b""
    
    def apply_effects(
        self, 
        audio: np.ndarray, 
        speed: float = 1.0, 
        pitch_shift: float = 0.0,
        volume: float = 1.0
    ) -> np.ndarray:
        """Apply audio effects to the input audio"""
        try:
            processed = audio.copy()
            
            # Apply speed change
            if speed != 1.0:
                processed = librosa.effects.time_stretch(processed, rate=speed)
            
            # Apply pitch shift
            if pitch_shift != 0.0:
                processed = librosa.effects.pitch_shift(
                    processed, 
                    sr=self.target_sr, 
                    n_steps=pitch_shift
                )
            
            # Apply volume change
            if volume != 1.0:
                processed = processed * volume
            
            # Ensure no clipping
            processed = np.clip(processed, -1.0, 1.0)
            
            return processed
            
        except Exception as e:
            print(f"Error applying audio effects: {e}")
            return audio
    
    def detect_speech_segments(self, audio: np.ndarray, sr: int) -> list:
        """Detect speech segments in audio"""
        try:
            # Use librosa to detect non-silent intervals
            intervals = librosa.effects.split(audio, top_db=20)
            
            # Convert to time-based segments
            segments = []
            for start_frame, end_frame in intervals:
                start_time = start_frame / sr
                end_time = end_frame / sr
                duration = end_time - start_time
                
                # Only include segments longer than 0.5 seconds
                if duration > 0.5:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': duration
                    })
            
            return segments
            
        except Exception as e:
            print(f"Error detecting speech segments: {e}")
            return []
