import streamlit as st
import os
import tempfile
import time
from voice_cloner import VoiceCloner
from audio_utils import AudioProcessor
import soundfile as sf
import numpy as np

# Initialize session state
if 'voice_cloner' not in st.session_state:
    st.session_state.voice_cloner = VoiceCloner()
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if 'speaker_embedding' not in st.session_state:
    st.session_state.speaker_embedding = None
if 'reference_audio_name' not in st.session_state:
    st.session_state.reference_audio_name = None

st.set_page_config(
    page_title="Voice Cloning Studio",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice Cloning Studio")
st.markdown("Upload a voice sample and clone it to speak any text!")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    **Step 1:** Upload a clear audio sample
    - Supported formats: WAV, MP3, M4A
    - Any length (long files auto-clipped to 30 seconds)
    - Clear speech with minimal background noise
    - Single speaker recommended
    
    **Step 2:** Wait for voice analysis to complete
    
    **Step 3:** Enter the text you want the cloned voice to speak
    
    **Step 4:** Generate and download your cloned audio
    """)
    
    st.header("🎯 Tips for Best Results")
    st.markdown("""
    - Use high-quality audio recordings
    - Avoid background music or noise
    - Choose samples with emotional variety
    - Ensure clear pronunciation
    """)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎵 Upload Voice Sample")
    
    # File uploader with better error handling
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a'],
        help="Upload a 10-30 second voice sample for cloning",
        accept_multiple_files=False,
        key="audio_upload"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"Uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / 1024:.1f} KB")
        
        # Play original audio
        st.audio(uploaded_file, format='audio/wav')
        
        # Process audio if it's a new file
        if st.session_state.reference_audio_name != uploaded_file.name:
            with st.spinner("🔍 Analyzing voice characteristics..."):
                try:
                    # Process the uploaded file
                    audio_data = st.session_state.audio_processor.process_uploaded_audio(uploaded_file)
                    
                    if audio_data is None:
                        st.error("❌ Failed to process audio file. Please check the file format and try again.")
                    else:
                        # Save uploaded file temporarily for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            sf.write(tmp_file.name, audio_data, 22050)
                            tmp_file_path = tmp_file.name
                        
                        # Extract speaker embedding
                        embedding = st.session_state.voice_cloner.extract_speaker_embedding(tmp_file_path)
                        
                        if embedding is not None:
                            st.session_state.speaker_embedding = embedding
                            st.session_state.reference_audio_name = uploaded_file.name
                            st.success("✅ Voice analysis complete! Ready for text-to-speech.")
                            
                            # Show audio info
                            duration = len(audio_data) / 22050
                            st.info(f"✅ Audio processed successfully: {duration:.1f} seconds")
                            
                            if duration >= 30:
                                st.warning("⚠️ Long audio was clipped to 30 seconds for optimal voice analysis")
                        else:
                            st.error("❌ Failed to analyze voice. Please try a different audio file.")
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass  # Ignore cleanup errors
                    
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                    st.error("Please ensure your audio file is in a supported format (WAV, MP3, M4A) and try again.")
        else:
            st.success("✅ Voice already analyzed and ready!")

with col2:
    st.header("📝 Generate Cloned Speech")
    
    if st.session_state.speaker_embedding is not None:
        # Text input
        text_input = st.text_area(
            "Enter text to synthesize",
            placeholder="Type the text you want the cloned voice to speak...",
            height=100,
            help="Enter any text and it will be spoken in the cloned voice"
        )
        
        # Voice settings
        st.subheader("🎛️ Voice Settings")
        col_speed, col_pitch = st.columns(2)
        
        with col_speed:
            speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
        
        with col_pitch:
            pitch_shift = st.slider("Pitch Adjustment", -1.0, 1.0, 0.0, 0.1)
        
        # Generate button
        if st.button("🎤 Generate Cloned Speech", type="primary", disabled=not text_input.strip()):
            if text_input.strip():
                with st.spinner("🔊 Synthesizing speech..."):
                    try:
                        # Generate audio
                        generated_audio = st.session_state.voice_cloner.synthesize_speech(
                            text=text_input.strip(),
                            speaker_embedding=st.session_state.speaker_embedding,
                            speed=speed,
                            pitch_shift=pitch_shift
                        )
                        
                        if generated_audio is not None:
                            # Save generated audio
                            output_path = "generated_speech.wav"
                            sf.write(output_path, generated_audio, 22050)
                            
                            st.success("🎉 Speech generated successfully!")
                            
                            # Play generated audio
                            st.audio(output_path, format='audio/wav')
                            
                            # Download button
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Generated Speech",
                                    data=file.read(),
                                    file_name=f"cloned_speech_{int(time.time())}.wav",
                                    mime="audio/wav"
                                )
                        else:
                            st.error("❌ Failed to generate speech. Please try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating speech: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text to synthesize.")
    else:
        st.info("👆 Please upload and analyze a voice sample first.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Voice Cloning Studio - Powered by Coqui TTS and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

# Error handling for missing models
if not st.session_state.voice_cloner.models_loaded:
    st.error("""
    ⚠️ **Models not loaded properly**
    
    This may be due to:
    - Missing model files
    - Insufficient system resources
    - Network connectivity issues
    
    Please ensure all required models are available and try refreshing the page.
    """)
