import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io # Add library for Streamlit file processing.

# --- (External Component for Recording) ---
# NOTE: This line requires 'pip install streamlit-mic-recorder' 
# to run in a real environment.
try:
    from st_custom_components import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    st.warning("Install 'streamlit-mic-recorder' to enable recording feature.")
    MIC_RECORDER_AVAILABLE = False
# ------------------------------------------


# --- (1) define function: frequency to MIDI note ---
def freq_to_midi(frequency):
    """
    frquency(Hz) to MIDI note number (A4=440Hz, MIDI 69)
    """
    if frequency <= 0:
        return -1
    # MIDI note = 69 + 12 * log2(frequency / 440.0)
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    return int(round(midi_note))

# --- (2) Streamlit web page settings ---
st.set_page_config(layout="wide")
st.title("FFT-based Automatic Chord Recognition")
st.markdown("### Identify Chords from Audio Signals by Analyzing the Fourier Transform.")

# --- File Selection Method ---
st.subheader("Select Audio Source")
col_upload, col_record = st.columns(2)

with col_upload:
    st.markdown("#### 1. Upload File")
    # --- (3) file uploader widget ---
    uploaded_file = st.file_uploader("Upload the audio file for analysis (WAV recommended).", type=['wav', 'mp3'])
    
uploaded_file_bytes = None
if uploaded_file is not None:
    uploaded_file_bytes = uploaded_file.read()

with col_record:
    st.markdown("#### 2. Record Live")
    
    # --- (3-Additional) Live Recording Widget ---
    if MIC_RECORDER_AVAILABLE:
        # The mic_recorder component saves the recording data to the Streamlit session state.
        # It's an external component and might require installation.
        # The returned object has an 'audio_bytes' key with the raw recording data.
        audio_data = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            key='recorder',
            format="wav" # Ensure format is WAV for broad compatibility
        )
        
        recorded_audio_bytes = None
        if audio_data:
            recorded_audio_bytes = audio_data.get('audio_bytes')
            if recorded_audio_bytes:
                st.success("Recording complete. Analysis will proceed with the recorded audio.")
    else:
        st.error("Live Recording component is not available. Please install 'streamlit-mic-recorder'.")

# --- Decide which file to process ---
file_to_process = None

if uploaded_file_bytes is not None:
    # Prioritize uploaded file if both are present (or only upload is present)
    file_to_process = io.BytesIO(uploaded_file_bytes)
    st.markdown("---")
    st.info("**Processing the Uploaded File.**")
elif 'recorded_audio_bytes' in locals() and recorded_audio_bytes is not None:
    # Use recorded audio if no upload, but a recording exists
    file_to_process = io.BytesIO(recorded_audio_bytes)
    st.markdown("---")
    st.info("**Processing the Recorded Audio.**")
else:
    # File Upload Pending
    st.markdown("---")
    st.info("Upload an audio file or record live to start the analysis.")
    # Stop execution if no file is ready
    st.stop()
    
if file_to_process is not None:
    # ----------------------------------------------------
    # Run analysis logic only if the file is successfully loaded (either by upload or record).
    # ----------------------------------------------------
    
    try:
        # Load the file into memory
        # y: Audio data, sr: Sampling rate
        # Use io.BytesIO object for librosa.load from memory
        y, sr = librosa.load(file_to_process, sr=None)
        
        # --- Display File Information ---
        st.success("File successfully loaded!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sampling Rate (sr)", f"{sr} Hz")
        with col2:
            st.metric("Duration", f"{len(y)/sr:.2f} seconds")
        
        # --- 4. Perform FFT and Calculate Spectrum ---
        N = len(y)
        yf = fft(y)
        xf = fftfreq(N, 1/sr)
        
        half_n = N // 2
        xf_positive = xf[:half_n] # Positive Frequencies
        yf_positive = np.abs(yf[:half_n]) # Magnitude
        
        st.subheader("Visualize Frequency Spectrum")
        
        # --- 5. Visualize Specturm ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf_positive, yf_positive)
        ax.set_title('Frequency Spectrum (Raw)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([20, 2000]) # 20Hz ~ 2000Hz (Musical Frequency Range)
        ax.grid(True)
        st.pyplot(fig)
        

        # --- 6. Peak Identification and Harmonic Filtering ---
        
        # 6-1. Initial Peak Identification
        magnitude_threshold = np.max(yf_positive) * 0.05
        frequency_resolution = sr / N
        min_freq_separation_hz = 10 # Identify only peaks separated by 10Hz or more initially.
        distance_bins = int(min_freq_separation_hz / frequency_resolution)
        
        peak_indices, _ = find_peaks(yf_positive, height=magnitude_threshold, distance=distance_bins)
        peak_frequencies = xf_positive[peak_indices]
        peak_magnitudes = yf_positive[peak_indices]

        # 6-2. Harmonic Filtering
        initial_sorted_peaks = sorted(zip(peak_magnitudes, peak_frequencies), key=lambda x: x[0], reverse=True)
        filtered_fundamentals = []
        tolerance = 0.015 # 1.5% Tolerance
        
        for mag, freq in initial_sorted_peaks:
            is_harmonic = False
            for fundamental_freq, fundamental_mag in filtered_fundamentals:
                for n in range(2, 6): # Check 2nd–5th Harmonics
                    expected_harmonic_freq = fundamental_freq * n
                    if abs(freq - expected_harmonic_freq) / expected_harmonic_freq < tolerance:
                        is_harmonic = True
                        break
                if is_harmonic:
                    break
            if not is_harmonic:
                # Add as fundamental if the peak is not identified as a harmonic.
                filtered_fundamentals.append((freq, mag))

        filtered_fundamentals.sort(key=lambda x: x[0])
        fundamental_frequencies = [f for f, m in filtered_fundamentals]
        fundamental_midi_notes = [freq_to_midi(f) for f in fundamental_frequencies if f > 50] # 50Hz 미만 노이즈 제거

        st.subheader("Fundamental Frequencies Analysis Results")
        st.markdown(f"**Identified Fundamental Frequencies (Hz):** `{np.round(fundamental_frequencies, 2)}`")
        
        # --- 7. Chord Identification ---
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_templates = {
            'Major': [0, 4, 7], 'Minor': [0, 3, 7], 'Dominant 7th': [0, 4, 7, 10],
            'Major 7th': [0, 4, 7, 11], 'Minor 7th': [0, 3, 7, 10]
        }
        
        best_match_score = -1
        best_root_midi = -1
        best_chord_type = ""
        identified_chord = "No chord identified."
        
        unique_fundamental_midi_notes = sorted(list(set(fundamental_midi_notes)))
        
        # Match the identified notes to chord templates based on music theory.
        for root_midi in unique_fundamental_midi_notes:
            observed_intervals = set((note - root_midi) % 12 for note in fundamental_midi_notes)

            for chord_type, template_intervals in chord_templates.items():
                match_score = sum(1 for interval in template_intervals if interval in observed_intervals)

                if match_score >= 2 and match_score > best_match_score: # Consider a chord a candidate only if at least two notes match.
                    best_match_score = match_score
                    best_root_midi = root_midi
                    best_chord_type = chord_type

        # Final Results
        if best_root_midi != -1 and best_match_score >= 2:
            root_name = note_names[best_root_midi % 12]
            identified_chord = f"**{root_name} {best_chord_type}**"
        
        st.markdown(f"### Final Identified Chord: {identified_chord}")
        st.info(f"Match Score (maximum {len(unique_fundamental_midi_notes)}): {best_match_score}")

    except Exception as e:
        # When the audio file is corrupted or incorrectly formatted
        st.error(f"Error: Failed to analyze audio file.: {e}")
        st.info("Please check if the file is a supported format (WAV or MP3) and retry.")
