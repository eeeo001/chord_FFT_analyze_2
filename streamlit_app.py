import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io 

# Define constants
TOLERANCE = 0.015 # 1.5% Tolerance for harmonic check
MIN_FREQ_HZ = 50  # Filter out noise below 50Hz
MAX_FREQ_HZ = 2000 # Max frequency for spectrum visualization

# --- (1) define function: frequency to MIDI note ---
def freq_to_midi(frequency):
    """
    frquency(Hz) to MIDI note number (A4=440Hz, MIDI 69).
    """
    if frequency <= 0:
        return -1
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    # Ensure MIDI note is within the standard range (0-127)
    return int(max(0, min(127, round(midi_note))))

# --- (2) Streamlit web page settings ---
st.set_page_config(layout="wide")
st.title("FFT-based Chord Analyzer")
st.markdown("### Identify Chords from Audio Signals by Analyzing the Fourier Transform.")

# --- (3) file uploader widget ---
uploaded_file = st.file_uploader("Upload the audio file for analysis (WAV recommended).", type=['wav', 'mp3'])

if uploaded_file is not None:
    
    # Run analysis logic only if the file is successfully loaded.
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        
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
        yf_positive = np.abs(yf[:half_n]) # Magnitude (Amplitude Spectrum)
        
        st.subheader("Visualize Frequency Spectrum")
        
        # --- 5. Visualize Specturm ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf_positive, yf_positive)
        ax.set_title('Frequency Spectrum (Raw)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([MIN_FREQ_HZ, MAX_FREQ_HZ]) # Musical Frequency Range
        ax.grid(True)
        st.pyplot(fig)

        # ----------------------------------------------------------------------
        # --- 6. Peak Identification and Harmonic Filtering (Core Logic) ---
        # ----------------------------------------------------------------------
        
        # 6-1. Initial Peak Identification
        magnitude_threshold = np.max(yf_positive) * 0.05
        
        # 'distance' constraint removed (improved)
        peak_indices, properties = find_peaks(yf_positive, 
                                                height=magnitude_threshold, 
                                                prominence=magnitude_threshold*0.2) 
        
        # Filter peaks to the musical range
        valid_indices = [i for i in peak_indices if MIN_FREQ_HZ <= xf_positive[i] <= MAX_FREQ_HZ]
        peak_frequencies = xf_positive[valid_indices]
        peak_magnitudes = yf_positive[valid_indices]

        # 6-2. Harmonic Filtering (Correcting Fundamental vs. Harmonic confusion)
        initial_sorted_peaks = sorted(zip(peak_frequencies, peak_magnitudes), key=lambda x: x[1], reverse=True)
        filtered_fundamentals = []
        
        # Use a wider range for harmonic check to cover notes that may be perceived as harmonics (for 9th, 11th, 13th)
        MAX_HARMONIC_N = 8 
        
        for freq, mag in initial_sorted_peaks:
            is_harmonic = False
            
            # CRITICAL FIX: Inverse Harmonic Check (Checks if the current strong peak is a harmonic of a *pre-identified* lower fundamental)
            for n in range(2, MAX_HARMONIC_N + 1):
                hypothetical_fundamental = freq / n
                
                for existing_freq, _ in filtered_fundamentals:
                    expected_harmonic_freq = existing_freq * n
                    
                    if abs(freq - expected_harmonic_freq) / expected_harmonic_freq < TOLERANCE:
                        is_harmonic = True
                        break
                if is_harmonic:
                    break
            
            # --- Standard Check (Secondary Check) ---
            if not is_harmonic:
                 for fundamental_freq, _ in filtered_fundamentals:
                    for n in range(2, MAX_HARMONIC_N + 1): 
                        expected_harmonic_freq = fundamental_freq * n
                        if abs(freq - expected_harmonic_freq) / expected_harmonic_freq < TOLERANCE:
                            is_harmonic = True
                            break
                    if is_harmonic:
                        break
            
            if not is_harmonic:
                # Add as fundamental if the peak is not identified as a harmonic.
                filtered_fundamentals.append((freq, mag))

        # --- Final Fundamental List Creation ---
        filtered_fundamentals.sort(key=lambda x: x[0])
        fundamental_frequencies = [f for f, m in filtered_fundamentals]
        fundamental_midi_notes = [freq_to_midi(f) for f in fundamental_frequencies if f >= MIN_FREQ_HZ]

        st.subheader("Fundamental Frequencies Analysis Results")
        st.markdown(f"**Identified Fundamental Frequencies (Hz):** `{np.round(fundamental_frequencies, 2)}`")
        
        # ----------------------------------------------------------------------
        # --- 7. Chord Identification (Optimized for up to 7 Notes) ---
        # ----------------------------------------------------------------------
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # OPTIMIZATION: Max 7 notes (13th chords). Priority: Complexity > Major > Minor > Dominant.
        chord_templates = {
            # 7음 화음 (7 Notes) - 13th: [1, 3, 5, 7, 9, 11, 13]
            'Major 13th': [0, 4, 7, 11, 2, 5, 9],  
            'Minor 13th': [0, 3, 7, 10, 2, 5, 9],
            'Dominant 13th': [0, 4, 7, 10, 2, 5, 9], 

            # 6음 화음 (6 Notes) - 11th: [1, 3, 5, 7, 9, 11]
            'Major 11th': [0, 4, 7, 11, 2, 5], 
            'Minor 11th': [0, 3, 7, 10, 2, 5],
            'Dominant 11th': [0, 4, 7, 10, 2, 5], 

            # 5음 화음 (5 Notes) - 9th: [1, 3, 5, 7, 9]
            'Major 9th': [0, 4, 7, 11, 2],         
            'Minor 9th': [0, 3, 7, 10, 2],
            'Dominant 9th': [0, 4, 7, 10, 2],

            # 4음 화음 (4 Notes) - 7th: [1, 3, 5, 7]
            'Major 7th': [0, 4, 7, 11],
            'Minor 7th': [0, 3, 7, 10],
            'Dominant 7th': [0, 4, 7, 10],

            # 3음 화음 (3 Notes) - Triads: [1, 3, 5]
            'Major': [0, 4, 7], 
            'Minor': [0, 3, 7]
        }
        
        best_match_score = -1
        best_root_midi = -1
        best_chord_type = ""
        identified_chord = "No chord identified."
        
        # Use only unique note classes (C, C#, etc.) regardless of octave
        unique_fundamental_midi_notes = sorted(list(set(note % 12 for note in fundamental_midi_notes)))
        
        # Match the identified notes to chord templates based on music theory.
        for root_midi_interval in unique_fundamental_midi_notes:
            observed_intervals = set(unique_fundamental_midi_notes)

            for chord_type, template_intervals in chord_templates.items():
                
                # Calculate the notes expected in the chord template based on the current root
                expected_notes = set((root_midi_interval + interval) % 12 for interval in template_intervals)
                
                # The score is the number of expected notes found in the observed intervals
                match_score = sum(1 for note in expected_notes if note in observed_intervals)

                # Prioritize based on match score and the earlier templates (Major, more complex) if scores are equal.
                if match_score >= 2 and match_score > best_match_score: 
                    best_match_score = match_score
                    best_root_midi = root_midi_interval
                    best_chord_type = chord_type

        # Final Results
        if best_root_midi != -1 and best_match_score >= 2:
            root_name = note_names[best_root_midi]
            identified_chord = f"**{root_name} {best_chord_type}**"
        
        st.markdown(f"### Final Identified Chord: {identified_chord}")
        st.info(f"Match Score (maximum 7 for 13th chords): {best_match_score}")

    except Exception as e:
        # When the audio file is corrupted or incorrectly formatted
        st.error(f"Error: Failed to analyze audio file.: {e}")
        st.info("Please check if the file is a supported format (WAV or MP3) and retry.")

else:
    # File Upload Pending
    st.info("Upload an audio file and check the analysis results (WAV file recommended).")
