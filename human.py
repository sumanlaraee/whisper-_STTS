import librosa
import numpy as np
import os
from pathlib import Path
import pandas as pd
from scipy import signal

# Install: pip install librosa numpy scipy pandas pyannote.audio torch torchaudio

def extract_acoustic_features(audio_segment):
    """
    Extract acoustic features from an audio segment
    """
    try:
        if len(audio_segment) < 1000:  # Too short
            return None
        
        features = {}
        
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=16000, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Spectral Centroid
        spec_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=16000)[0]
        features['spectral_centroid_mean'] = np.mean(spec_centroid)
        features['spectral_centroid_std'] = np.std(spec_centroid)
        
        # 3. Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=16000)[0]
        features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
        features['spectral_rolloff_std'] = np.std(spec_rolloff)
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_segment, sr=16000)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=audio_segment)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_ratio'] = np.max(rms) / (np.min(rms) + 1e-10)
        
        # 7. Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=16000)
        features['spec_contrast_mean'] = np.mean(spec_contrast, axis=1)
        
        return features
    
    except Exception as e:
        return None


def classify_speaker(features):
    """
    Classify a speaker as HUMAN or MACHINE based on acoustic features
    Returns: 'HUMAN', 'MACHINE', or 'UNCLEAR'
    """
    if features is None:
        return 'UNCLEAR', 0.5
    
    rms_ratio = features['rms_ratio']
    mfcc_std = np.mean(features['mfcc_std'])
    zcr_std = features['zcr_std']
    spec_centroid_std = features['spectral_centroid_std']
    rms_std = features['rms_std']
    
    synthetic_score = 0.0
    
    # RMS ratio check
    if rms_ratio < 2.5:
        synthetic_score += 0.35
    elif rms_ratio > 8.0:
        synthetic_score -= 0.15
    
    # MFCC variability
    if mfcc_std < 1.8:
        synthetic_score += 0.35
    elif mfcc_std > 3.0:
        synthetic_score -= 0.15
    
    # ZCR patterns
    if zcr_std > 0.10:
        synthetic_score += 0.15
    elif zcr_std < 0.03:
        synthetic_score += 0.15
    
    # Spectral stability
    if spec_centroid_std < 400:
        synthetic_score += 0.15
    
    # RMS variability
    if rms_std < 0.01:
        synthetic_score += 0.10
    
    synthetic_score = np.clip(synthetic_score, 0.0, 1.0)
    
    if synthetic_score > 0.55:
        return 'MACHINE', synthetic_score
    elif synthetic_score < 0.35:
        return 'HUMAN', 1.0 - synthetic_score
    else:
        return 'UNCLEAR', synthetic_score


def segment_audio_by_silence(audio_path, silence_threshold=0.02, min_segment_length=0.5):
    """
    Segment audio by detecting silence/pauses to identify different speakers
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Compute RMS energy frame by frame
        frame_length = 2048
        hop_length = 512
        
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        energy = np.sqrt(np.mean(S**2, axis=0))
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        
        # Find silence frames
        silent_frames = np.where(energy < silence_threshold)[0]
        
        # Find continuous silent regions
        segments = []
        current_segment_start = 0
        in_silence = False
        
        for i in range(len(energy)):
            is_silent = i in silent_frames
            
            if is_silent and not in_silence:
                # Start of silence
                segment_duration = librosa.frames_to_time(i, sr=sr, hop_length=hop_length) - \
                                   librosa.frames_to_time(current_segment_start, sr=sr, hop_length=hop_length)
                if segment_duration > min_segment_length:
                    end_frame = i
                    segments.append((
                        current_segment_start,
                        end_frame,
                        librosa.frames_to_time(current_segment_start, sr=sr, hop_length=hop_length),
                        librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
                    ))
                current_segment_start = i
                in_silence = True
            
            elif not is_silent and in_silence:
                # End of silence
                current_segment_start = i
                in_silence = False
        
        # Add final segment
        segment_duration = librosa.frames_to_time(len(energy), sr=sr, hop_length=hop_length) - \
                           librosa.frames_to_time(current_segment_start, sr=sr, hop_length=hop_length)
        if segment_duration > min_segment_length:
            segments.append((
                current_segment_start,
                len(energy),
                librosa.frames_to_time(current_segment_start, sr=sr, hop_length=hop_length),
                librosa.frames_to_time(len(energy), sr=sr, hop_length=hop_length)
            ))
        
        # If no segments found, return the whole audio as one segment
        if not segments:
            segments = [(0, len(energy), 0, librosa.get_duration(y=y, sr=sr))]
        
        return y, sr, segments, hop_length
    
    except Exception as e:
        print(f"   Error segmenting audio: {str(e)}")
        return None, None, None, None


def analyze_audio_speakers(audio_path):
    """
    Analyze an audio file and identify HUMAN vs MACHINE speakers
    """
    filename = os.path.basename(audio_path)
    
    y, sr, segments, hop_length = segment_audio_by_silence(audio_path)
    
    if y is None:
        return {
            "filename": filename,
            "status": "error",
            "speakers": []
        }
    
    speakers = []
    
    for idx, (start_frame, end_frame, start_time, end_time) in enumerate(segments):
        segment_samples = y[int(start_frame * 512):int(end_frame * 512)]
        
        if len(segment_samples) < 8000:  # Too short to classify
            continue
        
        features = extract_acoustic_features(segment_samples)
        speaker_type, confidence = classify_speaker(features)
        
        duration = end_time - start_time
        
        speakers.append({
            "speaker_num": idx + 1,
            "type": speaker_type,
            "confidence": confidence,
            "start_time": f"{start_time:.2f}s",
            "end_time": f"{end_time:.2f}s",
            "duration": f"{duration:.2f}s"
        })
    
    return {
        "filename": filename,
        "status": "success",
        "speakers": speakers
    }


def process_folder(folder_path):
    """
    Process all audio files in a folder
    """
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm']
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}\n")
        return []
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(folder_path.glob(f'*{ext}'))
        audio_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    audio_files = sorted(list(set(audio_files)))
    
    if not audio_files:
        print(f"❌ No audio files found in {folder_path}\n")
        return []
    
    print("\n" + "=" * 100)
    print(f"🎵 SPEAKER DETECTION ANALYSIS (HUMAN vs MACHINE)")
    print(f"📁 Folder: {folder_path}")
    print(f"📊 Found {len(audio_files)} audio file(s)")
    print("=" * 100)
    print()
    
    results = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        print(f"[{idx}/{len(audio_files)}] {filename}")
        
        result = analyze_audio_speakers(str(audio_file))
        results.append(result)
        
        if result['status'] == 'error':
            print("   ❌ Error processing file\n")
        else:
            for speaker in result['speakers']:
                speaker_type = speaker['type']
                confidence = f"{speaker['confidence']:.0%}"
                time_range = f"{speaker['start_time']} - {speaker['end_time']}"
                print(f"   Speaker {speaker['speaker_num']}: [{speaker_type}] (Confidence: {confidence}) [{time_range}]")
            print()
    
    # Summary table
    print("\n" + "=" * 100)
    print("📋 SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Filename':<45} {'Speaker 1':<15} {'Speaker 2':<15} {'Speaker 3':<15}")
    print("-" * 100)
    
    for result in results:
        if result['status'] == 'success':
            filename = result['filename'][:42] + "..." if len(result['filename']) > 45 else result['filename']
            speakers_str = []
            for speaker in result['speakers']:
                speakers_str.append(f"{speaker['type']}")
            
            # Pad to 3 speakers
            while len(speakers_str) < 3:
                speakers_str.append("-")
            
            print(f"{filename:<45} {speakers_str[0]:<15} {speakers_str[1]:<15} {speakers_str[2]:<15}")
    
    print("=" * 100)


# Main execution
if __name__ == "__main__":
    folder_path = "data/recordings_all2"
    process_folder(folder_path)
