# # import os
# # import sys
# # import argparse
# # import subprocess
# # import tempfile
# # from pyannote.audio import Pipeline

# # def convert_to_wav(input_path: str) -> str:
# #     """
# #     Convert any audio file to mono 16 kHz WAV using ffmpeg.
# #     Returns the path to the converted WAV file.
# #     """
# #     # create a temporary WAV file
# #     tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
# #     tmp_wav_path = tmp_wav.name
# #     tmp_wav.close()

# #     cmd = [
# #         "ffmpeg",
# #         "-y",                   # overwrite if exists
# #         "-i", input_path,       # input file
# #         "-ac", "1",             # mono
# #         "-ar", "16000",         # 16 kHz
# #         "-vn",                  # no video
# #         tmp_wav_path
# #     ]

# #     try:
# #         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# #     except subprocess.CalledProcessError:
# #         os.unlink(tmp_wav_path)
# #         raise RuntimeError(f"ffmpeg failed to convert {input_path} to WAV")

# #     return tmp_wav_path

# # def diarize_audio(file_path: str, hf_token: str):
# #     """
# #     Perform speaker diarization on `file_path` (any format) and print:
# #       - Number of unique speakers
# #       - Time-stamped speaker turns
# #     """
# #     # 1. Ensure WAV input
# #     ext = os.path.splitext(file_path)[1].lower()
# #     if ext != ".wav":
# #         wav_path = convert_to_wav(file_path)
# #     else:
# #         wav_path = file_path

# #     # 2. Load pretrained diarization pipeline
# #     pipeline = Pipeline.from_pretrained(
# #         "pyannote/speaker-diarization-3.1",
# #         use_auth_token=hf_token
# #     )

# #     # 3. Apply diarization
# #     diarization = pipeline(wav_path)

# #     # 4. Extract segments and speaker labels
# #     segments = []
# #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# #         segments.append({
# #             "speaker": speaker,
# #             "start": round(turn.start, 2),
# #             "end":   round(turn.end,   2)
# #         })

# #     # 5. Count distinct speakers
# #     speakers = sorted({seg["speaker"] for seg in segments})

# #     # 6. Output results
# #     print(f"\nDetected {len(speakers)} speakers: {', '.join(speakers)}\n")
# #     print("Speaker turns:")
# #     for seg in segments:
# #         print(f"  {seg['speaker']}: {seg['start']}s – {seg['end']}s")

# #     # 7. Clean up temporary file if created
# #     if ext != ".wav":
# #         os.unlink(wav_path)

# #     return segments

# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Unsupervised speaker diarization using pyannote.audio"
# #     )
# #     parser.add_argument(
# #         "audio_file",
# #         help="Path to the input audio file (e.g., your_clip.m4a or .wav)"
# #     )
# #     parser.add_argument(
# #         "--hf_token",
# #         help="Hugging Face token (default: read from HF_TOKEN env var)",
# #         default=os.getenv("HF_TOKEN")
# #     )
# #     args = parser.parse_args()

# #     # Validate audio file
# #     if not os.path.isfile(args.audio_file):
# #         print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
# #         sys.exit(1)

# #     # Validate token
# #     if not args.hf_token:
# #         print(
# #             "Error: Hugging Face token not provided.\n"
# #             "  • Set the HF_TOKEN environment variable, or\n"
# #             "  • pass --hf_token your_token_here",
# #             file=sys.stderr
# #         )
# #         sys.exit(1)

# #     try:
# #         diarize_audio(args.audio_file, args.hf_token)
# #     except Exception as e:
# #         print(f"Error during diarization: {e}", file=sys.stderr)
# #         sys.exit(1)

# # if __name__ == "__main__":
# #     main()


# # '''
# #     to run script 
# #     commands for cmd : 
# #     set HF_TOKEN=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
# #     echo %HF_TOKEN%
# #     python diarize_audio.py input\audio1290408906.m4a


# # '''












# #!/usr/bin/env python3
# import os
# import sys
# import time
# import argparse
# import tempfile
# import subprocess

# from pydub import AudioSegment
# from tqdm import tqdm
# from pyannote.audio import Pipeline
# import openai

# def convert_to_wav(input_path: str) -> str:
#     tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     wav_path = tmp.name
#     tmp.close()

#     subprocess.run([
#         "ffmpeg", "-y", "-i", input_path,
#         "-ac", "1", "-ar", "16000", "-vn", wav_path
#     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     return wav_path

# def transcribe_with_openai(segment_path: str) -> str:
#     """
#     Uses OpenAI's Whisper API (model=whisper-1) to transcribe one segment.
#     """
#     with open(segment_path, "rb") as f:
#         resp = openai.Audio.transcribe(
#             model="whisper-1",
#             file=f
#         )
#     return resp["text"].strip()

# def diarize_and_transcribe(
#     input_file: str,
#     hf_token: str,
#     openai_key: str
# ):
#     start_total = time.time()

#     # Phase 1: ensure WAV
#     ext = os.path.splitext(input_file)[1].lower()
#     wav_path = convert_to_wav(input_file) if ext != ".wav" else input_file

#     # Phase 2: load & run diarization
#     print("[1/4] Loading diarization pipeline…")
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=hf_token
#     )
#     t0 = time.time()
#     print("[2/4] Running diarization (this may take a bit)...")
#     diarization = pipeline(wav_path)
#     print(f"    ↳ Diarization done in {time.time() - t0:.1f}s")

#     # Phase 3: prepare OpenAI client
#     openai.api_key = openai_key
#     print("[3/4] Preparing transcription via OpenAI API…")

#     # Phase 4: load audio into memory
#     audio = AudioSegment.from_wav(wav_path)
#     turns = list(diarization.itertracks(yield_label=True))

#     # Loop through turns with a progress bar showing ETA
#     print("[4/4] Transcribing segments (this will show ETA)…")
#     for turn, _, speaker in tqdm(turns, desc="Segments", unit="seg"):
#         s_ms = int(turn.start * 1000)
#         e_ms = int(turn.end   * 1000)

#         # export one-time temp file for this segment
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             seg_path = tmp.name
#         audio[s_ms:e_ms].export(seg_path, format="wav")

#         # transcribe via API
#         text = transcribe_with_openai(seg_path)
#         os.unlink(seg_path)

#         # print results in original format + text
#         print(f"{speaker} | {turn.start:.2f}s → {turn.end:.2f}s")
#         print(f"    “{text}”\n")

#     # cleanup WAV if needed
#     if ext != ".wav":
#         os.unlink(wav_path)

#     print(f"[✔] Total elapsed: {(time.time() - start_total)/60:.1f} min")

# def main():
#     parser = argparse.ArgumentParser("Diarize + Transcribe with OpenAI")
#     parser.add_argument("audio_file", help="Path to .m4a/.wav/etc")
#     parser.add_argument(
#         "--hf_token",
#         default=os.getenv("HF_TOKEN"),
#         help="Hugging Face token (env var HF_TOKEN)"
#     )
#     parser.add_argument(
#         "--openai_key",
#         default=os.getenv("OPENAI_API_KEY"),
#         help="OpenAI API key (env var OPENAI_API_KEY)"
#     )
#     args = parser.parse_args()

#     if not os.path.isfile(args.audio_file):
#         print(f"[ERROR] File not found: {args.audio_file}", file=sys.stderr)
#         sys.exit(1)
#     if not args.hf_token:
#         print("[ERROR] HF_TOKEN not set", file=sys.stderr); sys.exit(1)
#     if not args.openai_key:
#         print("[ERROR] OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)

#     try:
#         diarize_and_transcribe(
#             args.audio_file,
#             args.hf_token,
#             args.openai_key
#         )
#     except Exception as e:
#         print(f"[ERROR] {e}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     main()














# import os
# import sys
# import argparse
# import subprocess
# import tempfile
# from pyannote.audio import Pipeline

# def convert_to_wav(input_path: str) -> str:
#     """
#     Convert any audio file to mono 16 kHz WAV using ffmpeg.
#     Returns the path to the converted WAV file.
#     """
#     # create a temporary WAV file
#     tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     tmp_wav_path = tmp_wav.name
#     tmp_wav.close()

#     cmd = [
#         "ffmpeg",
#         "-y",                   # overwrite if exists
#         "-i", input_path,       # input file
#         "-ac", "1",             # mono
#         "-ar", "16000",         # 16 kHz
#         "-vn",                  # no video
#         tmp_wav_path
#     ]

#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     except subprocess.CalledProcessError:
#         os.unlink(tmp_wav_path)
#         raise RuntimeError(f"ffmpeg failed to convert {input_path} to WAV")

#     return tmp_wav_path

# def diarize_audio(file_path: str, hf_token: str):
#     """
#     Perform speaker diarization on `file_path` (any format) and print:
#       - Number of unique speakers
#       - Time-stamped speaker turns
#     """
#     # 1. Ensure WAV input
#     ext = os.path.splitext(file_path)[1].lower()
#     if ext != ".wav":
#         wav_path = convert_to_wav(file_path)
#     else:
#         wav_path = file_path

#     # 2. Load pretrained diarization pipeline
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=hf_token
#     )

#     # 3. Apply diarization
#     diarization = pipeline(wav_path)

#     # 4. Extract segments and speaker labels
#     segments = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         segments.append({
#             "speaker": speaker,
#             "start": round(turn.start, 2),
#             "end":   round(turn.end,   2)
#         })

#     # 5. Count distinct speakers
#     speakers = sorted({seg["speaker"] for seg in segments})

#     # 6. Output results
#     print(f"\nDetected {len(speakers)} speakers: {', '.join(speakers)}\n")
#     print("Speaker turns:")
#     for seg in segments:
#         print(f"  {seg['speaker']}: {seg['start']}s – {seg['end']}s")

#     # 7. Clean up temporary file if created
#     if ext != ".wav":
#         os.unlink(wav_path)

#     return segments

# def main():
#     parser = argparse.ArgumentParser(
#         description="Unsupervised speaker diarization using pyannote.audio"
#     )
#     parser.add_argument(
#         "audio_file",
#         help="Path to the input audio file (e.g., your_clip.m4a or .wav)"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: read from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     args = parser.parse_args()

#     # Validate audio file
#     if not os.path.isfile(args.audio_file):
#         print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
#         sys.exit(1)

#     # Validate token
#     if not args.hf_token:
#         print(
#             "Error: Hugging Face token not provided.\n"
#             "  • Set the HF_TOKEN environment variable, or\n"
#             "  • pass --hf_token your_token_here",
#             file=sys.stderr
#         )
#         sys.exit(1)

#     try:
#         diarize_audio(args.audio_file, args.hf_token)
#     except Exception as e:
#         print(f"Error during diarization: {e}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     main()







# import os
# import sys
# import argparse
# import subprocess
# import tempfile
# import warnings

# import torch
# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32   = True

# # Suppress Pyannote reproducibility and pooling warnings
# warnings.filterwarnings(
#     "ignore",
#     message=r"TensorFloat-32.*",
#     module="pyannote.audio.utils.reproducibility"
# )
# warnings.filterwarnings(
#     "ignore",
#     message=r"std\(\): degrees of freedom is <= 0.*",
#     module="pyannote.audio.models.blocks.pooling"
# )

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm

# # Optional: install and import faster-whisper for even faster ASR
# # from faster_whisper import WhisperModel

# def convert_to_wav(input_path: str) -> str:
#     """
#     Convert any audio file to mono 16 kHz WAV using ffmpeg.
#     Returns the path to the converted WAV file.
#     """
#     print("🔄 Converting to WAV…", end=" ", flush=True)
#     tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     tmp_wav_path = tmp_wav.name
#     tmp_wav.close()

#     cmd = [
#         "ffmpeg", "-y",
#         "-i", input_path,
#         "-ac", "1",
#         "-ar", "16000",
#         "-vn",
#         tmp_wav_path
#     ]

#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     except subprocess.CalledProcessError:
#         os.unlink(tmp_wav_path)
#         raise RuntimeError(f"ffmpeg failed to convert {input_path} to WAV")

#     print("done")
#     return tmp_wav_path


# def diarize_and_transcribe(file_path: str, hf_token: str):
#     """
#     Perform speaker diarization on `file_path` and transcribe each segment using GPU acceleration.
#     """
#     # 1. Ensure WAV input
#     ext = os.path.splitext(file_path)[1].lower()
#     wav_path = convert_to_wav(file_path) if ext != ".wav" else file_path

#     # 2. Load pretrained diarization pipeline on GPU
#     print("🧠 Loading diarization model (GPU)…", flush=True)
#     diarize_pipe = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=hf_token
#     ).to(torch.device("cuda"))
#     print("done")

#     # 3. Apply diarization (GPU)
#     print("🎙️ Running diarization…", flush=True)
#     diarization = diarize_pipe(wav_path)
#     print("done")

#     # 4. Extract segments
#     segments = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         segments.append({
#             "speaker": speaker,
#             "start": round(turn.start, 2),
#             "end": round(turn.end, 2)
#         })

#     speakers = sorted({seg['speaker'] for seg in segments})
#     print(f"\nDetected {len(speakers)} speakers: {', '.join(speakers)}\n")

#     # 5. Initialize ASR on GPU (Whisper Tiny for speed)
#     print("📝 Loading ASR model (GPU)…", flush=True)
#     asr = hf_pipeline(
#         "automatic-speech-recognition",
#         model="openai/whisper-tiny.en",
#         device=0
#     )
#     print("done")

#     # Optional: faster-whisper ASR
#     # asr_model = WhisperModel("tiny", device="cuda", compute_type="int8")

#     # 6. Transcribe each segment in sequence with progress
#     print("🔊 Transcribing segments…")
#     for seg in tqdm(segments, desc="Segments", unit="seg", colour="green"):
#         start, end = seg["start"], seg["end"]
#         tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_seg_path = tmp_seg.name
#         tmp_seg.close()

#         subprocess.run([
#             "ffmpeg", "-y",
#             "-i", wav_path,
#             "-ss", str(start),
#             "-to", str(end),
#             "-ac", "1",
#             "-ar", "16000",
#             "-vn",
#             tmp_seg_path
#         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         result = asr(tmp_seg_path)
#         transcript = result.get("text", "").strip()

#         print(f"{seg['speaker']}: {start}s – {end}s\n  Transcript: {transcript}\n")
#         os.unlink(tmp_seg_path)

#     # 7. Cleanup temp WAV
#     if ext != ".wav":
#         os.unlink(wav_path)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Speaker diarization + GPU-accelerated transcription"
#     )
#     parser.add_argument(
#         "audio_file",
#         help="Path to input audio/video (e.g., .mp4, .m4a, .wav)"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     args = parser.parse_args()

#     if not os.path.isfile(args.audio_file):
#         print(f"Error: file not found: {args.audio_file}", file=sys.stderr)
#         sys.exit(1)
#     if not args.hf_token:
#         print(
#             "Error: Hugging Face token not provided.\n"
#             "  • Set HF_TOKEN env var, or\n"
#             "  • pass --hf_token YOUR_TOKEN",
#             file=sys.stderr
#         )
#         sys.exit(1)

#     try:
#         diarize_and_transcribe(args.audio_file, args.hf_token)
#     except KeyboardInterrupt:
#         print("\nProcess interrupted by user.", file=sys.stderr)
#         sys.exit(1)
#     except Exception as e:
#         print(f"Error: {e}", file=sys.stderr)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()







# import os
# import sys
# import argparse
# import subprocess
# import tempfile
# import warnings

# # 🔕 Force pyannote to avoid torchcodec entirely
# os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"

# import torch

# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Suppress Pyannote warnings
# warnings.filterwarnings(
#     "ignore",
#     message=r"TensorFloat-32.*",
#     module="pyannote.audio.utils.reproducibility"
# )
# warnings.filterwarnings(
#     "ignore",
#     message=r"std\(\): degrees of freedom is <= 0.*",
#     module="pyannote.audio.models.blocks.pooling"
# )

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm


# def convert_to_wav(input_path: str) -> str:
#     """Convert any audio/video to mono 16kHz WAV using ffmpeg."""
#     print("🔄 Converting to WAV…", end=" ", flush=True)

#     tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     tmp_wav_path = tmp_wav.name
#     tmp_wav.close()

#     cmd = [
#         "ffmpeg", "-y",
#         "-i", input_path,
#         "-ac", "1",
#         "-ar", "16000",
#         "-vn",
#         tmp_wav_path
#     ]

#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     except subprocess.CalledProcessError:
#         os.unlink(tmp_wav_path)
#         raise RuntimeError(f"ffmpeg failed to convert {input_path}")

#     print("done")
#     return tmp_wav_path


# def diarize_and_transcribe(file_path: str, hf_token: str):
#     """Run speaker diarization + ASR on GPU."""

#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available. GPU is required.")

#     # Ensure WAV
#     ext = os.path.splitext(file_path)[1].lower()
#     wav_path = convert_to_wav(file_path) if ext != ".wav" else file_path

#     # Load diarization model
#     print("🧠 Loading diarization model (GPU)…", flush=True)
#     diarize_pipe = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=hf_token
#     ).to("cuda")
#     print("done")

#     # Run diarization
#     print("🎙️ Running diarization…", flush=True)
#     diarization = diarize_pipe(wav_path)
#     print("done")

#     # Collect segments
#     segments = [
#         {
#             "speaker": speaker,
#             "start": round(turn.start, 2),
#             "end": round(turn.end, 2)
#         }
#         for turn, _, speaker in diarization.itertracks(yield_label=True)
#     ]

#     speakers = sorted({s["speaker"] for s in segments})
#     print(f"\nDetected {len(speakers)} speakers: {', '.join(speakers)}\n")

#     # Load ASR
#     print("📝 Loading ASR model (GPU)…", flush=True)
#     asr = hf_pipeline(
#         "automatic-speech-recognition",
#         model="openai/whisper-tiny.en",
#         device=0
#     )
#     print("done")

#     # Transcribe segments
#     print("🔊 Transcribing segments…")
#     for seg in tqdm(segments, desc="Segments", unit="seg"):
#         tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_seg_path = tmp_seg.name
#         tmp_seg.close()

#         subprocess.run([
#             "ffmpeg", "-y",
#             "-i", wav_path,
#             "-ss", str(seg["start"]),
#             "-to", str(seg["end"]),
#             "-ac", "1",
#             "-ar", "16000",
#             "-vn",
#             tmp_seg_path
#         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         result = asr(tmp_seg_path)
#         text = result.get("text", "").strip()

#         print(f'{seg["speaker"]}: {seg["start"]}s – {seg["end"]}s')
#         print(f"  Transcript: {text}\n")

#         os.unlink(tmp_seg_path)

#     if ext != ".wav":
#         os.unlink(wav_path)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Speaker diarization + GPU transcription"
#     )
#     parser.add_argument("audio_file", nargs="?", help="Path to audio/video file")
#     parser.add_argument(
#         "--hf_token",
#         default=os.getenv("HF_TOKEN"),
#         help="Hugging Face token (or set HF_TOKEN env var)"
#     )

#     args = parser.parse_args()

#     # 🧠 Jupyter-friendly fallback
#     if not args.audio_file:
#         args.audio_file = input("Enter path to audio file: ").strip()

#     if not os.path.isfile(args.audio_file):
#         sys.exit(f"Error: file not found: {args.audio_file}")

#     if not args.hf_token:
#         sys.exit(
#             "Error: Hugging Face token missing.\n"
#             "Set HF_TOKEN env var or pass --hf_token"
#         )

#     diarize_and_transcribe(args.audio_file, args.hf_token)


# if __name__ == "__main__":
#     main()















# #!/usr/bin/env python3
# import os
# import sys
# import argparse
# import subprocess
# import tempfile
# import datetime
# import warnings

# import torch
# import soundfile as sf
# from pyannote.audio import Model, Inference
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm

# warnings.filterwarnings("ignore")

# SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg")
# FINAL_OUTPUT_FILE = "final_diarization_output.txt"

# def convert_to_wav(input_path: str) -> str:
#     tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     tmp.close()
#     subprocess.run(
#         [
#             "ffmpeg", "-y",
#             "-i", input_path,
#             "-ac", "1",
#             "-ar", "16000",
#             "-vn",
#             tmp.name
#         ],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#         check=True
#     )
#     return tmp.name

# def get_duration(wav_path: str) -> float:
#     with sf.SoundFile(wav_path) as f:
#         return len(f) / f.samplerate

# def collect_audio_files(folder: str):
#     files = []
#     for root, _, names in os.walk(folder):
#         for n in names:
#             if n.lower().endswith(SUPPORTED_EXTENSIONS):
#                 files.append(os.path.join(root, n))
#     return sorted(files)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("input_folder", help="Folder with audio recordings")
#     parser.add_argument("--whisper_model", default="openai/whisper-tiny.en")
#     args = parser.parse_args()

#     if not torch.cuda.is_available():
#         sys.exit("❌ CUDA GPU required")

#     audio_files = collect_audio_files(args.input_folder)
#     if not audio_files:
#         sys.exit("❌ No audio files found")

#     print("🧠 Loading VAD model (public)...")
#     model = Model.from_pretrained("pyannote/voice-activity-detection", use_auth_token=None)
#     vad = Inference(model, device="cuda")
#     print("done")

#     print("📝 Loading Whisper ASR model...")
#     asr = hf_pipeline(
#         "automatic-speech-recognition",
#         model=args.whisper_model,
#         device=0
#     )
#     print("done")

#     with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as OUT:
#         total_files = len(audio_files)

#         for idx, audio in enumerate(audio_files, 1):
#             print(f"\n▶ Processing [{idx}/{total_files}]: {audio}")

#             wav = convert_to_wav(audio)
#             duration = get_duration(wav)
#             OUT.write(f"\nDURATION: {duration:.2f} seconds\n\n")

#             waveform, sample_rate = sf.read(wav)
#             waveform = waveform.T if waveform.ndim > 1 else waveform.reshape(1, -1)

#             segments = vad({'waveform': waveform, 'sample_rate': sample_rate})

#             speaker_full_text = {}
#             for i, (start, end) in enumerate(segments, 1):
#                 seg_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#                 seg_file.close()

#                 subprocess.run(
#                     [
#                         "ffmpeg", "-y",
#                         "-i", wav,
#                         "-ss", str(start),
#                         "-to", str(end),
#                         "-ac", "1",
#                         "-ar", "16000",
#                         "-vn",
#                         seg_file.name
#                     ],
#                     stdout=subprocess.DEVNULL,
#                     stderr=subprocess.DEVNULL,
#                     check=True
#                 )

#                 text = asr(seg_file.name).get("text", "").strip()
#                 os.unlink(seg_file.name)

#                 speaker_label = f"Speaker 01"  # simple VAD = single speaker
#                 OUT.write(f"[{start:.2f}s -> {end:.2f}s] {speaker_label}: {text}\n")
#                 speaker_full_text.setdefault(speaker_label, []).append(text)

#             OUT.write("\n")
#             for spk, texts in speaker_full_text.items():
#                 OUT.write(f"{spk} (FULL): {' '.join(texts)}\n\n")

#             OUT.write("="*100 + "\n")
#             os.unlink(wav)

#         OUT.write(
#             f"""PROCESSING COMPLETE
# Company: MEDICARE
# Total files processed: {total_files}
# Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# {"="*100}
# """
#         )

#     print(f"\n✅ DONE — Output saved to: {FINAL_OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()








# import os
# import sys
# import json
# import argparse
# import subprocess
# import tempfile
# import warnings
# from pathlib import Path
# from datetime import datetime
# import torch
# from collections import defaultdict

# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Suppress warnings
# warnings.filterwarnings("ignore", message=r"TensorFloat-32.*", module="pyannote.audio.utils.reproducibility")
# warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", module="pyannote.audio.models.blocks.pooling")

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm


# class AudioProcessor:
#     def __init__(self, hf_token: str, output_dir: str = "output_transcripts"):
#         self.hf_token = hf_token
#         self.output_dir = output_dir
#         self.diarize_pipe = None
#         self.asr = None
        
#         # Create output directory
#         Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
#         # Load models once
#         self.load_models()
    
#     def load_models(self):
#         """Load diarization and ASR models on GPU"""
#         print("\n🧠 Loading diarization model (GPU)…", flush=True)
#         self.diarize_pipe = Pipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1"
#         ).to(torch.device("cuda"))
#         print("✅ Diarization model loaded")
        
#         print("📝 Loading ASR model (GPU)…", flush=True)
#         self.asr = hf_pipeline(
#             "automatic-speech-recognition",
#             model="openai/whisper-tiny.en",
#             device=0
#         )
#         print("✅ ASR model loaded\n")
    
#     def convert_to_wav(self, input_path: str) -> str:
#         """Convert audio file to mono 16 kHz WAV"""
#         ext = os.path.splitext(input_path)[1].lower()
#         if ext == ".wav":
#             return input_path
        
#         print(f"   🔄 Converting to WAV…", end=" ", flush=True)
#         tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_wav_path = tmp_wav.name
#         tmp_wav.close()
        
#         cmd = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-ac", "1", "-ar", "16000", "-vn", tmp_wav_path
#         ]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print("done")
#             return tmp_wav_path
#         except subprocess.CalledProcessError as e:
#             os.unlink(tmp_wav_path)
#             raise RuntimeError(f"ffmpeg failed: {e}")
    
#     def process_audio(self, file_path: str) -> dict:
#         """Process single audio file and return transcript data"""
#         file_name = Path(file_path).name
#         print(f"\n📁 Processing: {file_name}")
        
#         try:
#             # Convert to WAV
#             wav_path = self.convert_to_wav(file_path)
            
#             # Diarization
#             print(f"   🎙️  Running diarization…", end=" ", flush=True)
#             diarization = self.diarize_pipe(wav_path)
#             print("done")
            
#             # Extract segments
#             segments = []
#             for turn, _, speaker in diarization.itertracks(yield_label=True):
#                 segments.append({
#                     "speaker": speaker,
#                     "start": round(turn.start, 2),
#                     "end": round(turn.end, 2)
#                 })
            
#             speakers = sorted({seg['speaker'] for seg in segments})
#             print(f"   ✅ Detected {len(speakers)} speakers: {', '.join(speakers)}")
            
#             # Transcribe segments
#             print(f"   🔊 Transcribing {len(segments)} segments…")
#             transcripts = []
#             speaker_texts = defaultdict(list)
            
#             for seg in tqdm(segments, desc="   ", unit="seg", leave=False, colour="green"):
#                 start, end = seg["start"], seg["end"]
#                 tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#                 tmp_seg_path = tmp_seg.name
#                 tmp_seg.close()
                
#                 try:
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", wav_path,
#                         "-ss", str(start), "-to", str(end),
#                         "-ac", "1", "-ar", "16000", "-vn", tmp_seg_path
#                     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
#                     result = self.asr(tmp_seg_path)
#                     transcript = result.get("text", "").strip()
                    
#                     if transcript:
#                         output = {
#                             "speaker": seg['speaker'],
#                             "start": start,
#                             "end": end,
#                             "duration": round(end - start, 2),
#                             "transcript": transcript
#                         }
#                         transcripts.append(output)
#                         speaker_texts[seg['speaker']].append(transcript)
                        
#                         # Print in real-time
#                         print(f"      [{start}s → {end}s] {seg['speaker']}: {transcript}")
                
#                 finally:
#                     if os.path.exists(tmp_seg_path):
#                         os.unlink(tmp_seg_path)
            
#             # Cleanup temp WAV if converted
#             if Path(file_path).suffix.lower() != ".wav" and os.path.exists(wav_path):
#                 os.unlink(wav_path)
            
#             # Full speaker transcripts
#             full_transcripts = {}
#             for speaker, texts in speaker_texts.items():
#                 full_transcripts[speaker] = " ".join(texts)
            
#             return {
#                 "file": file_name,
#                 "duration": self.get_audio_duration(file_path),
#                 "speakers": speakers,
#                 "segment_count": len(transcripts),
#                 "segments": transcripts,
#                 "full_transcripts": full_transcripts,
#                 "status": "success"
#             }
        
#         except Exception as e:
#             print(f"      ❌ Error: {str(e)}")
#             return {
#                 "file": file_name,
#                 "status": "error",
#                 "error": str(e)
#             }
    
#     def get_audio_duration(self, file_path: str) -> float:
#         """Get audio duration in seconds"""
#         try:
#             result = subprocess.run(
#                 ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
#                  "-of", "default=noprint_wrappers=1:nokey=1:nokey=1", file_path],
#                 capture_output=True, text=True, timeout=10
#             )
#             return float(result.stdout.strip())
#         except:
#             return 0.0
    
#     def save_output(self, file_path: str, data: dict):
#         """Save transcript output to file"""
#         if data.get("status") != "success":
#             return
        
#         output_file = Path(self.output_dir) / f"{Path(file_path).stem}_transcript.txt"
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(f"DURATION: {data.get('duration', 0):.2f} seconds\n\n")
            
#             # Timestamped segments
#             for seg in data['segments']:
#                 f.write(f"[{seg['start']}s -> {seg['end']}s] {seg['speaker']}: {seg['transcript']}\n")
            
#             f.write("\n" + "="*100 + "\n")
#             f.write("FULL SPEAKER TRANSCRIPTS\n")
#             f.write("="*100 + "\n\n")
            
#             # Full transcripts per speaker
#             for speaker, text in data['full_transcripts'].items():
#                 f.write(f"{speaker}:\n{text}\n\n")
        
#         print(f"   💾 Saved: {output_file}")
    
#     def process_folder(self, folder_path: str):
#         """Process all audio files in folder"""
#         folder = Path(folder_path)
#         if not folder.exists():
#             print(f"❌ Folder not found: {folder_path}")
#             return
        
#         # Supported audio formats
#         audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
#         audio_files = [f for f in folder.iterdir() 
#                       if f.is_file() and f.suffix.lower() in audio_extensions]
        
#         if not audio_files:
#             print(f"❌ No audio files found in: {folder_path}")
#             return
        
#         print(f"\n{'='*100}")
#         print(f"🚀 PyAnnote 4.0.3 + PyTorch {torch.__version__}")
#         print(f"📁 Input folder: {folder_path}")
#         print(f"📊 Found {len(audio_files)} audio files")
#         print(f"💾 Output: {self.output_dir}")
#         print(f"{'='*100}")
        
#         results = []
#         successful = 0
#         failed = 0
        
#         for i, audio_file in enumerate(audio_files, 1):
#             print(f"\n[{i}/{len(audio_files)}]", end="")
#             data = self.process_audio(str(audio_file))
#             results.append(data)
            
#             if data.get("status") == "success":
#                 self.save_output(str(audio_file), data)
#                 successful += 1
#             else:
#                 failed += 1
        
#         # Save summary
#         self.save_summary(results, successful, failed, len(audio_files))
    
#     def save_summary(self, results: list, successful: int, failed: int, total: int):
#         """Save processing summary"""
#         summary_file = Path(self.output_dir) / "SUMMARY.txt"
        
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             f.write("="*100 + "\n")
#             f.write("PROCESSING SUMMARY\n")
#             f.write("="*100 + "\n\n")
#             f.write(f"Total files: {total}\n")
#             f.write(f"Successful: {successful}\n")
#             f.write(f"Failed: {failed}\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
#             f.write("FILE DETAILS:\n")
#             f.write("-"*100 + "\n")
#             for result in results:
#                 f.write(f"File: {result['file']}\n")
#                 if result.get("status") == "success":
#                     f.write(f"  Duration: {result.get('duration', 0):.2f}s\n")
#                     f.write(f"  Speakers: {', '.join(result['speakers'])}\n")
#                     f.write(f"  Segments: {result['segment_count']}\n")
#                 else:
#                     f.write(f"  Status: ERROR - {result.get('error', 'Unknown error')}\n")
#                 f.write("\n")
        
#         print(f"\n{'='*100}")
#         print(f"✅ Processing complete!")
#         print(f"   Successful: {successful}/{total}")
#         print(f"   Failed: {failed}/{total}")
#         print(f"   Summary: {summary_file}")
#         print(f"{'='*100}\n")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Batch process audio files with speaker diarization + transcription"
#     )
#     parser.add_argument(
#         "folder",
#         help="Path to folder containing audio files"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     parser.add_argument(
#         "--output",
#         help="Output directory (default: output_transcripts)",
#         default="output_transcripts"
#     )
#     args = parser.parse_args()
    
#     if not args.hf_token:
#         print("❌ Error: Hugging Face token not provided.\n"
#               "   • Set HF_TOKEN env var: export HF_TOKEN='your_token'\n"
#               "   • Or pass: --hf_token YOUR_TOKEN")
#         sys.exit(1)
    
#     processor = AudioProcessor(args.hf_token, args.output)
#     processor.process_folder(args.folder)


# if __name__ == "__main__":
#     main()
















# import os
# import sys
# import json
# import argparse
# import subprocess
# import tempfile
# import warnings
# from pathlib import Path
# from datetime import datetime
# import torch
# from collections import defaultdict

# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Suppress warnings
# warnings.filterwarnings("ignore", message=r"TensorFloat-32.*", module="pyannote.audio.utils.reproducibility")
# warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", module="pyannote.audio.models.blocks.pooling")

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm


# class AudioProcessor:
#     def __init__(self, hf_token: str, output_dir: str = "output_transcripts"):
#         self.hf_token = hf_token
#         self.output_dir = output_dir
#         self.diarize_pipe = None
#         self.asr = None
        
#         # Create output directory
#         Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
#         # Load models once
#         self.load_models()
    
#     def load_models(self):
#         """Load diarization and ASR models on GPU"""
#         print("\n🧠 Loading diarization model (GPU)…", flush=True)
#         self.diarize_pipe = Pipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1"
#         ).to(torch.device("cuda"))
#         print("✅ Diarization model loaded")
        
#         print("📝 Loading ASR model (GPU)…", flush=True)
#         self.asr = hf_pipeline(
#             "automatic-speech-recognition",
#             model="openai/whisper-tiny.en",
#             device=0
#         )
#         print("✅ ASR model loaded\n")
    
#     def convert_to_wav(self, input_path: str) -> str:
#         """Convert audio file to mono 16 kHz WAV"""
#         ext = os.path.splitext(input_path)[1].lower()
#         if ext == ".wav":
#             return input_path
        
#         print(f"   🔄 Converting to WAV…", end=" ", flush=True)
#         tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_wav_path = tmp_wav.name
#         tmp_wav.close()
        
#         cmd = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-ac", "1", "-ar", "16000", "-vn", tmp_wav_path
#         ]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print("done")
#             return tmp_wav_path
#         except subprocess.CalledProcessError as e:
#             os.unlink(tmp_wav_path)
#             raise RuntimeError(f"ffmpeg failed: {e}")
    
#     def process_audio(self, file_path: str) -> dict:
#         """Process single audio file and return transcript data"""
#         file_name = Path(file_path).name
#         print(f"\n📁 Processing: {file_name}")
        
#         try:
#             # Convert to WAV
#             wav_path = self.convert_to_wav(file_path)
            
#             # Diarization
#             print(f"   🎙️  Running diarization…", end=" ", flush=True)
#             diarization = self.diarize_pipe(wav_path)
#             print("done")
            
#             # Extract segments (PyAnnote 4.0.3 API)
#             segments = []
#             try:
#                 # Try new API first (PyAnnote 4.0.3)
#                 for segment, track, speaker in diarization.itertracks(yield_label=True):
#                     segments.append({
#                         "speaker": speaker,
#                         "start": round(segment.start, 2),
#                         "end": round(segment.end, 2)
#                     })
#             except AttributeError:
#                 # Fallback for PyAnnote 4.0.3 - iterate directly
#                 for segment in diarization:
#                     segments.append({
#                         "speaker": segment.label,
#                         "start": round(segment.start, 2),
#                         "end": round(segment.end, 2)
#                     })
            
#             speakers = sorted({seg['speaker'] for seg in segments})
#             print(f"   ✅ Detected {len(speakers)} speakers: {', '.join(speakers)}")
            
#             # Transcribe segments
#             print(f"   🔊 Transcribing {len(segments)} segments…")
#             transcripts = []
#             speaker_texts = defaultdict(list)
            
#             for seg in tqdm(segments, desc="   ", unit="seg", leave=False, colour="green"):
#                 start, end = seg["start"], seg["end"]
#                 tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#                 tmp_seg_path = tmp_seg.name
#                 tmp_seg.close()
                
#                 try:
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", wav_path,
#                         "-ss", str(start), "-to", str(end),
#                         "-ac", "1", "-ar", "16000", "-vn", tmp_seg_path
#                     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
#                     result = self.asr(tmp_seg_path)
#                     transcript = result.get("text", "").strip()
                    
#                     if transcript:
#                         output = {
#                             "speaker": seg['speaker'],
#                             "start": start,
#                             "end": end,
#                             "duration": round(end - start, 2),
#                             "transcript": transcript
#                         }
#                         transcripts.append(output)
#                         speaker_texts[seg['speaker']].append(transcript)
                        
#                         # Print in real-time
#                         print(f"      [{start}s → {end}s] {seg['speaker']}: {transcript}")
                
#                 finally:
#                     if os.path.exists(tmp_seg_path):
#                         os.unlink(tmp_seg_path)
            
#             # Cleanup temp WAV if converted
#             if Path(file_path).suffix.lower() != ".wav" and os.path.exists(wav_path):
#                 os.unlink(wav_path)
            
#             # Full speaker transcripts
#             full_transcripts = {}
#             for speaker, texts in speaker_texts.items():
#                 full_transcripts[speaker] = " ".join(texts)
            
#             return {
#                 "file": file_name,
#                 "duration": self.get_audio_duration(file_path),
#                 "speakers": speakers,
#                 "segment_count": len(transcripts),
#                 "segments": transcripts,
#                 "full_transcripts": full_transcripts,
#                 "status": "success"
#             }
        
#         except Exception as e:
#             print(f"      ❌ Error: {str(e)}")
#             return {
#                 "file": file_name,
#                 "status": "error",
#                 "error": str(e)
#             }
    
#     def get_audio_duration(self, file_path: str) -> float:
#         """Get audio duration in seconds"""
#         try:
#             result = subprocess.run(
#                 ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
#                  "-of", "default=noprint_wrappers=1:nokey=1:nokey=1", file_path],
#                 capture_output=True, text=True, timeout=10
#             )
#             return float(result.stdout.strip())
#         except:
#             return 0.0
    
#     def save_output(self, file_path: str, data: dict):
#         """Save transcript output to file"""
#         if data.get("status") != "success":
#             return
        
#         output_file = Path(self.output_dir) / f"{Path(file_path).stem}_transcript.txt"
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(f"DURATION: {data.get('duration', 0):.2f} seconds\n\n")
            
#             # Timestamped segments
#             for seg in data['segments']:
#                 f.write(f"[{seg['start']}s -> {seg['end']}s] {seg['speaker']}: {seg['transcript']}\n")
            
#             f.write("\n" + "="*100 + "\n")
#             f.write("FULL SPEAKER TRANSCRIPTS\n")
#             f.write("="*100 + "\n\n")
            
#             # Full transcripts per speaker
#             for speaker, text in data['full_transcripts'].items():
#                 f.write(f"{speaker}:\n{text}\n\n")
        
#         print(f"   💾 Saved: {output_file}")
    
#     def process_folder(self, folder_path: str):
#         """Process all audio files in folder"""
#         folder = Path(folder_path)
#         if not folder.exists():
#             print(f"❌ Folder not found: {folder_path}")
#             return
        
#         # Supported audio formats
#         audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
#         audio_files = [f for f in folder.iterdir() 
#                       if f.is_file() and f.suffix.lower() in audio_extensions]
        
#         if not audio_files:
#             print(f"❌ No audio files found in: {folder_path}")
#             return
        
#         print(f"\n{'='*100}")
#         print(f"🚀 PyAnnote 4.0.3 + PyTorch {torch.__version__}")
#         print(f"📁 Input folder: {folder_path}")
#         print(f"📊 Found {len(audio_files)} audio files")
#         print(f"💾 Output: {self.output_dir}")
#         print(f"{'='*100}")
        
#         results = []
#         successful = 0
#         failed = 0
        
#         for i, audio_file in enumerate(audio_files, 1):
#             print(f"\n[{i}/{len(audio_files)}]", end="")
#             data = self.process_audio(str(audio_file))
#             results.append(data)
            
#             if data.get("status") == "success":
#                 self.save_output(str(audio_file), data)
#                 successful += 1
#             else:
#                 failed += 1
        
#         # Save summary
#         self.save_summary(results, successful, failed, len(audio_files))
    
#     def save_summary(self, results: list, successful: int, failed: int, total: int):
#         """Save processing summary"""
#         summary_file = Path(self.output_dir) / "SUMMARY.txt"
        
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             f.write("="*100 + "\n")
#             f.write("PROCESSING SUMMARY\n")
#             f.write("="*100 + "\n\n")
#             f.write(f"Total files: {total}\n")
#             f.write(f"Successful: {successful}\n")
#             f.write(f"Failed: {failed}\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
#             f.write("FILE DETAILS:\n")
#             f.write("-"*100 + "\n")
#             for result in results:
#                 f.write(f"File: {result['file']}\n")
#                 if result.get("status") == "success":
#                     f.write(f"  Duration: {result.get('duration', 0):.2f}s\n")
#                     f.write(f"  Speakers: {', '.join(result['speakers'])}\n")
#                     f.write(f"  Segments: {result['segment_count']}\n")
#                 else:
#                     f.write(f"  Status: ERROR - {result.get('error', 'Unknown error')}\n")
#                 f.write("\n")
        
#         print(f"\n{'='*100}")
#         print(f"✅ Processing complete!")
#         print(f"   Successful: {successful}/{total}")
#         print(f"   Failed: {failed}/{total}")
#         print(f"   Summary: {summary_file}")
#         print(f"{'='*100}\n")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Batch process audio files with speaker diarization + transcription"
#     )
#     parser.add_argument(
#         "folder",
#         help="Path to folder containing audio files"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     parser.add_argument(
#         "--output",
#         help="Output directory (default: output_transcripts)",
#         default="output_transcripts"
#     )
#     args = parser.parse_args()
    
#     if not args.hf_token:
#         print("❌ Error: Hugging Face token not provided.\n"
#               "   • Set HF_TOKEN env var: export HF_TOKEN='your_token'\n"
#               "   • Or pass: --hf_token YOUR_TOKEN")
#         sys.exit(1)
    
#     processor = AudioProcessor(args.hf_token, args.output)
#     processor.process_folder(args.folder)


# if __name__ == "__main__":
#     main()




#***********************************************this is doing file wise  i need single transcript**************************************


# import os
# import sys
# import json
# import argparse
# import subprocess
# import tempfile
# import warnings
# from pathlib import Path
# from datetime import datetime
# import torch
# from collections import defaultdict
# from huggingface_hub import login

# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Suppress warnings
# warnings.filterwarnings("ignore", message=r"TensorFloat-32.*", module="pyannote.audio.utils.reproducibility")
# warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", module="pyannote.audio.models.blocks.pooling")

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm


# class AudioProcessor:
#     def __init__(self, hf_token: str, output_dir: str = "output_transcripts"):
#         self.hf_token = hf_token
#         self.output_dir = output_dir
#         self.diarize_pipe = None
#         self.asr = None
        
#         # Create output directory
#         Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
#         # Load models once
#         self.load_models()
    
#     def load_models(self):
#         """Load diarization and ASR models on GPU"""
#         print("\n🧠 Loading diarization model (GPU)…", flush=True)
#         try:
#             # Login to HuggingFace with token
#             if self.hf_token:
#                 login(token=self.hf_token, add_to_git_credential=False)
            
#             # For newer versions of pyannote.audio
#             self.diarize_pipe = Pipeline.from_pretrained(
#                 "pyannote/speaker-diarization-3.1",
#                 token=self.hf_token  # Use 'token' instead of 'use_auth_token'
#             )
            
#             # Move to GPU if available
#             if torch.cuda.is_available():
#                 self.diarize_pipe = self.diarize_pipe.to(torch.device("cuda"))
            
#             print("✅ Diarization model loaded")
#         except Exception as e:
#             print(f"❌ Failed to load diarization model: {e}")
#             sys.exit(1)
        
#         print("📝 Loading ASR model (GPU)…", flush=True)
#         self.asr = hf_pipeline(
#             "automatic-speech-recognition",
#             model="openai/whisper-tiny.en",
#             device=0 if torch.cuda.is_available() else -1,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#         )
#         print("✅ ASR model loaded\n")
    
#     def convert_to_wav(self, input_path: str) -> str:
#         """Convert audio file to mono 16 kHz WAV"""
#         ext = os.path.splitext(input_path)[1].lower()
#         if ext == ".wav":
#             return input_path
        
#         print(f"   🔄 Converting to WAV…", end=" ", flush=True)
#         tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_wav_path = tmp_wav.name
#         tmp_wav.close()
        
#         cmd = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-ac", "1", "-ar", "16000", "-vn", tmp_wav_path
#         ]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print("done")
#             return tmp_wav_path
#         except subprocess.CalledProcessError as e:
#             os.unlink(tmp_wav_path)
#             raise RuntimeError(f"ffmpeg failed: {e}")
    
#     def extract_segments(self, diarization):
#         """Extract segments from diarization output - handles DiarizeOutput dataclass"""
#         segments = []
#         try:
#             # The diarization is a DiarizeOutput dataclass
#             # We need to access the speaker_diarization attribute which contains the actual diarization
#             if hasattr(diarization, 'speaker_diarization'):
#                 diarization_result = diarization.speaker_diarization
                
#                 # Now iterate through the diarization result
#                 for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#                     segments.append({
#                         "speaker": speaker,
#                         "start": round(turn.start, 2),
#                         "end": round(turn.end, 2)
#                     })
#                 return segments
#             else:
#                 # Fallback: try to see if it's directly iterable
#                 for turn, _, speaker in diarization.itertracks(yield_label=True):
#                     segments.append({
#                         "speaker": speaker,
#                         "start": round(turn.start, 2),
#                         "end": round(turn.end, 2)
#                     })
#                 return segments
                
#         except Exception as e:
#             raise RuntimeError(f"Failed to extract segments: {str(e)}")
    
#     def process_audio(self, file_path: str) -> dict:
#         """Process single audio file and return transcript data"""
#         file_name = Path(file_path).name
#         print(f"\n📁 Processing: {file_name}")
        
#         try:
#             # Convert to WAV
#             wav_path = self.convert_to_wav(file_path)
            
#             # Diarization
#             print(f"   🎙️  Running diarization…", end=" ", flush=True)
#             try:
#                 diarization = self.diarize_pipe(wav_path)
#                 print("done")
#             except Exception as e:
#                 print(f"failed")
#                 raise RuntimeError(f"Diarization failed: {str(e)[:100]}")
            
#             # Extract segments
#             try:
#                 segments = self.extract_segments(diarization)
#             except Exception as e:
#                 raise RuntimeError(f"Segment extraction failed: {str(e)[:100]}")
            
#             if not segments:
#                 raise RuntimeError("No speakers detected in audio")
            
#             speakers = sorted({seg['speaker'] for seg in segments})
#             print(f"   ✅ Detected {len(speakers)} speakers: {', '.join(speakers)}")
            
#             # Transcribe segments
#             print(f"   🔊 Transcribing {len(segments)} segments…")
#             transcripts = []
#             speaker_texts = defaultdict(list)
            
#             for seg in tqdm(segments, desc="   ", unit="seg", leave=False, colour="green"):
#                 start, end = seg["start"], seg["end"]
#                 tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#                 tmp_seg_path = tmp_seg.name
#                 tmp_seg.close()
                
#                 try:
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", wav_path,
#                         "-ss", str(start), "-to", str(end),
#                         "-ac", "1", "-ar", "16000", "-vn", tmp_seg_path
#                     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
#                     result = self.asr(tmp_seg_path)
#                     transcript = result.get("text", "").strip()
                    
#                     if transcript:
#                         output = {
#                             "speaker": seg['speaker'],
#                             "start": start,
#                             "end": end,
#                             "duration": round(end - start, 2),
#                             "transcript": transcript
#                         }
#                         transcripts.append(output)
#                         speaker_texts[seg['speaker']].append(transcript)
                        
#                         # Print in real-time
#                         print(f"      [{start}s → {end}s] {seg['speaker']}: {transcript}")
                
#                 finally:
#                     if os.path.exists(tmp_seg_path):
#                         os.unlink(tmp_seg_path)
            
#             # Cleanup temp WAV if converted
#             if Path(file_path).suffix.lower() != ".wav" and os.path.exists(wav_path):
#                 os.unlink(wav_path)
            
#             # Full speaker transcripts
#             full_transcripts = {}
#             for speaker, texts in speaker_texts.items():
#                 full_transcripts[speaker] = " ".join(texts)
            
#             return {
#                 "file": file_name,
#                 "duration": self.get_audio_duration(file_path),
#                 "speakers": speakers,
#                 "segment_count": len(transcripts),
#                 "segments": transcripts,
#                 "full_transcripts": full_transcripts,
#                 "status": "success"
#             }
        
#         except KeyboardInterrupt:
#             raise
#         except Exception as e:
#             error_msg = str(e)[:100]
#             print(f"      ❌ Error: {error_msg}")
#             return {
#                 "file": file_name,
#                 "status": "error",
#                 "error": error_msg
#             }
    
#     def get_audio_duration(self, file_path: str) -> float:
#         """Get audio duration in seconds"""
#         try:
#             result = subprocess.run(
#                 ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
#                  "-of", "default=noprint_wrappers=1:nokey=1:nokey=1", file_path],
#                 capture_output=True, text=True, timeout=10
#             )
#             return float(result.stdout.strip())
#         except:
#             return 0.0
    
#     def save_output(self, file_path: str, data: dict):
#         """Save transcript output to file"""
#         if data.get("status") != "success":
#             return
        
#         output_file = Path(self.output_dir) / f"{Path(file_path).stem}_transcript.txt"
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(f"DURATION: {data.get('duration', 0):.2f} seconds\n\n")
            
#             # Timestamped segments
#             for seg in data['segments']:
#                 f.write(f"[{seg['start']}s -> {seg['end']}s] {seg['speaker']}: {seg['transcript']}\n")
            
#             f.write("\n" + "="*100 + "\n")
#             f.write("FULL SPEAKER TRANSCRIPTS\n")
#             f.write("="*100 + "\n\n")
            
#             # Full transcripts per speaker
#             for speaker, text in data['full_transcripts'].items():
#                 f.write(f"{speaker}:\n{text}\n\n")
        
#         print(f"   💾 Saved: {output_file}")
    
#     def process_folder(self, folder_path: str):
#         """Process all audio files in folder"""
#         folder = Path(folder_path)
#         if not folder.exists():
#             print(f"❌ Folder not found: {folder_path}")
#             return
        
#         # Supported audio formats
#         audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
#         audio_files = [f for f in folder.iterdir() 
#                       if f.is_file() and f.suffix.lower() in audio_extensions]
        
#         if not audio_files:
#             print(f"❌ No audio files found in: {folder_path}")
#             return
        
#         # Process only first few files for testing
#         test_files = audio_files[:5]  # Just test with first 5 files
#         print(f"\n{'='*100}")
#         print(f"🚀 Testing with first {len(test_files)} files")
#         print(f"📁 Input folder: {folder_path}")
#         print(f"📊 Total files: {len(audio_files)}")
#         print(f"💾 Output: {self.output_dir}")
#         print(f"{'='*100}")
        
#         results = []
#         successful = 0
#         failed = 0
        
#         for i, audio_file in enumerate(test_files, 1):
#             print(f"\n[{i}/{len(test_files)}]", end="")
#             data = self.process_audio(str(audio_file))
#             results.append(data)
            
#             if data.get("status") == "success":
#                 self.save_output(str(audio_file), data)
#                 successful += 1
#             else:
#                 failed += 1
        
#         # Save summary
#         self.save_summary(results, successful, failed, len(test_files))
    
#     def save_summary(self, results: list, successful: int, failed: int, total: int):
#         """Save processing summary"""
#         summary_file = Path(self.output_dir) / "SUMMARY.txt"
        
#         with open(summary_file, 'w', encoding='utf-8') as f:
#             f.write("="*100 + "\n")
#             f.write("PROCESSING SUMMARY\n")
#             f.write("="*100 + "\n\n")
#             f.write(f"Total files: {total}\n")
#             f.write(f"Successful: {successful}\n")
#             f.write(f"Failed: {failed}\n")
#             f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
#             f.write("FILE DETAILS:\n")
#             f.write("-"*100 + "\n")
#             for result in results:
#                 f.write(f"File: {result['file']}\n")
#                 if result.get("status") == "success":
#                     f.write(f"  Duration: {result.get('duration', 0):.2f}s\n")
#                     f.write(f"  Speakers: {', '.join(result['speakers'])}\n")
#                     f.write(f"  Segments: {result['segment_count']}\n")
#                 else:
#                     f.write(f"  Status: ERROR - {result.get('error', 'Unknown error')}\n")
#                 f.write("\n")
        
#         print(f"\n{'='*100}")
#         print(f"✅ Processing complete!")
#         print(f"   Successful: {successful}/{total}")
#         print(f"   Failed: {failed}/{total}")
#         print(f"   Summary: {summary_file}")
#         print(f"{'='*100}\n")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Batch process audio files with speaker diarization + transcription"
#     )
#     parser.add_argument(
#         "folder",
#         help="Path to folder containing audio files"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     parser.add_argument(
#         "--output",
#         help="Output directory (default: output_transcripts)",
#         default="output_transcripts"
#     )
#     parser.add_argument(
#         "--test",
#         help="Test mode: process only first N files",
#         type=int,
#         default=0
#     )
#     args = parser.parse_args()
    
#     if not args.hf_token:
#         print("❌ Error: Hugging Face token not provided.\n"
#               "   • Set HF_TOKEN env var: export HF_TOKEN='your_token'\n"
#               "   • Or pass: --hf_token YOUR_TOKEN")
#         sys.exit(1)
    
#     try:
#         processor = AudioProcessor(args.hf_token, args.output)
#         processor.process_folder(args.folder)
#     except KeyboardInterrupt:
#         print("\n\n⚠️  Processing interrupted by user")
#         sys.exit(0)


# if __name__ == "__main__":
#     main()





# import os
# import sys
# import json
# import argparse
# import subprocess
# import tempfile
# import warnings
# from pathlib import Path
# from datetime import datetime
# import torch
# from collections import defaultdict
# from huggingface_hub import login

# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Suppress warnings
# warnings.filterwarnings("ignore", message=r"TensorFloat-32.*", module="pyannote.audio.utils.reproducibility")
# warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", module="pyannote.audio.models.blocks.pooling")

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline
# from tqdm import tqdm


# class AudioProcessor:
#     def __init__(self, hf_token: str, output_dir: str = "output_transcripts"):
#         self.hf_token = hf_token
#         self.output_dir = output_dir
#         self.diarize_pipe = None
#         self.asr = None
#         self.all_transcripts = []  # Store all transcripts
        
#         # Create output directory
#         Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
#         # Load models once
#         self.load_models()
    
#     def load_models(self):
#         """Load diarization and ASR models on GPU"""
#         print("\n🧠 Loading diarization model (GPU)…", flush=True)
#         try:
#             # Login to HuggingFace with token
#             if self.hf_token:
#                 login(token=self.hf_token, add_to_git_credential=False)
            
#             # For newer versions of pyannote.audio
#             self.diarize_pipe = Pipeline.from_pretrained(
#                 "pyannote/speaker-diarization-3.1",
#                 token=self.hf_token
#             )
            
#             # Move to GPU if available
#             if torch.cuda.is_available():
#                 self.diarize_pipe = self.diarize_pipe.to(torch.device("cuda"))
            
#             print("✅ Diarization model loaded")
#         except Exception as e:
#             print(f"❌ Failed to load diarization model: {e}")
#             sys.exit(1)
        
#         print("📝 Loading ASR model (GPU)…", flush=True)
#         self.asr = hf_pipeline(
#             "automatic-speech-recognition",
#             model="openai/whisper-tiny.en",
#             device=0 if torch.cuda.is_available() else -1,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#         )
#         print("✅ ASR model loaded\n")
    
#     def convert_to_wav(self, input_path: str) -> str:
#         """Convert audio file to mono 16 kHz WAV"""
#         ext = os.path.splitext(input_path)[1].lower()
#         if ext == ".wav":
#             return input_path
        
#         print(f"   🔄 Converting to WAV…", end=" ", flush=True)
#         tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_wav_path = tmp_wav.name
#         tmp_wav.close()
        
#         cmd = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-ac", "1", "-ar", "16000", "-vn", tmp_wav_path
#         ]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print("done")
#             return tmp_wav_path
#         except subprocess.CalledProcessError as e:
#             os.unlink(tmp_wav_path)
#             raise RuntimeError(f"ffmpeg failed: {e}")
    
#     def extract_segments(self, diarization):
#         """Extract segments from diarization output - handles DiarizeOutput dataclass"""
#         segments = []
#         try:
#             # The diarization is a DiarizeOutput dataclass
#             # We need to access the speaker_diarization attribute which contains the actual diarization
#             if hasattr(diarization, 'speaker_diarization'):
#                 diarization_result = diarization.speaker_diarization
                
#                 # Now iterate through the diarization result
#                 for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#                     segments.append({
#                         "speaker": speaker,
#                         "start": round(turn.start, 2),
#                         "end": round(turn.end, 2)
#                     })
#                 return segments
#             else:
#                 # Fallback: try to see if it's directly iterable
#                 for turn, _, speaker in diarization.itertracks(yield_label=True):
#                     segments.append({
#                         "speaker": speaker,
#                         "start": round(turn.start, 2),
#                         "end": round(turn.end, 2)
#                     })
#                 return segments
                
#         except Exception as e:
#             raise RuntimeError(f"Failed to extract segments: {str(e)}")
    
#     def process_audio(self, file_path: str) -> dict:
#         """Process single audio file and return transcript data"""
#         file_name = Path(file_path).name
#         print(f"\n📁 Processing: {file_name}")
        
#         try:
#             # Convert to WAV
#             wav_path = self.convert_to_wav(file_path)
            
#             # Diarization
#             print(f"   🎙️  Running diarization…", end=" ", flush=True)
#             try:
#                 diarization = self.diarize_pipe(wav_path)
#                 print("done")
#             except Exception as e:
#                 print(f"failed")
#                 raise RuntimeError(f"Diarization failed: {str(e)[:100]}")
            
#             # Extract segments
#             try:
#                 segments = self.extract_segments(diarization)
#             except Exception as e:
#                 raise RuntimeError(f"Segment extraction failed: {str(e)[:100]}")
            
#             if not segments:
#                 raise RuntimeError("No speakers detected in audio")
            
#             speakers = sorted({seg['speaker'] for seg in segments})
#             print(f"   ✅ Detected {len(speakers)} speakers: {', '.join(speakers)}")
            
#             # Transcribe segments
#             print(f"   🔊 Transcribing {len(segments)} segments…")
#             transcripts = []
#             speaker_texts = defaultdict(list)
            
#             for seg in tqdm(segments, desc="   ", unit="seg", leave=False, colour="green"):
#                 start, end = seg["start"], seg["end"]
#                 tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#                 tmp_seg_path = tmp_seg.name
#                 tmp_seg.close()
                
#                 try:
#                     subprocess.run([
#                         "ffmpeg", "-y", "-i", wav_path,
#                         "-ss", str(start), "-to", str(end),
#                         "-ac", "1", "-ar", "16000", "-vn", tmp_seg_path
#                     ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
#                     result = self.asr(tmp_seg_path)
#                     transcript = result.get("text", "").strip()
                    
#                     if transcript:
#                         output = {
#                             "speaker": seg['speaker'],
#                             "start": start,
#                             "end": end,
#                             "duration": round(end - start, 2),
#                             "transcript": transcript
#                         }
#                         transcripts.append(output)
#                         speaker_texts[seg['speaker']].append(transcript)
                        
#                         # Print in real-time
#                         print(f"      [{start}s → {end}s] {seg['speaker']}: {transcript}")
                
#                 finally:
#                     if os.path.exists(tmp_seg_path):
#                         os.unlink(tmp_seg_path)
            
#             # Cleanup temp WAV if converted
#             if Path(file_path).suffix.lower() != ".wav" and os.path.exists(wav_path):
#                 os.unlink(wav_path)
            
#             # Full speaker transcripts
#             full_transcripts = {}
#             for speaker, texts in speaker_texts.items():
#                 full_transcripts[speaker] = " ".join(texts)
            
#             # Store combined transcript for this file
#             file_transcript = {
#                 "file": file_name,
#                 "duration": self.get_audio_duration(file_path),
#                 "speakers": speakers,
#                 "segment_count": len(transcripts),
#                 "segments": transcripts,
#                 "full_transcripts": full_transcripts,
#                 "status": "success"
#             }
            
#             # Add to all transcripts
#             self.all_transcripts.append(file_transcript)
            
#             return file_transcript
        
#         except KeyboardInterrupt:
#             raise
#         except Exception as e:
#             error_msg = str(e)[:100]
#             print(f"      ❌ Error: {error_msg}")
            
#             # Add error entry to all transcripts
#             error_data = {
#                 "file": file_name,
#                 "status": "error",
#                 "error": error_msg
#             }
#             self.all_transcripts.append(error_data)
            
#             return error_data
    
#     def get_audio_duration(self, file_path: str) -> float:
#         """Get audio duration in seconds"""
#         try:
#             result = subprocess.run(
#                 ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
#                  "-of", "default=noprint_wrappers=1:nokey=1:nokey=1", file_path],
#                 capture_output=True, text=True, timeout=10
#             )
#             return float(result.stdout.strip())
#         except:
#             return 0.0
    
#     def save_combined_transcript(self):
#         """Save ALL transcripts into one single file"""
#         if not self.all_transcripts:
#             print("❌ No transcripts to save!")
#             return
        
#         output_file = Path(self.output_dir) / "ALL_TRANSCRIPTS_COMBINED.txt"
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write("="*100 + "\n")
#             f.write("COMBINED TRANSCRIPTS - ALL AUDIO FILES\n")
#             f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write("="*100 + "\n\n")
            
#             successful_files = 0
#             failed_files = 0
#             total_duration = 0
#             total_segments = 0
            
#             for i, data in enumerate(self.all_transcripts, 1):
#                 f.write(f"FILE {i}: {data['file']}\n")
#                 f.write("-"*80 + "\n")
                
#                 if data.get("status") == "success":
#                     successful_files += 1
#                     total_duration += data.get('duration', 0)
#                     total_segments += data.get('segment_count', 0)
                    
#                     f.write(f"DURATION: {data.get('duration', 0):.2f} seconds\n")
#                     f.write(f"SPEAKERS: {', '.join(data['speakers'])}\n")
#                     f.write(f"SEGMENTS: {data['segment_count']}\n\n")
                    
#                     # Timestamped segments
#                     for seg in data['segments']:
#                         f.write(f"[{seg['start']:6.2f}s → {seg['end']:6.2f}s] {seg['speaker']}: {seg['transcript']}\n")
                    
#                     f.write("\nFULL SPEAKER TRANSCRIPTS:\n")
#                     f.write("-"*40 + "\n")
                    
#                     # Full transcripts per speaker
#                     for speaker, text in data['full_transcripts'].items():
#                         f.write(f"\n{speaker}:\n{text}\n")
                    
#                     f.write("\n" + "="*80 + "\n\n")
#                 else:
#                     failed_files += 1
#                     f.write(f"❌ ERROR: {data.get('error', 'Unknown error')}\n")
#                     f.write("\n" + "="*80 + "\n\n")
            
#             # Write summary at the end
#             f.write("\n" + "="*100 + "\n")
#             f.write("PROCESSING SUMMARY\n")
#             f.write("="*100 + "\n\n")
#             f.write(f"Total files processed: {len(self.all_transcripts)}\n")
#             f.write(f"Successful files: {successful_files}\n")
#             f.write(f"Failed files: {failed_files}\n")
#             f.write(f"Total duration: {total_duration:.2f} seconds\n")
#             f.write(f"Total segments: {total_segments}\n")
#             f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
#         print(f"\n💾 Saved ALL transcripts to: {output_file}")
#         print(f"   Total files: {len(self.all_transcripts)}")
#         print(f"   Successful: {successful_files}")
#         print(f"   Failed: {failed_files}")
    
#     def process_folder(self, folder_path: str, test_limit: int = 0):
#         """Process all audio files in folder"""
#         folder = Path(folder_path)
#         if not folder.exists():
#             print(f"❌ Folder not found: {folder_path}")
#             return
        
#         # Supported audio formats
#         audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
#         audio_files = [f for f in folder.iterdir() 
#                       if f.is_file() and f.suffix.lower() in audio_extensions]
        
#         if not audio_files:
#             print(f"❌ No audio files found in: {folder_path}")
#             return
        
#         # Sort files by name for consistent processing
#         audio_files.sort(key=lambda x: x.name)
        
#         # Apply test limit if specified
#         if test_limit > 0:
#             audio_files = audio_files[:test_limit]
#             print(f"\n🔬 TEST MODE: Processing first {test_limit} files")
#         else:
#             print(f"\nProcessing ALL {len(audio_files)} files")
        
#         print(f"\n{'='*100}")
#         print(f"🚀 PyAnnote + Whisper Transcription")
#         print(f"📁 Input folder: {folder_path}")
#         print(f"📊 Files to process: {len(audio_files)}")
#         print(f"💾 Output: {self.output_dir}")
#         print(f"{'='*100}")
        
#         successful = 0
#         failed = 0
        
#         for i, audio_file in enumerate(audio_files, 1):
#             print(f"\n[{i}/{len(audio_files)}]", end="")
#             data = self.process_audio(str(audio_file))
            
#             if data.get("status") == "success":
#                 successful += 1
#             else:
#                 failed += 1
        
#         # Save ALL transcripts into one file
#         self.save_combined_transcript()
        
#         print(f"\n{'='*100}")
#         print(f"✅ Processing complete!")
#         print(f"   Successful: {successful}/{len(audio_files)}")
#         print(f"   Failed: {failed}/{len(audio_files)}")
#         print(f"   Combined transcript saved to: {self.output_dir}/ALL_TRANSCRIPTS_COMBINED.txt")
#         print(f"{'='*100}\n")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Batch process audio files with speaker diarization + transcription"
#     )
#     parser.add_argument(
#         "folder",
#         help="Path to folder containing audio files"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     parser.add_argument(
#         "--output",
#         help="Output directory (default: output_transcripts)",
#         default="output_transcripts"
#     )
#     parser.add_argument(
#         "--test",
#         help="Test mode: process only first N files (0 = all files)",
#         type=int,
#         default=0
#     )
#     args = parser.parse_args()
    
#     if not args.hf_token:
#         print("❌ Error: Hugging Face token not provided.\n"
#               "   • Set HF_TOKEN env var: export HF_TOKEN='your_token'\n"
#               "   • Or pass: --hf_token YOUR_TOKEN")
#         sys.exit(1)
    
#     try:
#         processor = AudioProcessor(args.hf_token, args.output)
#         processor.process_folder(args.folder, args.test)
#     except KeyboardInterrupt:
#         print("\n\n⚠️  Processing interrupted by user")
#         # Still save what we have
#         if hasattr(processor, 'all_transcripts') and processor.all_transcripts:
#             processor.save_combined_transcript()
#         sys.exit(0)


# if __name__ == "__main__":
#     main()







#better version but still providing issues 



# import os
# import sys
# import json
# import argparse
# import subprocess
# import tempfile
# import warnings
# import re
# from pathlib import Path
# from datetime import datetime
# import torch
# from collections import defaultdict
# from huggingface_hub import login
# import numpy as np

# # Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # Suppress warnings
# warnings.filterwarnings("ignore", message=r"TensorFloat-32.*", module="pyannote.audio.utils.reproducibility")
# warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", module="pyannote.audio.models.blocks.pooling")
# warnings.filterwarnings("ignore", category=UserWarning)

# from pyannote.audio import Pipeline
# from transformers import pipeline as hf_pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
# from tqdm import tqdm


# class AudioProcessor:
#     def __init__(self, hf_token: str, output_dir: str = "output_transcripts"):
#         self.hf_token = hf_token
#         self.output_dir = output_dir
#         self.diarize_pipe = None
#         self.asr = None
#         self.all_transcripts = []  # Store all transcripts
        
#         # Quality control parameters
#         self.min_segment_duration = 0.5  # Minimum segment duration in seconds
#         self.max_words_per_second = 4.0  # Maximum reasonable words per second
#         self.min_word_count = 1  # Minimum words to consider a valid transcript
#         self.max_repetition_count = 3  # Maximum consecutive word repetitions
        
#         # Create output directory
#         Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
#         # Load models once
#         self.load_models()
    
#     def load_models(self):
#         """Load diarization and ASR models on GPU"""
#         print("\n🧠 Loading diarization model (GPU)…", flush=True)
#         try:
#             # Login to HuggingFace with token
#             if self.hf_token:
#                 login(token=self.hf_token, add_to_git_credential=False)
            
#             # For newer versions of pyannote.audio
#             self.diarize_pipe = Pipeline.from_pretrained(
#                 "pyannote/speaker-diarization-3.1",
#                 token=self.hf_token
#             )
            
#             # Move to GPU if available
#             if torch.cuda.is_available():
#                 self.diarize_pipe = self.diarize_pipe.to(torch.device("cuda"))
            
#             print("✅ Diarization model loaded")
#         except Exception as e:
#             print(f"❌ Failed to load diarization model: {e}")
#             sys.exit(1)
        
#         print("📝 Loading ASR model (GPU)…", flush=True)
#         try:
#             # Use a better Whisper model with more control
#             model_id = "openai/whisper-small.en"  # English-only model
#             model = AutoModelForSpeechSeq2Seq.from_pretrained(
#                 model_id,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#             )
            
#             processor = AutoProcessor.from_pretrained(model_id)
            
#             # For English-only models, don't specify task/language
#             generate_kwargs = {
#                 "no_repeat_ngram_size": 3,  # Reduce repetitions
#                 "temperature": 0.0,  # More deterministic output
#             }
            
#             self.asr = hf_pipeline(
#                 "automatic-speech-recognition",
#                 model=model,
#                 tokenizer=processor.tokenizer,
#                 feature_extractor=processor.feature_extractor,
#                 device=0 if torch.cuda.is_available() else -1,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 generate_kwargs=generate_kwargs
#             )
#             print("✅ ASR model loaded (Whisper Small)")
#         except Exception as e:
#             print(f"⚠️  Failed to load small model, falling back to tiny: {e}")
#             # Fallback to simpler model
#             self.asr = hf_pipeline(
#                 "automatic-speech-recognition",
#                 model="openai/whisper-tiny.en",
#                 device=0 if torch.cuda.is_available() else -1,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#             )
#             print("✅ ASR model loaded (Whisper Tiny - fallback)")
    
#     def convert_to_wav(self, input_path: str) -> str:
#         """Convert audio file to mono 16 kHz WAV"""
#         ext = os.path.splitext(input_path)[1].lower()
#         if ext == ".wav":
#             return input_path
        
#         print(f"   🔄 Converting to WAV…", end=" ", flush=True)
#         tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#         tmp_wav_path = tmp_wav.name
#         tmp_wav.close()
        
#         cmd = [
#             "ffmpeg", "-y", "-i", input_path,
#             "-ac", "1", "-ar", "16000", "-vn",
#             "-acodec", "pcm_s16le", tmp_wav_path
#         ]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print("done")
#             return tmp_wav_path
#         except subprocess.CalledProcessError as e:
#             os.unlink(tmp_wav_path)
#             raise RuntimeError(f"ffmpeg failed: {e}")
    
#     def extract_segments(self, diarization):
#         """Extract segments from diarization output - handles DiarizeOutput dataclass"""
#         segments = []
#         try:
#             # The diarization is a DiarizeOutput dataclass
#             # We need to access the speaker_diarization attribute
#             if hasattr(diarization, 'speaker_diarization'):
#                 diarization_result = diarization.speaker_diarization
                
#                 # Now iterate through the diarization result
#                 for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#                     duration = turn.end - turn.start
                    
#                     # Only include segments with reasonable duration
#                     if duration >= self.min_segment_duration:
#                         segments.append({
#                             "speaker": speaker,
#                             "start": round(turn.start, 2),
#                             "end": round(turn.end, 2),
#                             "duration": round(duration, 2)
#                         })
#                     else:
#                         print(f"      ⚠️  Skipping short segment: {speaker} [{turn.start:.2f}s → {turn.end:.2f}s] ({duration:.2f}s)")
                
#                 return segments
#             else:
#                 # Fallback for older versions
#                 for turn, _, speaker in diarization.itertracks(yield_label=True):
#                     duration = turn.end - turn.start
#                     if duration >= self.min_segment_duration:
#                         segments.append({
#                             "speaker": speaker,
#                             "start": round(turn.start, 2),
#                             "end": round(turn.end, 2),
#                             "duration": round(duration, 2)
#                         })
#                 return segments
                
#         except Exception as e:
#             raise RuntimeError(f"Failed to extract segments: {str(e)}")
    
#     def clean_transcript(self, text: str, duration: float) -> str:
#         """Clean and validate transcript text"""
#         if not text or text.strip() == "":
#             return ""
        
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text.strip())
        
#         # Check for excessive word repetitions
#         words = text.split()
#         if len(words) > 0:
#             # Count consecutive repetitions
#             current_word = words[0].lower()
#             current_count = 1
#             for word in words[1:]:
#                 if word.lower() == current_word:
#                     current_count += 1
#                     if current_count > self.max_repetition_count:
#                         # Found excessive repetition, clean it
#                         text = " ".join(words[:words.index(word) - self.max_repetition_count + 1])
#                         break
#                 else:
#                     current_word = word.lower()
#                     current_count = 1
        
#         # Calculate words per second
#         word_count = len(text.split())
#         if duration > 0:
#             words_per_second = word_count / duration
            
#             # If words per second is too high, the transcript is likely invalid
#             if words_per_second > self.max_words_per_second:
#                 # Try to trim to reasonable length
#                 max_words = int(duration * self.max_words_per_second)
#                 if max_words > 0:
#                     words = text.split()[:max_words]
#                     text = " ".join(words)
#                     print(f"      ⚠️  Trimmed excessive transcript: {words_per_second:.1f} words/sec")
        
#         # Remove common transcription artifacts
#         artifacts = [
#             "thank you for watching",
#             "please like and subscribe",
#             "[music]",
#             "[applause]",
#             "[laughter]",
#             "um", "uh", "ah", "er",  # Fillers
#         ]
        
#         for artifact in artifacts:
#             text = text.replace(artifact, "")
        
#         # Final cleanup
#         text = re.sub(r'\s+', ' ', text.strip())
#         return text
    
#     def validate_transcript(self, transcript: str, duration: float, segment_start: float, segment_end: float) -> bool:
#         """Validate if a transcript is reasonable for the given segment"""
#         if not transcript or transcript.strip() == "":
#             return False
        
#         # Check word count
#         word_count = len(transcript.split())
#         if word_count < self.min_word_count:
#             return False
        
#         # Check words per second
#         if duration > 0:
#             words_per_second = word_count / duration
#             if words_per_second > self.max_words_per_second:
#                 print(f"      ⚠️  Skipping: {words_per_second:.1f} words/sec is too high")
#                 return False
        
#         # Check for obvious gibberish patterns
#         patterns = [
#             r'(.)\1{5,}',  # Single character repeated 5+ times
#             r'(\w+)( \1){5,}',  # Word repeated 5+ times
#         ]
        
#         for pattern in patterns:
#             if re.search(pattern, transcript, re.IGNORECASE):
#                 return False
        
#         return True
    
#     def process_audio(self, file_path: str) -> dict:
#         """Process single audio file and return transcript data"""
#         file_name = Path(file_path).name
#         print(f"\n📁 Processing: {file_name}")
        
#         try:
#             # Convert to WAV
#             wav_path = self.convert_to_wav(file_path)
            
#             # Diarization
#             print(f"   🎙️  Running diarization…", end=" ", flush=True)
#             try:
#                 diarization = self.diarize_pipe(wav_path)
#                 print("done")
#             except Exception as e:
#                 print(f"failed")
#                 raise RuntimeError(f"Diarization failed: {str(e)[:100]}")
            
#             # Extract segments
#             try:
#                 segments = self.extract_segments(diarization)
#             except Exception as e:
#                 raise RuntimeError(f"Segment extraction failed: {str(e)[:100]}")
            
#             if not segments:
#                 print(f"   ℹ️  No valid speech segments detected")
#                 return {
#                     "file": file_name,
#                     "duration": self.get_audio_duration(file_path),
#                     "speakers": [],
#                     "segment_count": 0,
#                     "segments": [],
#                     "full_transcripts": {},
#                     "status": "success_no_speech"
#                 }
            
#             speakers = sorted({seg['speaker'] for seg in segments})
#             print(f"   ✅ Detected {len(speakers)} speakers: {', '.join(speakers)}")
#             print(f"   📊 Found {len(segments)} valid segments")
            
#             # Transcribe segments
#             transcripts = []
#             speaker_texts = defaultdict(list)
#             skipped_count = 0
            
#             if len(segments) > 0:
#                 print(f"   🔊 Transcribing segments…")
#                 for seg in tqdm(segments, desc="   ", unit="seg", leave=False, colour="green"):
#                     start, end, duration = seg["start"], seg["end"], seg["duration"]
                    
#                     # Skip very short segments (already filtered, but double-check)
#                     if duration < self.min_segment_duration:
#                         skipped_count += 1
#                         continue
                    
#                     tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#                     tmp_seg_path = tmp_seg.name
#                     tmp_seg.close()
                    
#                     try:
#                         # Extract audio segment
#                         subprocess.run([
#                             "ffmpeg", "-y", "-i", wav_path,
#                             "-ss", str(start), "-to", str(end),
#                             "-ac", "1", "-ar", "16000", "-vn",
#                             "-acodec", "pcm_s16le", tmp_seg_path
#                         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
#                         # Transcribe
#                         try:
#                             result = self.asr(tmp_seg_path)
#                             raw_transcript = result.get("text", "").strip()
#                         except Exception as e:
#                             print(f"      ❌ ASR error: {str(e)[:50]}")
#                             raw_transcript = ""
                        
#                         # Clean and validate transcript
#                         clean_transcript = self.clean_transcript(raw_transcript, duration)
                        
#                         if clean_transcript and self.validate_transcript(clean_transcript, duration, start, end):
#                             output = {
#                                 "speaker": seg['speaker'],
#                                 "start": start,
#                                 "end": end,
#                                 "duration": duration,
#                                 "transcript": clean_transcript,
#                                 "word_count": len(clean_transcript.split())
#                             }
#                             transcripts.append(output)
#                             speaker_texts[seg['speaker']].append(clean_transcript)
                            
#                             # Print in real-time
#                             word_count = len(clean_transcript.split())
#                             wps = word_count / duration if duration > 0 else 0
#                             print(f"      [{start:6.2f}s → {end:6.2f}s] {seg['speaker']}: {clean_transcript[:80]}{'...' if len(clean_transcript) > 80 else ''}")
#                             if wps > 3:
#                                 print(f"         (words: {word_count}, {wps:.1f} words/sec)")
#                         else:
#                             skipped_count += 1
#                             if raw_transcript:
#                                 print(f"      ⚠️  Skipped: {seg['speaker']} [{start:.2f}s → {end:.2f}s] - Invalid transcript")
                    
#                     except Exception as e:
#                         skipped_count += 1
#                         print(f"      ❌ Error transcribing segment: {str(e)[:50]}")
                    
#                     finally:
#                         if os.path.exists(tmp_seg_path):
#                             os.unlink(tmp_seg_path)
            
#             # Cleanup temp WAV if converted
#             if Path(file_path).suffix.lower() != ".wav" and os.path.exists(wav_path):
#                 os.unlink(wav_path)
            
#             if skipped_count > 0:
#                 print(f"   ⚠️  Skipped {skipped_count} invalid segments")
            
#             # Full speaker transcripts
#             full_transcripts = {}
#             for speaker, texts in speaker_texts.items():
#                 if texts:  # Only include speakers with valid transcripts
#                     full_transcripts[speaker] = " ".join(texts)
            
#             return {
#                 "file": file_name,
#                 "duration": self.get_audio_duration(file_path),
#                 "speakers": list(full_transcripts.keys()),  # Only speakers with valid transcripts
#                 "segment_count": len(transcripts),
#                 "segments": transcripts,
#                 "full_transcripts": full_transcripts,
#                 "status": "success" if transcripts else "success_no_speech"
#             }
        
#         except KeyboardInterrupt:
#             raise
#         except Exception as e:
#             error_msg = str(e)[:100]
#             print(f"      ❌ Error: {error_msg}")
#             return {
#                 "file": file_name,
#                 "status": "error",
#                 "error": error_msg
#             }
    
#     def get_audio_duration(self, file_path: str) -> float:
#         """Get audio duration in seconds"""
#         try:
#             result = subprocess.run(
#                 ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
#                  "-of", "default=noprint_wrappers=1:nokey=1", file_path],
#                 capture_output=True, text=True, timeout=10
#             )
#             return float(result.stdout.strip())
#         except:
#             return 0.0
    
#     def save_combined_transcript(self):
#         """Save ALL transcripts into one single file"""
#         if not self.all_transcripts:
#             print("❌ No transcripts to save!")
#             return
        
#         output_file = Path(self.output_dir) / "ALL_TRANSCRIPTS_COMBINED.txt"
        
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write("="*120 + "\n")
#             f.write("COMBINED TRANSCRIPTS - ALL AUDIO FILES\n")
#             f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write("="*120 + "\n\n")
            
#             successful_files = 0
#             failed_files = 0
#             no_speech_files = 0
#             total_duration = 0
#             total_segments = 0
#             total_words = 0
            
#             for i, data in enumerate(self.all_transcripts, 1):
#                 f.write(f"\n{'='*80}\n")
#                 f.write(f"FILE {i}: {data['file']}\n")
#                 f.write(f"{'='*80}\n")
                
#                 if data.get("status") == "success":
#                     successful_files += 1
#                     file_duration = data.get('duration', 0)
#                     total_duration += file_duration
#                     total_segments += data.get('segment_count', 0)
                    
#                     f.write(f"DURATION: {file_duration:.2f} seconds\n")
#                     f.write(f"SPEAKERS: {', '.join(data['speakers']) if data['speakers'] else 'None'}\n")
#                     f.write(f"SEGMENTS: {data['segment_count']}\n\n")
                    
#                     # Timestamped segments
#                     for seg in data['segments']:
#                         word_count = len(seg['transcript'].split())
#                         total_words += word_count
#                         f.write(f"[{seg['start']:6.2f}s → {seg['end']:6.2f}s] {seg['speaker']}: {seg['transcript']}\n")
                    
#                     if data['full_transcripts']:
#                         f.write("\nFULL SPEAKER TRANSCRIPTS:\n")
#                         f.write("-"*40 + "\n")
                        
#                         # Full transcripts per speaker
#                         for speaker, text in data['full_transcripts'].items():
#                             word_count = len(text.split())
#                             f.write(f"\n{speaker} ({word_count} words):\n{text}\n")
                    
#                     f.write(f"\n{'='*80}\n")
                    
#                 elif data.get("status") == "success_no_speech":
#                     no_speech_files += 1
#                     f.write(f"⚠️  NO SPEECH DETECTED\n")
#                     f.write(f"DURATION: {data.get('duration', 0):.2f} seconds\n")
#                     f.write(f"{'='*80}\n")
                    
#                 else:
#                     failed_files += 1
#                     f.write(f"❌ ERROR: {data.get('error', 'Unknown error')}\n")
#                     f.write(f"{'='*80}\n")
            
#             # Write summary at the end
#             f.write("\n\n" + "="*120 + "\n")
#             f.write("PROCESSING SUMMARY\n")
#             f.write("="*120 + "\n\n")
#             f.write(f"Total files processed: {len(self.all_transcripts)}\n")
#             f.write(f"Successful (with speech): {successful_files}\n")
#             f.write(f"Successful (no speech): {no_speech_files}\n")
#             f.write(f"Failed files: {failed_files}\n")
#             f.write(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)\n")
#             f.write(f"Total segments: {total_segments}\n")
#             f.write(f"Total words: {total_words}\n")
#             if successful_files > 0:
#                 f.write(f"Average words per file: {total_words/successful_files:.1f}\n")
#                 f.write(f"Average segments per file: {total_segments/successful_files:.1f}\n")
#             f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
#         print(f"\n💾 Saved ALL transcripts to: {output_file}")
#         print(f"   Total files: {len(self.all_transcripts)}")
#         print(f"   Successful (with speech): {successful_files}")
#         print(f"   Successful (no speech): {no_speech_files}")
#         print(f"   Failed: {failed_files}")
#         if successful_files > 0:
#             print(f"   Total words: {total_words}")
    
#     def process_folder(self, folder_path: str, test_limit: int = 0):
#         """Process all audio files in folder"""
#         folder = Path(folder_path)
#         if not folder.exists():
#             print(f"❌ Folder not found: {folder_path}")
#             return
        
#         # Supported audio formats
#         audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
#         audio_files = [f for f in folder.iterdir() 
#                       if f.is_file() and f.suffix.lower() in audio_extensions]
        
#         if not audio_files:
#             print(f"❌ No audio files found in: {folder_path}")
#             return
        
#         # Sort files by name for consistent processing
#         audio_files.sort(key=lambda x: x.name)
        
#         # Apply test limit if specified
#         if test_limit > 0:
#             audio_files = audio_files[:test_limit]
#             print(f"\n🔬 TEST MODE: Processing first {test_limit} files")
#         else:
#             print(f"\nProcessing ALL {len(audio_files)} files")
        
#         print(f"\n{'='*100}")
#         print(f"🚀 PyAnnote + Whisper Transcription")
#         print(f"📁 Input folder: {folder_path}")
#         print(f"📊 Files to process: {len(audio_files)}")
#         print(f"💾 Output: {self.output_dir}")
#         print(f"⚙️  Settings:")
#         print(f"   - Min segment duration: {self.min_segment_duration}s")
#         print(f"   - Max words per second: {self.max_words_per_second}")
#         print(f"   - Min words per segment: {self.min_word_count}")
#         print(f"{'='*100}")
        
#         successful = 0
#         failed = 0
#         no_speech = 0
        
#         for i, audio_file in enumerate(audio_files, 1):
#             print(f"\n[{i}/{len(audio_files)}]", end="")
#             data = self.process_audio(str(audio_file))
#             self.all_transcripts.append(data)
            
#             if data.get("status") == "success":
#                 successful += 1
#             elif data.get("status") == "success_no_speech":
#                 no_speech += 1
#             else:
#                 failed += 1
        
#         # Save ALL transcripts into one file
#         self.save_combined_transcript()
        
#         print(f"\n{'='*100}")
#         print(f"✅ Processing complete!")
#         print(f"   Total processed: {len(audio_files)}")
#         print(f"   Successful (with speech): {successful}")
#         print(f"   Successful (no speech): {no_speech}")
#         print(f"   Failed: {failed}")
#         print(f"   Combined transcript: {self.output_dir}/ALL_TRANSCRIPTS_COMBINED.txt")
#         print(f"{'='*100}\n")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Batch process audio files with speaker diarization + transcription"
#     )
#     parser.add_argument(
#         "folder",
#         help="Path to folder containing audio files"
#     )
#     parser.add_argument(
#         "--hf_token",
#         help="Hugging Face token (default: from HF_TOKEN env var)",
#         default=os.getenv("HF_TOKEN")
#     )
#     parser.add_argument(
#         "--output",
#         help="Output directory (default: output_transcripts)",
#         default="output_transcripts"
#     )
#     parser.add_argument(
#         "--test",
#         help="Test mode: process only first N files (0 = all files)",
#         type=int,
#         default=0
#     )
#     parser.add_argument(
#         "--min_duration",
#         help="Minimum segment duration in seconds (default: 0.5)",
#         type=float,
#         default=0.5
#     )
#     parser.add_argument(
#         "--max_wps",
#         help="Maximum words per second (default: 4.0)",
#         type=float,
#         default=4.0
#     )
#     args = parser.parse_args()
    
#     if not args.hf_token:
#         print("❌ Error: Hugging Face token not provided.\n"
#               "   • Set HF_TOKEN env var: export HF_TOKEN='your_token'\n"
#               "   • Or pass: --hf_token YOUR_TOKEN")
#         sys.exit(1)
    
#     try:
#         processor = AudioProcessor(args.hf_token, args.output)
#         # Apply command line parameters
#         processor.min_segment_duration = args.min_duration
#         processor.max_words_per_second = args.max_wps
#         processor.process_folder(args.folder, args.test)
#     except KeyboardInterrupt:
#         print("\n\n⚠️  Processing interrupted by user")
#         # Still save what we have
#         if hasattr(processor, 'all_transcripts') and processor.all_transcripts:
#             processor.save_combined_transcript()
#         sys.exit(0)


# if __name__ == "__main__":
#     main()



# everything was good but i wanna real time update in output file and complete sentces in transcript 



import os
import sys
import json
import argparse
import subprocess
import tempfile
import warnings
import re
import time
from pathlib import Path
from datetime import datetime
import torch
from collections import defaultdict
from huggingface_hub import login

# Enable TensorFloat-32 (TF32) for speed on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress warnings
warnings.filterwarnings("ignore", message=r"TensorFloat-32.*", module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0.*", module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning)

from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm


class AudioProcessor:
    def __init__(self, hf_token: str, output_dir: str = "output_transcripts"):
        self.hf_token = hf_token
        self.output_dir = output_dir
        self.diarize_pipe = None
        self.asr = None
        self.output_file = None
        self.summary_file = None
        
        # Quality control parameters
        self.min_segment_duration = 0.5  # Minimum segment duration in seconds
        self.max_words_per_second = 5.0  # Maximum reasonable words per second
        self.min_word_count = 1  # Minimum words to consider a valid transcript
        self.max_repetition_count = 3  # Maximum consecutive word repetitions
        
        # Statistics tracking
        self.total_files = 0
        self.successful_files = 0
        self.no_speech_files = 0
        self.failed_files = 0
        self.total_duration = 0
        self.total_segments = 0
        self.total_words = 0
        self.start_time = None
        self.processed_files = []
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load models once
        self.load_models()
        
        # Initialize output files
        self.init_output_files()
    
    def load_models(self):
        """Load diarization and ASR models on GPU"""
        print("\n🧠 Loading diarization model (GPU)…", flush=True)
        try:
            # Login to HuggingFace with token
            if self.hf_token:
                login(token=self.hf_token, add_to_git_credential=False)
            
            # For newer versions of pyannote.audio
            self.diarize_pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.diarize_pipe = self.diarize_pipe.to(torch.device("cuda"))
            
            print("✅ Diarization model loaded")
        except Exception as e:
            print(f"❌ Failed to load diarization model: {e}")
            sys.exit(1)
        
        print("📝 Loading ASR model (GPU)…", flush=True)
        try:
            # Use Whisper model
            model_id = "openai/whisper-small.en"  # English-only model
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            # For English-only models, don't specify task/language
            generate_kwargs = {
                "no_repeat_ngram_size": 3,  # Reduce repetitions
                "temperature": 0.0,  # More deterministic output
            }
            
            self.asr = hf_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                generate_kwargs=generate_kwargs
            )
            print("✅ ASR model loaded (Whisper Small)")
        except Exception as e:
            print(f"⚠️  Failed to load small model, falling back to tiny: {e}")
            # Fallback to simpler model
            self.asr = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny.en",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ ASR model loaded (Whisper Tiny - fallback)")
    
    def init_output_files(self):
        """Initialize output files with headers"""
        # Main transcript file
        self.output_file = Path(self.output_dir) / "ALL_TRANSCRIPTS_COMBINED.txt"
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write("COMBINED TRANSCRIPTS - ALL AUDIO FILES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*120 + "\n\n")
            f.write("This file is updated in real-time as processing progresses.\n")
            f.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary file
        self.summary_file = Path(self.output_dir) / "PROGRESS_SUMMARY.txt"
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PROCESSING PROGRESS - UPDATING IN REAL-TIME\n")
            f.write("="*80 + "\n\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output file: {self.output_file}\n")
            f.write("="*80 + "\n\n")
        
        print(f"📄 Created output file: {self.output_file}")
        print(f"📊 Created progress file: {self.summary_file}")
    
    def append_to_output_file(self, data: dict, file_index: int):
        """Append transcript for a single file to the output file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FILE {file_index}: {data['file']}\n")
            f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            
            if data.get("status") == "success":
                file_duration = data.get('duration', 0)
                
                f.write(f"DURATION: {file_duration:.2f} seconds\n")
                f.write(f"SPEAKERS: {', '.join(data['speakers']) if data['speakers'] else 'None'}\n")
                f.write(f"SEGMENTS: {data['segment_count']}\n\n")
                
                # Timestamped segments
                for seg in data['segments']:
                    word_count = len(seg['transcript'].split())
                    f.write(f"[{seg['start']:6.2f}s → {seg['end']:6.2f}s] {seg['speaker']}: {seg['transcript']}\n")
                
                if data['full_transcripts']:
                    f.write("\nFULL SPEAKER TRANSCRIPTS:\n")
                    f.write("-"*40 + "\n")
                    
                    # Full transcripts per speaker
                    for speaker, text in data['full_transcripts'].items():
                        word_count = len(text.split())
                        f.write(f"\n{speaker} ({word_count} words):\n{text}\n")
                
            elif data.get("status") == "success_no_speech":
                f.write(f"⚠️  NO SPEECH DETECTED\n")
                f.write(f"DURATION: {data.get('duration', 0):.2f} seconds\n")
                
            else:
                f.write(f"❌ ERROR: {data.get('error', 'Unknown error')}\n")
            
            f.write(f"\n{'='*80}\n")
            f.flush()  # Force write to disk immediately
        
        # Also update the transcript file timestamp
        self.update_file_timestamp()
    
    def update_file_timestamp(self):
        """Update the timestamp at the top of the file"""
        with open(self.output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Update the last update line (line 7)
        if len(lines) > 7:
            lines[6] = f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    def update_progress_summary(self):
        """Update the progress summary file"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        files_processed = len(self.processed_files)
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PROCESSING PROGRESS - UPDATING IN REAL-TIME\n")
            f.write("="*80 + "\n\n")
            f.write(f"Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}\n")
            f.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Elapsed time: {elapsed_time:.0f} seconds\n")
            f.write(f"Output file: {self.output_file}\n")
            f.write("="*80 + "\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total files to process: {self.total_files}\n")
            f.write(f"Files processed: {files_processed}\n")
            f.write(f"Remaining files: {self.total_files - files_processed}\n")
            f.write(f"Progress: {files_processed/self.total_files*100:.1f}%\n\n")
            
            f.write("RESULTS SO FAR:\n")
            f.write("-"*40 + "\n")
            f.write(f"Successful (with speech): {self.successful_files}\n")
            f.write(f"Successful (no speech): {self.no_speech_files}\n")
            f.write(f"Failed: {self.failed_files}\n")
            f.write(f"Total duration processed: {self.total_duration:.2f} seconds\n")
            f.write(f"Total segments: {self.total_segments}\n")
            f.write(f"Total words: {self.total_words}\n\n")
            
            if self.successful_files > 0:
                f.write(f"Average words per file: {self.total_words/self.successful_files:.1f}\n")
                f.write(f"Average segments per file: {self.total_segments/self.successful_files:.1f}\n")
            
            # Show processing speed
            if elapsed_time > 0:
                files_per_second = files_processed / elapsed_time
                estimated_total_time = self.total_files / files_per_second if files_per_second > 0 else 0
                f.write(f"\nPROCESSING SPEED:\n")
                f.write(f"Files per second: {files_per_second:.3f}\n")
                f.write(f"Estimated total time: {estimated_total_time/60:.1f} minutes\n")
                f.write(f"Estimated time remaining: {(estimated_total_time - elapsed_time)/60:.1f} minutes\n")
            
            # Recent files
            f.write(f"\nLAST 10 PROCESSED FILES:\n")
            f.write("-"*40 + "\n")
            for file_data in self.processed_files[-10:]:
                status_icon = "✅" if file_data.get("status") == "success" else "⚠️" if file_data.get("status") == "success_no_speech" else "❌"
                f.write(f"{status_icon} {file_data['file']}\n")
    
    def convert_to_wav(self, input_path: str) -> str:
        """Convert audio file to mono 16 kHz WAV"""
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".wav":
            return input_path
        
        print(f"   🔄 Converting to WAV…", end=" ", flush=True)
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", "16000", "-vn",
            "-acodec", "pcm_s16le", tmp_wav_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("done")
            return tmp_wav_path
        except subprocess.CalledProcessError as e:
            os.unlink(tmp_wav_path)
            raise RuntimeError(f"ffmpeg failed: {e}")
    
    def extract_segments(self, diarization):
        """Extract segments from diarization output - handles DiarizeOutput dataclass"""
        segments = []
        try:
            # The diarization is a DiarizeOutput dataclass
            # We need to access the speaker_diarization attribute
            if hasattr(diarization, 'speaker_diarization'):
                diarization_result = diarization.speaker_diarization
                
                # Now iterate through the diarization result
                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    duration = turn.end - turn.start
                    
                    # Only include segments with reasonable duration
                    if duration >= self.min_segment_duration:
                        segments.append({
                            "speaker": speaker,
                            "start": round(turn.start, 2),
                            "end": round(turn.end, 2),
                            "duration": round(duration, 2)
                        })
                    else:
                        print(f"      ⚠️  Skipping short segment: {speaker} [{turn.start:.2f}s → {turn.end:.2f}s] ({duration:.2f}s)")
                
                return segments
            else:
                # Fallback for older versions
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    duration = turn.end - turn.start
                    if duration >= self.min_segment_duration:
                        segments.append({
                            "speaker": speaker,
                            "start": round(turn.start, 2),
                            "end": round(turn.end, 2),
                            "duration": round(duration, 2)
                        })
                return segments
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract segments: {str(e)}")
    
    def clean_transcript(self, text: str, duration: float) -> str:
        """Clean and validate transcript text"""
        if not text or text.strip() == "":
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Check for excessive word repetitions
        words = text.split()
        if len(words) > 0:
            # Count consecutive repetitions
            current_word = words[0].lower()
            current_count = 1
            for word in words[1:]:
                if word.lower() == current_word:
                    current_count += 1
                    if current_count > self.max_repetition_count:
                        # Found excessive repetition, clean it
                        text = " ".join(words[:words.index(word) - self.max_repetition_count + 1])
                        break
                else:
                    current_word = word.lower()
                    current_count = 1
        
        # Calculate words per second
        word_count = len(text.split())
        if duration > 0:
            words_per_second = word_count / duration
            
            # If words per second is too high, the transcript is likely invalid
            if words_per_second > self.max_words_per_second:
                # Try to trim to reasonable length
                max_words = int(duration * self.max_words_per_second)
                if max_words > 0:
                    words = text.split()[:max_words]
                    text = " ".join(words)
                    print(f"      ⚠️  Trimmed excessive transcript: {words_per_second:.1f} words/sec")
        
        # Remove common transcription artifacts
        artifacts = [
            "thank you for watching",
            "please like and subscribe",
            "[music]",
            "[applause]",
            "[laughter]",
            "um", "uh", "ah", "er",  # Fillers
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, "")
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def validate_transcript(self, transcript: str, duration: float, segment_start: float, segment_end: float) -> bool:
        """Validate if a transcript is reasonable for the given segment"""
        if not transcript or transcript.strip() == "":
            return False
        
        # Check word count
        word_count = len(transcript.split())
        if word_count < self.min_word_count:
            return False
        
        # Check words per second
        if duration > 0:
            words_per_second = word_count / duration
            if words_per_second > self.max_words_per_second:
                print(f"      ⚠️  Skipping: {words_per_second:.1f} words/sec is too high")
                return False
        
        # Check for obvious gibberish patterns
        patterns = [
            r'(.)\1{5,}',  # Single character repeated 5+ times
            r'(\w+)( \1){5,}',  # Word repeated 5+ times
        ]
        
        for pattern in patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                return False
        
        return True
    
    def process_audio(self, file_path: str) -> dict:
        """Process single audio file and return transcript data"""
        file_name = Path(file_path).name
        print(f"\n📁 Processing: {file_name}")
        
        try:
            # Convert to WAV
            wav_path = self.convert_to_wav(file_path)
            
            # Diarization
            print(f"   🎙️  Running diarization…", end=" ", flush=True)
            try:
                diarization = self.diarize_pipe(wav_path)
                print("done")
            except Exception as e:
                print(f"failed")
                raise RuntimeError(f"Diarization failed: {str(e)[:100]}")
            
            # Extract segments
            try:
                segments = self.extract_segments(diarization)
            except Exception as e:
                raise RuntimeError(f"Segment extraction failed: {str(e)[:100]}")
            
            if not segments:
                print(f"   ℹ️  No valid speech segments detected")
                return {
                    "file": file_name,
                    "duration": self.get_audio_duration(file_path),
                    "speakers": [],
                    "segment_count": 0,
                    "segments": [],
                    "full_transcripts": {},
                    "status": "success_no_speech"
                }
            
            speakers = sorted({seg['speaker'] for seg in segments})
            print(f"   ✅ Detected {len(speakers)} speakers: {', '.join(speakers)}")
            print(f"   📊 Found {len(segments)} valid segments")
            
            # Transcribe segments
            transcripts = []
            speaker_texts = defaultdict(list)
            skipped_count = 0
            
            if len(segments) > 0:
                print(f"   🔊 Transcribing segments…")
                for seg in tqdm(segments, desc="   ", unit="seg", leave=False, colour="green"):
                    start, end, duration = seg["start"], seg["end"], seg["duration"]
                    
                    # Skip very short segments (already filtered, but double-check)
                    if duration < self.min_segment_duration:
                        skipped_count += 1
                        continue
                    
                    tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_seg_path = tmp_seg.name
                    tmp_seg.close()
                    
                    try:
                        # Extract audio segment
                        subprocess.run([
                            "ffmpeg", "-y", "-i", wav_path,
                            "-ss", str(start), "-to", str(end),
                            "-ac", "1", "-ar", "16000", "-vn",
                            "-acodec", "pcm_s16le", tmp_seg_path
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # Transcribe
                        try:
                            result = self.asr(tmp_seg_path)
                            raw_transcript = result.get("text", "").strip()
                        except Exception as e:
                            print(f"      ❌ ASR error: {str(e)[:50]}")
                            raw_transcript = ""
                        
                        # Clean and validate transcript
                        clean_transcript = self.clean_transcript(raw_transcript, duration)
                        
                        if clean_transcript and self.validate_transcript(clean_transcript, duration, start, end):
                            output = {
                                "speaker": seg['speaker'],
                                "start": start,
                                "end": end,
                                "duration": duration,
                                "transcript": clean_transcript,
                                "word_count": len(clean_transcript.split())
                            }
                            transcripts.append(output)
                            speaker_texts[seg['speaker']].append(clean_transcript)
                            
                            # Print in real-time
                            word_count = len(clean_transcript.split())
                            wps = word_count / duration if duration > 0 else 0
                            print(f"      [{start:6.2f}s → {end:6.2f}s] {seg['speaker']}: {clean_transcript[:80]}{'...' if len(clean_transcript) > 80 else ''}")
                            if wps > 3:
                                print(f"         (words: {word_count}, {wps:.1f} words/sec)")
                        else:
                            skipped_count += 1
                            if raw_transcript:
                                print(f"      ⚠️  Skipped: {seg['speaker']} [{start:.2f}s → {end:.2f}s] - Invalid transcript")
                    
                    except Exception as e:
                        skipped_count += 1
                        print(f"      ❌ Error transcribing segment: {str(e)[:50]}")
                    
                    finally:
                        if os.path.exists(tmp_seg_path):
                            os.unlink(tmp_seg_path)
            
            # Cleanup temp WAV if converted
            if Path(file_path).suffix.lower() != ".wav" and os.path.exists(wav_path):
                os.unlink(wav_path)
            
            if skipped_count > 0:
                print(f"   ⚠️  Skipped {skipped_count} invalid segments")
            
            # Full speaker transcripts
            full_transcripts = {}
            for speaker, texts in speaker_texts.items():
                if texts:  # Only include speakers with valid transcripts
                    full_transcripts[speaker] = " ".join(texts)
            
            return {
                "file": file_name,
                "duration": self.get_audio_duration(file_path),
                "speakers": list(full_transcripts.keys()),  # Only speakers with valid transcripts
                "segment_count": len(transcripts),
                "segments": transcripts,
                "full_transcripts": full_transcripts,
                "status": "success" if transcripts else "success_no_speech"
            }
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"      ❌ Error: {error_msg}")
            return {
                "file": file_name,
                "status": "error",
                "error": error_msg
            }
    
    def get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True, timeout=10
            )
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def process_folder(self, folder_path: str, test_limit: int = 0):
        """Process all audio files in folder"""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
        audio_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"❌ No audio files found in: {folder_path}")
            return
        
        # Sort files by name for consistent processing
        audio_files.sort(key=lambda x: x.name)
        
        # Apply test limit if specified
        if test_limit > 0:
            audio_files = audio_files[:test_limit]
            print(f"\n🔬 TEST MODE: Processing first {test_limit} files")
        else:
            print(f"\nProcessing ALL {len(audio_files)} files")
        
        self.total_files = len(audio_files)
        self.start_time = time.time()
        
        print(f"\n{'='*100}")
        print(f"🚀 PyAnnote + Whisper Transcription")
        print(f"📁 Input folder: {folder_path}")
        print(f"📊 Files to process: {len(audio_files)}")
        print(f"💾 Output: {self.output_dir}")
        print(f"📄 Transcript file: {self.output_file}")
        print(f"📊 Progress file: {self.summary_file}")
        print(f"⚙️  Settings:")
        print(f"   - Min segment duration: {self.min_segment_duration}s")
        print(f"   - Max words per second: {self.max_words_per_second}")
        print(f"   - Min words per segment: {self.min_word_count}")
        print(f"{'='*100}")
        
        print(f"\n📄 Output file created! You can open it now to see real-time updates:")
        print(f"   {self.output_file}")
        print(f"\n📊 Progress tracking available at:")
        print(f"   {self.summary_file}")
        print(f"\nStarting processing...")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]", end="")
            data = self.process_audio(str(audio_file))
            self.processed_files.append(data)
            
            # Update statistics
            if data.get("status") == "success":
                self.successful_files += 1
                self.total_duration += data.get('duration', 0)
                self.total_segments += data.get('segment_count', 0)
                for seg in data.get('segments', []):
                    self.total_words += len(seg['transcript'].split())
            elif data.get("status") == "success_no_speech":
                self.no_speech_files += 1
            else:
                self.failed_files += 1
            
            # Save to output file immediately
            self.append_to_output_file(data, i)
            
            # Update progress summary
            self.update_progress_summary()
            
            # Show progress update
            print(f"   📊 Progress: {i}/{len(audio_files)} ({i/len(audio_files)*100:.1f}%)")
            print(f"   ✅ Success: {self.successful_files} | ⚠️ No Speech: {self.no_speech_files} | ❌ Failed: {self.failed_files}")
        
        # Write final summary
        self.write_final_summary()
        
        print(f"\n{'='*100}")
        print(f"✅ Processing complete!")
        print(f"   Total processed: {len(audio_files)}")
        print(f"   Successful (with speech): {self.successful_files}")
        print(f"   Successful (no speech): {self.no_speech_files}")
        print(f"   Failed: {self.failed_files}")
        print(f"   Total words: {self.total_words}")
        print(f"   Output file: {self.output_file}")
        print(f"   Final summary: {self.summary_file}")
        print(f"{'='*100}\n")
    
    def write_final_summary(self):
        """Write final summary to the output file"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write("\n\n" + "="*120 + "\n")
            f.write("FINAL PROCESSING SUMMARY\n")
            f.write("="*120 + "\n\n")
            f.write(f"Total files processed: {self.total_files}\n")
            f.write(f"Successful (with speech): {self.successful_files}\n")
            f.write(f"Successful (no speech): {self.no_speech_files}\n")
            f.write(f"Failed files: {self.failed_files}\n")
            f.write(f"Total duration: {self.total_duration:.2f} seconds ({self.total_duration/3600:.2f} hours)\n")
            f.write(f"Total segments: {self.total_segments}\n")
            f.write(f"Total words: {self.total_words}\n")
            if self.successful_files > 0:
                f.write(f"Average words per file: {self.total_words/self.successful_files:.1f}\n")
                f.write(f"Average segments per file: {self.total_segments/self.successful_files:.1f}\n")
            f.write(f"Processing time: {elapsed_time:.0f} seconds ({elapsed_time/60:.1f} minutes)\n")
            f.write(f"Files per second: {self.total_files/elapsed_time:.3f}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process audio files with speaker diarization + transcription"
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing audio files"
    )
    parser.add_argument(
        "--hf_token",
        help="Hugging Face token (default: from HF_TOKEN env var)",
        default=os.getenv("HF_TOKEN")
    )
    parser.add_argument(
        "--output",
        help="Output directory (default: output_transcripts)",
        default="output_transcripts"
    )
    parser.add_argument(
        "--test",
        help="Test mode: process only first N files (0 = all files)",
        type=int,
        default=0
    )
    parser.add_argument(
        "--min_duration",
        help="Minimum segment duration in seconds (default: 0.5)",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--max_wps",
        help="Maximum words per second (default: 4.0)",
        type=float,
        default=4.0
    )
    args = parser.parse_args()
    
    if not args.hf_token:
        print("❌ Error: Hugging Face token not provided.\n"
              "   • Set HF_TOKEN env var: export HF_TOKEN='your_token'\n"
              "   • Or pass: --hf_token YOUR_TOKEN")
        sys.exit(1)
    
    try:
        processor = AudioProcessor(args.hf_token, args.output)
        # Apply command line parameters
        processor.min_segment_duration = args.min_duration
        processor.max_words_per_second = args.max_wps
        processor.process_folder(args.folder, args.test)
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        # Write what we have
        if hasattr(processor, 'processed_files') and processor.processed_files:
            print(f"\n💾 Saving progress before exit...")
            processor.write_final_summary()
            print(f"   Partial results saved to: {processor.output_file}")
        sys.exit(0)


if __name__ == "__main__":
    main()




