# -*- coding: utf-8 -*-
"""
config/config.py

Central configuration for all Python scripts.
Mirrors config/config.js — single source of truth for the entire project.

Usage in Python scripts:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
    from config import BASE_DATA_DIR, VALID_COMPANIES, get_company_paths, TRANSCRIPTION, PREPROCESSING
"""

import os
from pathlib import Path

# Load .env if python-dotenv is available (optional — falls back to os.environ)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass  # python-dotenv not installed — use environment variables directly

# Company definitions
VALID_COMPANIES = ['medicare', 'mva', 'vodafone']

# Base data path
BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR") or Path(__file__).parent.parent / "Data"


def get_company_paths(company: str) -> dict:
    """
    Returns all relevant directory/file paths for a given company.
    Mirrors getCompanyPaths() in config/config.js.
    """
    if company not in VALID_COMPANIES:
        raise ValueError(f'Invalid company: "{company}". Must be one of: {", ".join(VALID_COMPANIES)}')

    base = Path(BASE_DATA_DIR)
    return {
        'audio_dir':      base / f'{company}_rec',
        'transcript_dir': base / f'{company}_rec' / 'transcripts',
        'txt_output_dir': base / 'txt_file',
        'questions_file': base / 'txt_file' / f'questions_{company}.txt',
        'summary_file':   base / 'txt_file' / f'summary_{company}.txt',
    }


# Transcription (transcription/fast_whisper3.py)
TRANSCRIPTION = {
    'num_speakers':          int(os.getenv('TRANSCRIPTION_NUM_SPEAKERS', 2)),
    'language':              os.getenv('TRANSCRIPTION_LANGUAGE', 'English'),
    'model_size':            os.getenv('TRANSCRIPTION_MODEL_SIZE', 'medium'),  # tiny|base|small|medium|large
    'supported_extensions':  {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.wma'},
    'output_file_name':      'all_transcripts.txt',
}

# Preprocessing (preprocessing/filter_questions.py)
PREPROCESSING = {
    'human_label':    '_human',
    'machine_label':  '_machine',
    'min_text_length': 3,
    'output_dir':     str(Path(BASE_DATA_DIR) / 'txt_file'),
}