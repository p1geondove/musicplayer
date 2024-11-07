import audio_metadata
import librosa
from librosa.feature import rhythm
import numpy as np
from ..models.song import SongMetadata

def analyze_audio(file_path: str) -> tuple[float, float]:
    """Extrahiert BPM und Länge aus einer Audiodatei"""
    y, sr = librosa.load(file_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
    duration = len(y) / sr
    return duration, tempo

def load_metadata(file_path: str) -> SongMetadata:
    """Lädt Metadaten aus einer Audiodatei"""
    metadata = audio_metadata.load(file_path)
    duration, tempo = analyze_audio(file_path)
    
    return SongMetadata(
        path=file_path,
        title=metadata.tags.title[0] if hasattr(metadata.tags, 'title') else None,
        artist=metadata.tags.artist[0] if hasattr(metadata.tags, 'artist') else None,
        album=metadata.tags.album[0] if hasattr(metadata.tags, 'album') else None,
        genre=metadata.tags.genre[0] if hasattr(metadata.tags, 'genre') else None,
        year=metadata.tags.date[0] if hasattr(metadata.tags, 'date') else None,
        length=duration,
        bpm=tempo
    )