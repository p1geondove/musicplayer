import os
import librosa
from librosa.feature import rhythm
from dataclasses import dataclass
from datetime import datetime
from mutagen import File
from mutagen.easyid3 import EasyID3
from typing import Optional, List

@dataclass
class Song:
    path: str
    cover: Optional[str]
    length: float
    bpm: float
    release: Optional[str] = None
    album: Optional[str] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    genre: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class SongCrawler:
    AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg'}
    IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
    
    def __init__(self, main_dir: Optional[str] = None):
        self.main_dir = main_dir or os.path.join(os.getcwd(), 'musik')
        self.music: List[Song] = []
        self.crawl()
    
    def _get_metadata(self, file_path: str) -> dict:
        """Extrahiert Metadaten aus der Audiodatei."""
        metadata = {}
        try:
            audio = File(file_path)
            print(audio.info.pprint())
            
            # Versuche zuerst EasyID3 für MP3s
            if file_path.lower().endswith('.mp3'):
                try:
                    id3 = EasyID3(file_path)
                    print(EasyID3.valid_keys.keys())
                    metadata.update({
                        'title': id3.get('title', [None])[0],
                        'artist': id3.get('artist', [None])[0],
                        'album': id3.get('album', [None])[0],
                        'date': id3.get('date', [None])[0],
                        'genre': id3.get('genre', [None])[0]
                    })
                except:
                    pass
            
            # Für andere Formate oder wenn ID3 fehlschlägt
            if not metadata and hasattr(audio, 'tags'):
                tags = audio.tags
                if tags:
                    metadata.update({
                        'title': tags.get('title', [None])[0] if 'title' in tags else None,
                        'artist': tags.get('artist', [None])[0] if 'artist' in tags else None,
                        'album': tags.get('album', [None])[0] if 'album' in tags else None,
                        'date': tags.get('date', [None])[0] if 'date' in tags else None,
                        'genre': tags.get('genre', [None])[0] if 'genre' in tags else None
                    })
                    
        except Exception as e:
            print(f"Warnung: Konnte Metadaten nicht lesen für {file_path}: {e}")
        
        return metadata

    def _find_cover_in_directory(self, directory: str) -> Optional[str]:
        """Sucht nach einem Coverbild im angegebenen Verzeichnis."""
        common_cover_names = {'cover', 'folder', 'album', 'front'}
        
        # Alle Bilddateien im Verzeichnis sammeln
        image_files = []
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                image_files.append(file)
        
        # Präferenz für Dateien mit typischen Cover-Namen
        for name in common_cover_names:
            for img in image_files:
                if name in img.lower():
                    return os.path.join(directory, img)
        
        # Wenn kein spezifisches Cover gefunden wurde, nimm das erste Bild
        return os.path.join(directory, image_files[0]) if image_files else None

    def crawl(self):
        """Durchsucht das Musikverzeichnis nach Audiodateien."""
        for root, _, files in os.walk(self.main_dir):
            audio_files = [f for f in files if any(f.lower().endswith(ext) for ext in self.AUDIO_EXTENSIONS)]
            
            for audio_file in audio_files:
                file_path = os.path.join(root, audio_file)
                try:
                    # Audio-Analyse
                    y, sr = librosa.load(file_path, sr=None)
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                    tempo = rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
                    length = len(y) / sr
                    
                    # Metadaten
                    metadata = self._get_metadata(file_path)
                    
                    # Cover suchen
                    cover = self._find_cover_in_directory(root)
                    
                    # Song erstellen
                    song = Song(
                        path=file_path,
                        cover=cover,
                        length=length,
                        bpm=tempo,
                        release=metadata.get('date'),
                        album=metadata.get('album'),
                        artist=metadata.get('artist'),
                        title=metadata.get('title'),
                        genre=metadata.get('genre')
                    )
                    
                    self.music.append(song)
                    print(f"Erfolgreich geladen: {os.path.basename(file_path)}")
                    
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    def get_songs_by_artist(self, artist: str) -> List[Song]:
        """Filtert Songs nach Künstler."""
        return [song for song in self.music if song.artist and artist.lower() in song.artist.lower()]

    def get_songs_by_genre(self, genre: str) -> List[Song]:
        """Filtert Songs nach Genre."""
        return [song for song in self.music if song.genre and genre.lower() in song.genre.lower()]

    def get_songs_by_bpm_range(self, min_bpm: float, max_bpm: float) -> List[Song]:
        """Filtert Songs nach BPM-Bereich."""
        return [song for song in self.music if min_bpm <= song.bpm <= max_bpm]

if __name__ == "__main__":
    crawler = SongCrawler()
    
    # Beispielausgabe
    for song in crawler.music:
        print("\nGefundener Song:")
        print(f"Titel: {song.title or 'Unbekannt'}")
        print(f"Künstler: {song.artist or 'Unbekannt'}")
        print(f"Album: {song.album or 'Unbekannt'}")
        print(f"Genre: {song.genre or 'Unbekannt'}")
        print(f"BPM: {song.bpm:.1f}")
        print(f"Länge: {int(song.length // 60)}:{int(song.length % 60):02d}")
        print(f"Cover: {'Gefunden' if song.cover else 'Nicht gefunden'}")
        print("-" * 50)