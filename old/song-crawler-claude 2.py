import os
from dataclasses import dataclass
from typing import Optional, List
import audio_metadata
import librosa
from librosa.feature import rhythm
import pygame
from io import BytesIO

@dataclass
class Song:
    path: str
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    genre: Optional[str]
    year: Optional[str]
    length: float
    bpm: float
    cover: Optional[pygame.Surface] = None  # Jetzt speichern wir die pygame Surface

class MusicLibrary:
    def __init__(self, music_dir: str = "musik"):
        self.music_dir = music_dir
        self.songs: List[Song] = []
        # Pygame initialisieren für Bildverarbeitung
        pygame.init()
        self.scan_directory()
    
    def calculate_bpm(self, file_path: str) -> float:
        """Berechnet BPM mit librosa."""
        y, sr = librosa.load(file_path, sr=None)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
        return tempo

    def extract_cover(self, metadata) -> Optional[pygame.Surface]:
        """Extrahiert das Cover-Bild aus den Metadaten und konvertiert es in eine Pygame Surface."""
        try:
            if hasattr(metadata, 'pictures') and metadata.pictures:
                # Nimm das erste Bild aus den Metadaten
                picture = metadata.pictures[0]
                # Bild-Bytes in einen BytesIO Stream laden
                image_stream = BytesIO(picture.data)
                # BytesIO in Pygame Surface umwandeln
                image_surface = pygame.image.load(image_stream)
                return image_surface
        except Exception as e:
            print(f"Fehler beim Laden des Covers: {e}")
        return None

    def scan_directory(self):
        """Scannt den Musikordner nach Audiodateien."""
        if not os.path.exists(self.music_dir):
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {self.music_dir}")

        for file_name in os.listdir(self.music_dir):
            file_path = os.path.join(self.music_dir, file_name)
            
            if not os.path.isfile(file_path):
                continue
                
            try:
                # Metadaten laden
                metadata = audio_metadata.load(file_path)
                
                # BPM berechnen
                bpm = self.calculate_bpm(file_path)
                
                # Cover extrahieren
                cover_surface = self.extract_cover(metadata)
                
                # Song-Objekt erstellen
                song = Song(
                    path=file_path,
                    title=metadata.tags.title[0] if hasattr(metadata.tags, 'title') else None,
                    artist=metadata.tags.artist[0] if hasattr(metadata.tags, 'artist') else None,
                    album=metadata.tags.album[0] if hasattr(metadata.tags, 'album') else None,
                    genre=metadata.tags.genre[0] if hasattr(metadata.tags, 'genre') else None,
                    year=metadata.tags.date[0] if hasattr(metadata.tags, 'date') else None,
                    length=metadata.streaminfo.duration,
                    bpm=bpm,
                    cover=cover_surface
                )
                
                self.songs.append(song)
                print(f"Erfolgreich geladen: {file_name}")
                
            except Exception as e:
                print(f"Fehler beim Laden von {file_name}: {e}")

    def print_library(self):
        """Gibt alle Songs mit ihren Details aus."""
        for song in self.songs:
            print("\n" + "="*50)
            print(f"Titel: {song.title or 'Unbekannt'}")
            print(f"Künstler: {song.artist or 'Unbekannt'}")
            print(f"Album: {song.album or 'Unbekannt'}")
            print(f"Genre: {song.genre or 'Unbekannt'}")
            print(f"Jahr: {song.year or 'Unbekannt'}")
            print(f"Länge: {int(song.length // 60)}:{int(song.length % 60):02d}")
            print(f"BPM: {song.bpm:.1f}")
            if song.cover:
                print(f"Cover: {song.cover.get_size()}")
            else:
                print("Cover: Nicht gefunden")

    def display_covers(self):
        """Zeigt alle gefundenen Cover als Beispiel an."""
        pygame.display.set_mode((800, 600))
        display = pygame.display.get_surface()
        
        running = True
        current_song_index = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        current_song_index = (current_song_index + 1) % len(self.songs)
                    elif event.key == pygame.K_LEFT:
                        current_song_index = (current_song_index - 1) % len(self.songs)
            
            display.fill((0, 0, 0))
            
            song = self.songs[current_song_index]
            if song.cover:
                # Cover in der Mitte des Fensters anzeigen
                song.cover = pygame.transform.scale(song.cover, display.get_size())
                cover_rect = song.cover.get_rect()
                cover_rect.center = display.get_rect().center
                display.blit(song.cover, cover_rect)
                
                # Titel anzeigen
                font = pygame.font.Font(None, 36)
                title_text = font.render(f"{song.artist} - {song.title}", True, (255, 255, 255))
                display.blit(title_text, (10, 10))
            
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    try:
        library = MusicLibrary()
        print(f"\nGefundene Songs: {len(library.songs)}")
        library.print_library()
        
        # Zeige die Cover in einem Pygame-Fenster
        library.display_covers()
    except Exception as e:
        print(f"Fehler: {e}")