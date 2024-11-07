import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import audio_metadata
import librosa
from librosa.feature import rhythm
import pygame
from pygame import mixer_music
from io import BytesIO
import numpy as np
from time import perf_counter as time
from line_profiler import LineProfiler
import numpy as np
from functools import lru_cache

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
    cover: Optional[pygame.Surface] = None
    
    def get_display_name(self) -> str:
        artist = self.artist or "Unknown Artist"
        title = self.title or "Unknown Title"
        return f"{artist} - {title}"

@dataclass
class SongMetadata:
    """Separate Klasse für die zu cachenden Metadaten"""
    path: str
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    genre: Optional[str]
    year: Optional[str]
    length: float
    bpm: float

class AudioVisualizer:
    def __init__(self, song: Song, window: pygame.Surface):
        self.song = song
        self.window = window
        
        # Audio laden
        self.y, self.sr = librosa.load(song.path, sr=None)
        
        # Spectrum setup
        self.spectrum_height = window.get_height() // 5
        self.spectrum_surface = pygame.Surface((window.get_width(), self.spectrum_height), pygame.SRCALPHA)
        
        # Volume visualizer setup
        self.volume_height = window.get_height() // 8
        self.volume_surface = pygame.Surface((window.get_width(), self.volume_height))
        self.volume_rect = pygame.Rect(0, window.get_height() - self.volume_height, 
                                     window.get_width(), self.volume_height)
        
        # FFT Parameter
        self.setup_spectrum_analyzer()
        self.setup_volume_visualizer()
        
        self.mouse_down = False

    def setup_spectrum_analyzer(self, min_freq=10, max_freq=16000, num_bands=300):
        self.D = librosa.stft(self.y, n_fft=10000, hop_length=512)
        self.S = np.abs(self.D)
        t1 = time()
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=10000)
        self.freq_bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands)

    def setup_spectrum_analyzer(self, min_freq=10, max_freq=16000, num_bands=300):
        self.n_fft = 10000
        self.hop_length = 512
        self.num_bands = num_bands
        self.freq_bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands)
        
        self.D = librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S = np.abs(self.D)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.times = librosa.times_like(self.S, sr=self.sr, hop_length=self.hop_length)

    def setup_volume_visualizer(self, num_samples=401):
        segment_size = len(self.y) // num_samples
        samples = []
        
        for i in range(num_samples):
            start = i * segment_size
            end = start + segment_size
            segment = self.y[start:end]
            amplitude = np.mean(np.abs(segment))
            samples.append(amplitude)
            
        self.volume_samples = (np.array(samples) / np.max(samples)).astype(float)
        self.num_volume_samples = num_samples
        
    @lru_cache(maxsize=None)
    def get_current_spectrum(self, time_ms: int) -> np.ndarray:
        """Optimierte Version mit vectorisierten Operationen"""
        frame_idx = int((time_ms / 1000.0) * self.sr / self.hop_length)
        frame_idx = min(frame_idx, self.S.shape[1] - 1)
        
        # Hole das aktuelle Spektrum
        current_spectrum = self.S[:, frame_idx]
        
        # Vektorisierte Berechnung der Bänder
        # Vorberechne die Indizes für jedes Frequenzband
        band_magnitudes = np.zeros(self.num_bands)
        
        # Vectorisierte Berechnung aller Bänder auf einmal
        for i in range(self.num_bands - 1):
            mask = (self.freqs >= self.freq_bands[i]) & (self.freqs < self.freq_bands[i + 1])
            if mask.any():  # Diese Prüfung ist billiger als mean auf leeren Arrays
                band_magnitudes[i] = np.mean(current_spectrum[mask])
        
        # Letzte Band separat
        mask = (self.freqs >= self.freq_bands[-1])
        if mask.any():
            band_magnitudes[-1] = np.mean(current_spectrum[mask])
        
        # Logarithmische Skalierung und Normalisierung
        band_magnitudes = np.log10(band_magnitudes + 1)
        max_val = np.max(band_magnitudes)
        if max_val > 0:  # Verhindere Division durch 0
            return band_magnitudes / max_val
        return band_magnitudes

    def draw_spectrum(self, time_pos: float):
        self.spectrum_surface.fill('grey15')
        bar_width = self.spectrum_surface.get_width() / self.num_bands
        
        for i, magnitude in enumerate(self.get_current_spectrum(time_pos)):
            height = int(magnitude * self.spectrum_surface.get_height())
            x = i * bar_width
            rect = pygame.Rect(x, self.spectrum_surface.get_height() - height, 2, height)
            
            hue = i / self.num_bands * 255
            brightness = int(magnitude * 100)
            color = pygame.Color(0)
            color.hsva = (hue, 100, brightness, 100)
            
            pygame.draw.rect(self.spectrum_surface, color, rect)
            
        self.window.blit(pygame.transform.flip(self.spectrum_surface, False, True), (0, 0))
    
    def draw_volume(self, time_pos: float):
        self.volume_surface.fill('grey15')
        width, height = self.volume_surface.get_size()
        
        x_coords = np.linspace(0, width, self.num_volume_samples)
        progress = time_pos / (len(self.y) / self.sr)
        colors = ['orange' if x/self.num_volume_samples < progress else 'grey30' 
                 for x in range(self.num_volume_samples)]
        
        rect_width = width // self.num_volume_samples + 1
        for x_pos, h, color in zip(x_coords, self.volume_samples * height, colors):
            rect = pygame.Rect(x_pos, 0, rect_width, h)
            pygame.draw.rect(self.volume_surface, color, rect)
        
        ypos = self.window.get_height() - height
        self.window.blit(pygame.transform.flip(self.volume_surface, False, True), (0, ypos))
    
    def get_time_from_click(self, x_pos: int) -> float:
        """Konvertiert eine x-Position zu einer Songposition mit Begrenzung."""
        song_length = len(self.y) / self.sr
        position = x_pos / self.window.get_width() * song_length
        return max(0, min(position, song_length - 1))  # Kleine Sicherheitsmarge

    def handle_volume_click(self, event: pygame.event.Event) -> Optional[float]:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.volume_rect.collidepoint(event.pos):
                self.mouse_down = True
                x = min(max(0, event.pos[0]), self.window.get_width())
                return self.get_time_from_click(x)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False
        elif event.type == pygame.MOUSEMOTION and self.mouse_down:
            x = min(max(0, event.pos[0]), self.window.get_width())
            return self.get_time_from_click(x)
        return None

class PlaybackManager:
    def __init__(self, song_length: float):
        self.start_time = time()
        self.current_time = 0.0
        self.is_paused = False
        self.pause_start = 0.0
        self.total_pause_time = 0.0
        self.manual_seek = 0.0
        self.song_length = song_length  # Speichern der Songlänge
    
    def seek(self, position: float):
        """
        Springt zu einer bestimmten Position im Song.
        Stellt sicher, dass die Position im gültigen Bereich liegt.
        """
        # Position auf den gültigen Bereich beschränken
        position = max(0, min(position, self.song_length))
        
        self.manual_seek = position
        self.start_time = time()
        self.total_pause_time = 0.0
        if self.is_paused:
            self.pause_start = self.start_time
        mixer_music.set_pos(position)
    
    def play(self):
        """Startet die Wiedergabe."""
        if self.is_paused:
            self.total_pause_time += time() - self.pause_start
            self.is_paused = False
            mixer_music.unpause()
    
    def pause(self):
        """Pausiert die Wiedergabe."""
        if not self.is_paused:
            self.pause_start = time()
            self.is_paused = True
            mixer_music.pause()
    
    def reset(self):
        """Setzt den Manager für einen neuen Song zurück."""
        self.start_time = time()
        self.current_time = 0.0
        self.is_paused = False
        self.pause_start = 0.0
        self.total_pause_time = 0.0
        self.manual_seek = 0.0
    
    def get_time(self) -> float:
        """
        Gibt die aktuelle Position im Song zurück.
        
        Returns:
            float: Aktuelle Position in Sekunden
        """
        if self.is_paused:
            return self.current_time
        
        self.current_time = time() - self.start_time - self.total_pause_time + self.manual_seek
        return self.current_time

class MusicPlayer:
    def __init__(self, music_dir: str = "musik"):
        pygame.init()
        pygame.mixer.init()
        
        self.width, self.height = 1200, 800
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Music Visualizer")
        
        self.music_dir = music_dir
        self.songs: List[Song] = []
        self.current_song_index = 0
        self.current_visualizer = None
        self.playback: PlaybackManager = None
        
        self.load_library()
        self.setup_current_song()
        
    def load_library(self):
        """Lädt die Musikbibliothek, nutzt Cache wenn möglich."""
        cache_path = "music_cache.json"
        metadata_cache = self.load_cache(cache_path)
        
        # Liste aller Audiodateien im Verzeichnis
        audio_files = [f for f in os.listdir(self.music_dir) 
                      if os.path.isfile(os.path.join(self.music_dir, f))]
        
        needs_cache_update = False
        self.songs = []

        for file_name in audio_files:
            file_path = os.path.join(self.music_dir, file_name)
            
            # Prüfe ob die Datei im Cache ist und sich nicht geändert hat
            file_mtime = os.path.getmtime(file_path)
            cache_entry = metadata_cache.get(file_path)
            
            if cache_entry and cache_entry.get('mtime') == file_mtime:
                # Cache hit - lade aus Cache
                metadata = SongMetadata(**{k: v for k, v in cache_entry.items() 
                                        if k != 'mtime'})
            else:
                # Cache miss - lade und analysiere Datei
                print(f"Analyzing {file_name}...")
                try:
                    metadata = self.analyze_file(file_path)
                    metadata_cache[file_path] = {**asdict(metadata), 'mtime': file_mtime}
                    needs_cache_update = True
                except Exception as e:
                    print(f"Error analyzing {file_name}: {e}")
                    continue

            # Cover laden
            try:
                cover = self.load_cover_from_file(file_path)
            except Exception:
                cover = None

            # Song-Objekt erstellen
            song = Song(
                path=metadata.path,
                title=metadata.title,
                artist=metadata.artist,
                album=metadata.album,
                genre=metadata.genre,
                year=metadata.year,
                length=metadata.length,
                bpm=metadata.bpm,
                cover=cover
            )
            self.songs.append(song)

        # Cache aktualisieren wenn nötig
        if needs_cache_update:
            self.save_cache(metadata_cache, cache_path)

    def setup_current_song(self):
        song = self.songs[self.current_song_index]
        mixer_music.load(song.path)
        self.current_visualizer = AudioVisualizer(song, self.window)
        # Songlänge an PlaybackManager übergeben
        self.playback = PlaybackManager(song.length)
        mixer_music.play()
    
    def change_song(self, direction: int):
        """Wechselt zum nächsten/vorherigen Song."""
        mixer_music.stop()
        self.current_song_index = (self.current_song_index + direction) % len(self.songs)
        self.setup_current_song()

    def load_cache(self, cache_path: str) -> Dict:
        """Lädt den Metadaten-Cache aus einer JSON-Datei."""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self, cache: Dict, cache_path: str):
        """Speichert den Metadaten-Cache in eine JSON-Datei."""
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)

    def analyze_file(self, file_path: str) -> SongMetadata:
        """Analysiert eine Audiodatei und extrahiert Metadaten."""
        metadata = audio_metadata.load(file_path)
        y, sr = librosa.load(file_path, sr=None)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        return SongMetadata(
            path=file_path,
            title=metadata.tags.title[0] if hasattr(metadata.tags, 'title') else None,
            artist=metadata.tags.artist[0] if hasattr(metadata.tags, 'artist') else None,
            album=metadata.tags.album[0] if hasattr(metadata.tags, 'album') else None,
            genre=metadata.tags.genre[0] if hasattr(metadata.tags, 'genre') else None,
            year=metadata.tags.date[0] if hasattr(metadata.tags, 'date') else None,
            length=metadata.streaminfo.duration,
            bpm=tempo
        )

    def load_cover_from_file(self, file_path: str) -> Optional[pygame.Surface]:
        """Lädt das Cover aus einer Audiodatei."""
        metadata = audio_metadata.load(file_path)
        if hasattr(metadata, 'pictures') and metadata.pictures:
            image_stream = BytesIO(metadata.pictures[0].data)
            return pygame.image.load(image_stream)
        return None

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if mixer_music.get_busy():
                            self.playback.pause()
                        else:
                            self.playback.play()
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self.change_song(1 if event.key == pygame.K_RIGHT else -1)
                
                # Handle volume visualizer clicks
                if (new_pos := self.current_visualizer.handle_volume_click(event)):
                    self.playback.seek(new_pos)
            
            # Prüfe ob das Lied zu Ende ist und wechsle zum nächsten
            if not mixer_music.get_busy() and not self.playback.is_paused:
                current_time = self.playback.get_time()
                if current_time >= self.songs[self.current_song_index].length - 0.1:
                    self.change_song(1)
                    continue
            
            self.window.fill('grey15')
            current_time = self.playback.get_time()
            
            # Draw visualizations
            song = self.songs[self.current_song_index]
            if song.cover:
                # Scale and draw cover
                scaled_cover = pygame.transform.scale(song.cover, 
                    (self.width // 2, int(self.height * 0.6)))
                cover_rect = scaled_cover.get_rect(center=(self.width // 2, self.height // 2))
                self.window.blit(scaled_cover, cover_rect)
            
            # Draw song info
            font = pygame.font.Font(None, 48)
            title_text = font.render(song.get_display_name(), True, (255, 255, 255))
            self.window.blit(title_text, (20, 20))
            
            # Draw time info
            time_text = font.render(f"{int(current_time // 60)}:{int(current_time % 60):02d}", True, (255, 255, 255))
            self.window.blit(time_text, (20, 70))
            
            # Draw visualizers
            self.current_visualizer.draw_spectrum(current_time)
            self.current_visualizer.draw_volume(current_time)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    player = MusicPlayer()
    player.run()