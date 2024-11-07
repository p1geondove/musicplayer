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
from functools import lru_cache
import itertools

WIDTH = 1200
HEIGHT = 800

@dataclass
class Song:
    path: str
    title: str
    artist: str
    album: str
    genre: str
    year: str
    length: float
    bpm: float
    cover: Optional[pygame.Surface] = None
    
    def get_display_name(self) -> str:
        artist = self.artist or "Unknown Artist"
        title = self.title or "Unknown Title"
        return f"{artist} - {title}"

    def get_str(self) -> list[str]:
        _min, _sec = divmod(self.length,60)
        return [
            self.title,
            self.artist,
            self.album,
            self.genre,
            self.year,
            f'{int(_min):2d}:{int(_sec):2d}',
            f'{self.bpm:.1f}'
        ]

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
        # Erstelle Oberflächen für beide Visualisierungen
        self.surface_spectrum_width = window.get_width()
        self.surface_spectrum_height = window.get_height()//5
        self.surface_volume_width = window.get_width()
        self.surface_volume_height = window.get_height()//10
        self.surface_spectrum = pygame.Surface((self.surface_spectrum_width, self.surface_spectrum_height), pygame.SRCALPHA)
        self.surface_volume = pygame.Surface((self.surface_volume_width, window.get_height()//10), pygame.SRCALPHA)
        
        # Audio laden
        self.y, self.sr = librosa.load(song.path, sr=None)
        
        # FFT Setup mit optimierten Parametern
        self.n_fft = 4096  # Optimierte FFT-Größe
        self.hop_length = 512
        
        # Spektrogramm für die ersten 30 Sekunden berechnen
        chunk_duration = 30  # Sekunden
        chunk_samples = int(chunk_duration * self.sr)
        y_chunk = self.y[:chunk_samples] if len(self.y) > chunk_samples else self.y
        self.D = librosa.stft(y_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S = np.abs(self.D)
        
        # Frequenzbänder
        self.num_bands = 1000
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.freq_bands = np.logspace(np.log10(10), np.log10(16000), self.num_bands)
        
        # Cache für die Spektren
        self._spectrum_cache = {}
        
        # Volume Visualizer Setup
        self.setup_volume_visualizer()
        
        # Maus-Interaktion
        self.mouse_down = False
        self.volume_rect = pygame.Rect(0, window.get_height() - self.surface_volume.get_height(),
                                     self.surface_volume_width, self.surface_volume.get_height())
    
    def setup_volume_visualizer(self, num_samples: int = 1201):
        """Initialisiert die Volumen-Visualisierung"""
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
    def get_current_spectrum(self, time_pos: float) -> np.ndarray:
        """Optimierte und gecachte Spektrum-Berechnung"""

        minmax = lambda x : int(min(max(0,x),len(self.y)))
        pos = time_pos*self.sr
        width = self.sr*5 #5 sekunde zu jeder seite = 10sekunden - alles was geladen ist im lru
        # start, stop = minmax(pos-width), minmax(pos+width)
        start, stop = 0, minmax(pos+width)

        y_chunk = self.y[start:stop] if len(self.y) > stop-start else self.y
        self.D = librosa.stft(y_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S = np.abs(self.D)

        frame_idx = int(time_pos * self.sr / self.hop_length)
        frame_idx = min(frame_idx, self.S.shape[1] - 1)
        current_spectrum = self.S[:, frame_idx]
        band_magnitudes = np.zeros(self.num_bands)
        
        for i in range(self.num_bands):
            if i < self.num_bands - 1:
                freq_mask = (self.freqs >= self.freq_bands[i]) & (self.freqs < self.freq_bands[i + 1])
            else:
                freq_mask = (self.freqs >= self.freq_bands[i])
            
            if np.any(freq_mask):
                band_magnitudes[i] = np.mean(current_spectrum[freq_mask])
        
        band_magnitudes = np.log10(band_magnitudes + 1)
        return band_magnitudes / np.max(band_magnitudes)
    
    def draw_spectrum(self, time_pos: float):
        """Zeichnet das Frequenzspektrum"""
        spectrum = self.get_current_spectrum(time_pos)
        positions = [(x,y) for x, y in enumerate(spectrum) if y]
        width_mult = self.surface_spectrum_width/max(positions, key=lambda x:x[0])[0]
        positions = [(x*width_mult,y) for x,y in positions]
        # positions = [(int(x*self.surface_spectrum_width/len(positions)),y) for x,y in positions]
        self.surface_spectrum.fill('grey15')
        last_x = 0
        for (a,magnitude), (b,_) in itertools.pairwise(positions):
            height = int(magnitude * self.surface_spectrum_height)
            new_x = (a+b)//2
            rect = pygame.Rect(last_x, 0, new_x - last_x, height)
            last_x = new_x
            brightness = int(magnitude * 100)
            color = pygame.Color(0)
            color.hsva = (40, brightness, brightness, brightness)
            pygame.draw.rect(self.surface_spectrum, color, rect)
            
        self.window.blit(self.surface_spectrum, (0, 0))
    
    def draw_volume(self, time_pos: float):
        """Zeichnet die Volumen-Visualisierung"""
        self.surface_volume.fill('grey15')
        width, height = self.surface_volume.get_size()
        
        x_coords = np.linspace(0, width, self.num_volume_samples)
        progress = time_pos / (len(self.y) / self.sr)
        colors = ['orange' if x/self.num_volume_samples < progress else 'grey30' 
                 for x in range(self.num_volume_samples)]
        
        rect_width = width // self.num_volume_samples + 1
        for x_pos, h, color in zip(x_coords, self.volume_samples * height, colors):
            rect = pygame.Rect(x_pos, 0, rect_width, h)
            pygame.draw.rect(self.surface_volume, color, rect)
        
        ypos = self.window.get_height() - height
        self.window.blit(pygame.transform.flip(self.surface_volume, False, True), (0, ypos))
    
    def get_time_from_click(self, x_pos: int) -> float:
        """Konvertiert eine x-Position zu einer Songposition"""
        return x_pos / self.surface_volume_width * (len(self.y) / self.sr)
        # song_length = len(self.y) / self.sr
        # position = x_pos / self.window.get_width() * song_length
        # return max(0, min(position, song_length - 1))

    def handle_volume_click(self, event: pygame.event.Event) -> Optional[float]:
        """Behandelt Mausinteraktionen mit der Volumen-Visualisierung"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.volume_rect.collidepoint(event.pos):
                self.mouse_down = True
                x = min(max(0, event.pos[0]), self.surface_volume_width)
                return self.get_time_from_click(x)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False
        elif event.type == pygame.MOUSEMOTION and self.mouse_down:
            x = min(max(0, event.pos[0]), self.surface_volume_width)
            return self.get_time_from_click(x)
        return None

    def draw(self, time_pos: float):
        """Zeichnet alle Visualisierungen"""
        self.draw_spectrum(time_pos)
        self.draw_volume(time_pos)

class PlaybackManager:
    def __init__(self, song_length: float):
        self.start_time = time()
        self.current_time = 0.0
        self.is_paused = False
        self.pause_start = 0.0
        self.total_pause_time = 0.0
        self.manual_seek = 0.0
        self.song_length = song_length
    
    def seek(self, position: float):
        position = max(0, min(position, self.song_length))
        self.manual_seek = position
        self.start_time = time()
        self.total_pause_time = 0.0
        if self.is_paused:
            self.pause_start = self.start_time
        mixer_music.set_pos(position)
    
    def play(self):
        if self.is_paused:
            self.total_pause_time += time() - self.pause_start
            self.is_paused = False
            mixer_music.unpause()
    
    def pause(self):
        if not self.is_paused:
            self.pause_start = time()
            self.is_paused = True
            mixer_music.pause()
    
    def reset(self):
        self.start_time = time()
        self.current_time = 0.0
        self.is_paused = False
        self.pause_start = 0.0
        self.total_pause_time = 0.0
        self.manual_seek = 0.0
    
    def get_time(self) -> float:
        if self.is_paused:
            return self.current_time
        
        self.current_time = time() - self.start_time - self.total_pause_time + self.manual_seek
        return int(self.current_time*1000)

class MusicPlayer:
    def __init__(self, music_dir: str = "musik"):
        pygame.init()
        pygame.mixer.init()
        
        self.width, self.height = WIDTH, HEIGHT
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
        cache_path = "music_cache.json"
        metadata_cache = self.load_cache(cache_path)
        
        audio_files = [f for f in os.listdir(self.music_dir) 
                      if os.path.isfile(os.path.join(self.music_dir, f))]
        
        needs_cache_update = False
        self.songs = []

        for file_name in audio_files:
            file_path = os.path.join(self.music_dir, file_name)
            file_mtime = os.path.getmtime(file_path)
            cache_entry = metadata_cache.get(file_path)
            
            if cache_entry and cache_entry.get('mtime') == file_mtime:
                metadata = SongMetadata(**{k: v for k, v in cache_entry.items() 
                                        if k != 'mtime'})
            else:
                print(f"Analyzing {file_name}...")
                try:
                    metadata = self.analyze_file(file_path)
                    metadata_cache[file_path] = {**asdict(metadata), 'mtime': file_mtime}
                    needs_cache_update = True
                except Exception as e:
                    print(f"Error analyzing {file_name}: {e}")
                    continue

            try:
                cover = self.load_cover_from_file(file_path)
            except Exception:
                cover = None

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

        if needs_cache_update:
            self.save_cache(metadata_cache, cache_path)

    def change_song(self, direction: int):
        """Wechselt zum nächsten/vorherigen Song."""
        mixer_music.stop()
        self.current_song_index = (self.current_song_index + direction) % len(self.songs)
        self.setup_current_song()
        mixer_music.play()  # Wichtig: Song direkt starten

    def setup_current_song(self):
        song = self.songs[self.current_song_index]
        mixer_music.load(song.path)
        self.current_visualizer = AudioVisualizer(song, self.window)
        self.playback = PlaybackManager(song.length)

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
        
        # Initial song starten
        mixer_music.play()
        
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
                
                # Handle volume visualizer clicks für Seeking
                if (new_pos := self.current_visualizer.handle_volume_click(event)) is not None:
                    self.playback.seek(new_pos)
            
            # Prüfe ob das Lied zu Ende ist und wechsle zum nächsten
            if not mixer_music.get_busy() and not self.playback.is_paused:
                current_time = self.playback.get_time()/1000
                if current_time >= self.songs[self.current_song_index].length - 0.1:
                    self.change_song(1)
                    continue
            
            # Rendering
            self.window.fill('grey15')
            current_time = self.playback.get_time()/1000
            
            # Draw Cover wenn vorhanden
            song = self.songs[self.current_song_index]
            if song.cover:
                scaled_cover = pygame.transform.scale(song.cover, 
                    (self.width // 2, int(self.height * 0.6)))
                cover_rect = scaled_cover.get_rect(center=(self.width // 2, self.height // 2))
                self.window.blit(scaled_cover, cover_rect)
            
            # Song Info rendern
            font = pygame.font.Font(None, 48)
            start_y = self.window.get_height()//5
            for idx, info in enumerate(song.get_str()):
                title_text = font.render(info, True, (255, 255, 255))
                self.window.blit(title_text, (20, start_y+20+idx*50))
            
            # Zeit Info rendern
            current_min, current_sec = divmod(int(current_time), 60)
            total_min, total_sec = divmod(int(song.length), 60)
            time_text = font.render(
                f"{current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}", 
                True, (255, 255, 255)
            )
            self.window.blit(time_text, (self.width - 200, self.window.get_height()//5+70))
            
            # Visualizer zeichnen
            self.current_visualizer.draw(current_time)
            
            pygame.display.flip()
            clock.tick()
            print(f'{clock.get_fps():.1f} fps', end=f'{" "*9}\r')
        
        pygame.quit()

if __name__ == "__main__":
    player = MusicPlayer()
    player.run()