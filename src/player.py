from .models import Song, SongMetadata
from .visualizer import AudioVisualizer
from .utils import analyze_audio, load_metadata
from .config import *

import os
import pygame
from pygame import mixer_music
from dataclasses import asdict
import json
import audio_metadata
from io import BytesIO
from time import time

class MusicPlayer:
    def __init__(self, music_dir: str = DIR_MUSIC):
        pygame.init()
        pygame.mixer.init()
        
        self.width, self.height = MIN_WIDTH, MIN_HEIGHT
        self.window = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Music Visualizer")
        
        self.music_dir = music_dir
        self.songs: list[Song] = []
        self.current_song_index = 0
        self.current_visualizer = None
        self.playback: PlaybackManager = None
        
        self.load_library()
        self.setup_current_song()
    
    def load_library(self):
        """Lädt Metadaten aus dem cache oder erstellt diesen"""
        cache_path = CACHE_FILE
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
                    # metadata = self.analyze_file(file_path)
                    metadata = load_metadata(file_path)
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
        """Wechselt zum nächsten/vorherigen Song.

        :param direction: 1 für nächsten, -1 für vorherigen Song
        :type direction: int
        """
        mixer_music.stop()
        
        # Cache des alten Visualizers leeren
        if self.current_visualizer:
            self.current_visualizer.clear_cache()
            
        self.current_song_index = (self.current_song_index + direction) % len(self.songs)
        self.setup_current_song()
        mixer_music.play()

    def setup_current_song(self):
        """Lädt song von index in Mixer und aktualisiert AudioVisualizer sowie PlaybackManager"""
        song = self.songs[self.current_song_index]
        mixer_music.load(song.path)
        self.current_visualizer = AudioVisualizer(song, self.window)
        self.playback = PlaybackManager(song.length)

    def load_cache(self, cache_path: str) -> dict:
        """Lädt den Metadaten-Cache aus einer JSON-Datei."""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self, cache: dict, cache_path: str):
        """Speichert den Metadaten-Cache in eine JSON-Datei."""
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)

    def analyze_file(self, file_path: str) -> SongMetadata:
        """Analysiert eine Audiodatei und extrahiert Metadaten."""
        metadata = audio_metadata.load(file_path)
        _,tempo = analyze_audio(file_path)
        
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

    def load_cover_from_file(self, file_path: str) -> pygame.Surface|None:
        """Lädt das Cover aus einer Audiodatei."""
        metadata = audio_metadata.load(file_path)
        if hasattr(metadata, 'pictures') and metadata.pictures:
            image_stream = BytesIO(metadata.pictures[0].data)
            return pygame.image.load(image_stream)

    def draw_cover(self, cover: pygame.Surface):
        """Zeichnet das Cover mit korrektem Aspect Ratio über den gesamten Screen."""
        scale = min(self.width / cover.get_width(), self.height / cover.get_height())
        size = (int(cover.get_width() * scale), int(cover.get_height() * scale))
        scaled = pygame.transform.scale(cover, size)
        pos = ((self.width - size[0]) // 2, (self.height - size[1]) // 2)
        self.window.blit(scaled, pos)

    def draw_frame(self, current_time: int):
        """Zeichnet einen einzelnen Frame.
        
        :param current_time: Aktuelle Zeit in ms
        :type current_time: int
        """
        self.window.fill('grey5')
        
        # Draw Cover wenn vorhanden
        song = self.songs[self.current_song_index]
        if song.cover:
            self.draw_cover(song.cover)
        
        # Song Info rendern
        font = pygame.font.SysFont('agavenerdfontmonoregular', 48)
        start_y = self.window.get_height() * HEIGHT_RATIO_SPECTRUM
        for idx, info in enumerate(song.get_str()):
            title_text = font.render(info, True, (255, 255, 255))
            self.window.blit(title_text, (20, start_y+20+idx*50))
        
        # Zeit Info rendern
        current_min, current_sec = divmod(int(current_time/1000), 60)
        time_text = font.render(
            f"{current_min}:{current_sec:02d}", 
            True, (255, 255, 255)
        )
        self.window.blit(time_text,(self.width - 100, self.window.get_height()-self.window.get_height()*HEIGHT_RATIO_VOLUME-40))
        
        # Visualizer zeichnen
        self.current_visualizer.draw(current_time)

    def run(self):
        """Main Loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Initial song starten
        mixer_music.play()
        last_time = None  # Für Pause-Updates
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width = max(MIN_WIDTH, event.w)
                    self.height = max(MIN_HEIGHT, event.h)
                    self.window = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                    self.current_visualizer.handle_resize((self.width, self.height))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if mixer_music.get_busy():
                            self.playback.pause()
                            last_time = self.playback.get_time()  # Speichere Zeit beim Pausieren
                        else:
                            self.playback.play()
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self.change_song(1 if event.key == pygame.K_RIGHT else -1)
                
                # Handle volume visualizer clicks für Seeking
                if (new_pos := self.current_visualizer.handle_volume_click(event)) is not None:
                    self.playback.seek(new_pos)
                    last_time = new_pos  # Update last_time bei Seek
                    # Force redraw im Pause-Zustand
                    if self.playback.is_paused:
                        self.draw_frame(new_pos)
            
            # Prüfe ob das Lied zu Ende ist und wechsle zum nächsten
            if not mixer_music.get_busy() and not self.playback.is_paused:
                current_time = self.playback.get_time()
                if current_time >= self.songs[self.current_song_index].length - 0.1:
                    self.change_song(1)
                    continue
            
            # Rendering
            current_time = last_time if self.playback.is_paused else self.playback.get_time()
            self.draw_frame(current_time)
            
            pygame.display.flip()
            clock.tick()
            print(f'{clock.get_fps():.1f} fps', end=f'{" "*9}\r')

        pygame.quit()

class PlaybackManager:
    def __init__(self, song_length: float):
        """
        :param song_length: Länge des Songs in Sekunden
        """
        from time import perf_counter_ns
        self.get_time_ns = perf_counter_ns  # Höhere Präzision als time()
        
        self.song_length = int(song_length * 1000)  # in ms
        self.start_time_ns = self.get_time_ns()
        self.is_paused = False
        self.current_time = 0  # in ms
        self.pause_offset_ns = 0
        self.total_pause_ns = 0
        
    def seek(self, position_ms: int) -> None:
        """Springt zu einer Position mit Nanosekunden-Präzision"""
        position_ms = max(0, min(position_ms, self.song_length))
        
        if self.is_paused:
            self.current_time = position_ms
        else:
            # Reset time tracking
            self.start_time_ns = self.get_time_ns()
            self.total_pause_ns = 0
            self.current_time = position_ms
            
            # Reset audio
            mixer_music.rewind()
            mixer_music.play()
            mixer_music.set_pos(position_ms / 1000.0)
    
    def play(self):
        """Startet oder setzt die Wiedergabe fort"""
        if self.is_paused:
            self.total_pause_ns += self.get_time_ns() - self.pause_offset_ns
            self.is_paused = False
            
            # Reset audio an exakter Position
            mixer_music.rewind()
            mixer_music.play()
            mixer_music.set_pos(self.current_time / 1000.0)
    
    def pause(self):
        """Pausiert die Wiedergabe"""
        if not self.is_paused:
            self.pause_offset_ns = self.get_time_ns()
            self.is_paused = True
            self.current_time = self.get_time()
            mixer_music.pause()
    
    def get_time(self) -> int:
        """Gibt die aktuelle Zeit in Millisekunden zurück.
        Nutzt Nanosekunden-Präzision für die Berechnung."""
        if self.is_paused:
            return self.current_time
            
        elapsed_ns = self.get_time_ns() - self.start_time_ns - self.total_pause_ns
        return min(max(0,self.current_time + int(elapsed_ns / 1_000_000) + PLAYBACK_MANAGER_DELAY_S*1000),self.song_length)  # ns to ms
