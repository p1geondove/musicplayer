from src.models.song import Song
from src.config import *

import pygame
import librosa
import numpy as np
from functools import lru_cache
from itertools import pairwise

class AudioVisualizer:
    def __init__(self, song: Song, window: pygame.Surface):
        self.get_current_spectrum.cache_clear()
        self.window = window
        # Erstelle Oberflächen für beide Visualisierungen
        self.surface_spectrum_width = window.get_width()
        self.surface_spectrum_height = window.get_height() * HEIGHT_RATIO_SPECTRUM
        self.surface_volume_width = window.get_width()
        self.surface_volume_height = window.get_height() * HEIGHT_RATIO_VOLUME
        self.surface_spectrum = pygame.Surface((self.surface_spectrum_width, self.surface_spectrum_height), pygame.SRCALPHA)
        self.surface_volume = pygame.Surface((self.surface_volume_width, window.get_height()*HEIGHT_RATIO_VOLUME), pygame.SRCALPHA)
        
        # Audio laden
        self.y, self.sr = librosa.load(song.path, sr=None)
        
        # FFT Setup mit optimierten Parametern
        self.n_fft = FFT_SIZE  # Optimierte FFT-Größe
        self.hop_length = HOP_LENGTH
        
        # Berechne das komplette Spektrogramm
        self.D = librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S = np.abs(self.D)
        
        # Frequenzbänder
        self.num_bands = window.get_width()
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.freq_bands = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), self.num_bands)
        
        # Volume Visualizer Setup
        self.setup_volume_visualizer()
        
        # Maus-Interaktion
        self.mouse_down = False
        self.volume_rect = pygame.Rect(0, window.get_height() - self.surface_volume.get_height(),
                                     self.surface_volume_width, self.surface_volume.get_height())
        
        self.song_path = song.path
        self._y = None
        self._sr = None
        self._D = None
        self._S = None

    def setup_volume_visualizer(self, num_samples: int = None):
        """Initialisiert die Volumen-Visualisierung"""
        if num_samples is None:
            num_samples = self.surface_volume_width+1
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
    
    @classmethod
    def clear_cache(cls):
        """Leert den Spektrum-Cache"""
        cls.get_current_spectrum.cache_clear()  # type: ignore

    @lru_cache(maxsize=None)
    def get_current_spectrum(self, time_ms: int) -> np.ndarray:
        """Optimierte und gecachte Spektrum-Berechnung"""
        # Konvertiere ms zu Frames
        frame_idx = int((time_ms / 1000.0) * self.sr / self.hop_length)
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
    
    def draw_spectrum(self, time_ms: int):
        """Zeichnet das Frequenzspektrum"""
        self.surface_spectrum = pygame.Surface(self.surface_spectrum.get_size(), pygame.SRCALPHA)
        spectrum = self.get_current_spectrum(time_ms)
        positions = [(x,y) for x, y in enumerate(spectrum) if y]
        width_mult = self.surface_spectrum_width/max(positions, key=lambda x:x[0])[0]
        positions = [(x*width_mult,y) for x,y in positions]
        
        last_x = 0
        for (a,magnitude), (b,_) in pairwise(positions):
            height = int(magnitude * self.surface_spectrum_height)
            new_x = (a+b)//2
            rect = pygame.Rect(last_x, 0, new_x - last_x, height)
            last_x = new_x
            brightness = int(magnitude * 100)
            color = pygame.Color(0)
            color.hsva = (40, brightness, brightness, brightness)
            pygame.draw.rect(self.surface_spectrum, color, rect)
            
        self.window.blit(self.surface_spectrum, (0, 0))
    
    def draw_volume(self, time_ms: int):
        """Zeichnet die Volumen-Visualisierung"""
        width, height = self.surface_volume.get_size()
        
        progress = time_ms / (len(self.y) / self.sr) / 1000.0
        x_coords = np.linspace(0, width, self.num_volume_samples)
        colors = ['orange' if x/self.num_volume_samples < progress else 'grey30' 
                 for x in range(self.num_volume_samples)]
        
        rect_width = width // self.num_volume_samples + 1
        for x_pos, h, color in zip(x_coords, self.volume_samples * height, colors):
            rect = pygame.Rect(x_pos, 0, rect_width, h)
            pygame.draw.rect(self.surface_volume, color, rect)
        
        ypos = self.window.get_height() - height
        self.window.blit(pygame.transform.flip(self.surface_volume, False, True), (0, ypos))
    
    def get_time_from_click(self, x_pos: int) -> int:
        """Konvertiert eine x-Position zu einer Songposition in Millisekunden"""
        song_length_sec = len(self.y) / self.sr
        position_ms = int((x_pos / self.surface_volume_width) * song_length_sec * 1000)
        return max(0, min(position_ms, int(song_length_sec * 1000) - 1))

    def handle_volume_click(self, event: pygame.event.Event) -> int|None:
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

    def draw(self, time_pos: float):
        """Zeichnet alle Visualisierungen"""
        self.draw_spectrum(time_pos)
        self.draw_volume(time_pos)

    def handle_resize(self, new_size: tuple[int, int]):
        """Behandelt die Größenänderung des Fensters"""
        self.window = pygame.display.get_surface()
        
        # Update surfaces
        self.surface_spectrum_width = self.window.get_width()
        self.surface_spectrum_height = self.window.get_height() * HEIGHT_RATIO_SPECTRUM
        self.surface_volume_width = self.window.get_width()
        self.surface_volume_height = self.window.get_height() * HEIGHT_RATIO_VOLUME
        
        self.surface_spectrum = pygame.Surface((self.surface_spectrum_width, self.surface_spectrum_height), pygame.SRCALPHA)
        self.surface_volume = pygame.Surface((self.surface_volume_width, self.surface_volume_height), pygame.SRCALPHA)
        
        # Update spectrum bands
        self.num_bands = self.surface_spectrum_width
        self.freq_bands = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), self.num_bands)
        
        # Update volume visualizer
        self.setup_volume_visualizer()
        self.volume_rect = pygame.Rect(0, self.window.get_height() - self.surface_volume_height,
                                    self.surface_volume_width, self.surface_volume_height)
        
        # Clear spectrum cache
        self.get_current_spectrum.cache_clear()
