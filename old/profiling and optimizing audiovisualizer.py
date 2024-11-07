from line_profiler import LineProfiler
import numpy as np
from functools import lru_cache

class AudioVisualizer:
    """Profile einzelne Methoden des Visualizers."""
    def __init__(self, song: Song, window: pygame.Surface):
        self.profiler = LineProfiler()
        self._profile_init(song, window)
        
    def _profile_init(self, song: Song, window: pygame.Surface):
        t0 = time()
        
        # Audio laden
        print("Loading audio...")
        t1 = time()
        self.y, self.sr = librosa.load(song.path, sr=None)
        print(f"Audio load time: {(time()-t1)*1000:.1f}ms")
        
        # FFT Setup
        print("Setting up FFT...")
        t1 = time()
        self.setup_spectrum_analyzer()
        print(f"FFT setup time: {(time()-t1)*1000:.1f}ms")
        
        # Rest des Setups
        self.window = window
        self.screen = pygame.Surface((window.get_width(), window.get_width()//5), pygame.SRCALPHA)
        
        print(f"Total init time: {(time()-t0)*1000:.1f}ms")

    def setup_spectrum_analyzer(self, min_freq=10, max_freq=16000, num_bands=300):
        t0 = time()
        
        # STFT berechnen
        print("Computing STFT...")
        t1 = time()
        self.D = librosa.stft(self.y, n_fft=10000, hop_length=512)
        self.S = np.abs(self.D)
        print(f"STFT time: {(time()-t1)*1000:.1f}ms")
        
        # Frequenzbänder
        print("Setting up frequency bands...")
        t1 = time()
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=10000)
        self.freq_bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands)
        print(f"Frequency setup time: {(time()-t1)*1000:.1f}ms")
        
        print(f"Total spectrum setup time: {(time()-t0)*1000:.1f}ms")

# Optimierte Version:
class OptimizedAudioVisualizer:
    def __init__(self, song: Song, window: pygame.Surface):
        self.window = window
        self.screen = pygame.Surface((window.get_width(), window.get_width()//5), pygame.SRCALPHA)
        
        # Lazy loading - nur was wir sofort brauchen
        self.song_path = song.path
        self._y = None
        self._sr = None
        self._D = None
        self._S = None
        self._freqs = None
        
        # FFT Parameter
        self.n_fft = 4096  # Kleinerer FFT-Size für schnellere Berechnung
        self.hop_length = 512
        self.num_bands = 300
        self.min_freq = 10
        self.max_freq = 16000
        
        # Frequenzbänder vorberechnen (schnell)
        self.freq_bands = np.logspace(
            np.log10(self.min_freq), 
            np.log10(self.max_freq), 
            self.num_bands
        )
        
        # Cache für spektrale Daten
        self._spectrum_cache = {}
        self._max_cache_size = 1000  # Cache-Größe begrenzen
    
    @property
    def y(self):
        """Lazy loading für Audio-Daten."""
        if self._y is None:
            self._y, self._sr = librosa.load(self.song_path, sr=None)
        return self._y
    
    @property
    def sr(self):
        """Lazy loading für Sample Rate."""
        if self._sr is None:
            self._y, self._sr = librosa.load(self.song_path, sr=None)
        return self._sr
    
    @property
    def S(self):
        """Lazy loading für Spektrogramm."""
        if self._S is None:
            # Berechne STFT nur für den ersten Teil des Songs
            chunk_duration = 30  # Sekunden
            chunk_samples = int(chunk_duration * self.sr)
            y_chunk = self.y[:chunk_samples] if len(self.y) > chunk_samples else self.y
            
            self._D = librosa.stft(y_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
            self._S = np.abs(self._D)
        return self._S
    
    @property
    def freqs(self):
        """Lazy loading für Frequenzen."""
        if self._freqs is None:
            self._freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        return self._freqs
    
    @lru_cache(maxsize=1024)
    def get_spectrum_cached(self, frame_idx: int) -> np.ndarray:
        """Cached Version der Spektrum-Berechnung."""
        current_spectrum = self.S[:, min(frame_idx, self.S.shape[1] - 1)]
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
    
    def get_current_spectrum(self, time_pos: float) -> np.ndarray:
        """Optimierte Version der Spektrum-Berechnung."""
        frame_idx = int(time_pos * self.sr / self.hop_length)
        return self.get_spectrum_cached(frame_idx)