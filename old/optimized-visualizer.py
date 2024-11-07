class AudioVisualizer:
    def __init__(self, song: Song, window: pygame.Surface):
        self.window = window
        self.screen = pygame.Surface((window.get_width(), window.get_width()//5), pygame.SRCALPHA)
        
        # Audio laden
        print("Loading audio...")
        t1 = time()
        self.y, self.sr = librosa.load(song.path, sr=None)
        print(f"Audio load time: {(time()-t1)*1000:.1f}ms")
        
        # FFT Setup mit kleinerer Fenstergröße
        print("Setting up FFT...")
        t1 = time()
        self.n_fft = 4096  # Kleinerer FFT-Size (war 10000)
        self.hop_length = 512
        
        # Spektrogramm für die ersten 30 Sekunden berechnen
        chunk_duration = 30  # Sekunden
        chunk_samples = int(chunk_duration * self.sr)
        y_chunk = self.y[:chunk_samples] if len(self.y) > chunk_samples else self.y
        
        self.D = librosa.stft(y_chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S = np.abs(self.D)
        print(f"STFT time: {(time()-t1)*1000:.1f}ms")
        
        # Frequenzbänder
        self.num_bands = 300
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.freq_bands = np.logspace(np.log10(10), np.log10(16000), self.num_bands)
        
        # Cache für die Spektren
        self._spectrum_cache = {}
        
    def get_current_spectrum(self, time_pos: float) -> np.ndarray:
        """Optimierte Version mit Caching."""
        frame_idx = int(time_pos * self.sr / self.hop_length)
        
        # Prüfe Cache
        if frame_idx in self._spectrum_cache:
            return self._spectrum_cache[frame_idx]
            
        # Begrenze frame_idx auf verfügbare Daten
        frame_idx = min(frame_idx, self.S.shape[1] - 1)
        
        # Berechne Spektrum
        current_spectrum = self.S[:, frame_idx]
        band_magnitudes = np.zeros(self.num_bands)
        
        for i in range(self.num_bands):
            if i < self.num_bands - 1:
                freq_mask = (self.freqs >= self.freq_bands[i]) & (self.freqs < self.freq_bands[i + 1])
            else:
                freq_mask = (self.freqs >= self.freq_bands[i])
            
            if np.any(freq_mask):
                band_magnitudes[i] = np.mean(current_spectrum[freq_mask])
        
        # Logarithmische Skalierung und Normalisierung
        band_magnitudes = np.log10(band_magnitudes + 1)
        band_magnitudes = band_magnitudes / np.max(band_magnitudes)
        
        # Cache das Ergebnis (begrenzte Cache-Größe)
        if len(self._spectrum_cache) > 1000:  # Cache-Größe begrenzen
            self._spectrum_cache.clear()
        self._spectrum_cache[frame_idx] = band_magnitudes
        
        return band_magnitudes
        
    def draw(self, time_pos):
        """Unveränderte draw Methode."""
        self.screen.fill('grey15')
        bar_width = self.screen.get_width() / self.num_bands
        for i, magnitude in enumerate(self.get_current_spectrum(time_pos)):
            height = int(magnitude * self.screen.get_height())
            x = i * bar_width
            rect = pygame.Rect(x, self.screen.get_height() - height, 2, height)
            
            # Farbe basierend auf Frequenz und Amplitude
            hue = i / self.num_bands * 255
            brightness = int(magnitude * 100)
            color = pygame.Color(0)
            color.hsva = (hue, 100, brightness, 100)
            
            pygame.draw.rect(self.screen, color, rect)
            
        self.window.blit(pygame.transform.flip(self.screen, False, True), (0, 0))
