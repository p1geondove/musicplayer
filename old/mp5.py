from music_player4 import *
import cProfile
import pstats
from line_profiler import LineProfiler
import time
import pygame.surfarray

def profile_visualizer():
    """Profile die Visualizer-Klasse"""
    prof = LineProfiler()
    # Dekoriere die wichtigsten Methoden
    prof.add_function(AudioVisualizer.get_current_spectrum)
    prof.add_function(AudioVisualizer.draw_spectrum)
    prof.add_function(AudioVisualizer.draw_volume)
    return prof

class OptimizedAudioVisualizer(AudioVisualizer):
    """Optimierte Version des AudioVisualizers"""
    def __init__(self, song: Song, window: pygame.Surface):
        super().__init__(song, window)
        
        # Pre-allocate surfaces und numpy arrays
        self.spectrum_array = pygame.surfarray.pixels3d(self.surface_spectrum)
        self.volume_array = pygame.surfarray.pixels3d(self.surface_volume)
        
        # Vorberechne häufig verwendete Werte
        self.spectrum_width = self.surface_spectrum_width
        self.spectrum_height = self.surface_spectrum_height
        self.volume_width = self.surface_volume_width
        self.volume_height = self.surface_volume_height
        
        # Cache für Farben
        self.color_cache = {}
        self._precalculate_colors()
        
        # Numpy-Arrays für die Visualisierung
        self.x_coords = np.linspace(0, self.volume_width, self.num_volume_samples)
        self.rect_width = self.volume_width // self.num_volume_samples + 1

        # Vorberechne ALLE Frequenzmasken als numpy-Arrays
        print("Precomputing frequency masks...")
        self.freq_masks = np.zeros((self.num_bands, len(self.freqs)), dtype=bool)
        for i in range(self.num_bands - 1):
            self.freq_masks[i] = (self.freqs >= self.freq_bands[i]) & (self.freqs < self.freq_bands[i + 1])
        self.freq_masks[-1] = self.freqs >= self.freq_bands[-1]

        # Erstelle vorberechnete Matrizen für die Spektrum-Berechnung
        self.precalc_spectrum = np.zeros((self.S.shape[1], self.num_bands))
        print("Precomputing spectrum matrices...")
        for i in range(self.num_bands):
            mask = self.freq_masks[i]
            if mask.any():
                self.precalc_spectrum[:, i] = np.mean(self.S[mask], axis=0)

        # Logarithmische Skalierung der vorberechneten Spektren
        self.precalc_spectrum = np.log10(self.precalc_spectrum + 1)
        # Normalisierung pro Frame
        max_vals = np.maximum(np.max(self.precalc_spectrum, axis=1), 1e-10)
        self.precalc_spectrum = (self.precalc_spectrum.T / max_vals).T

        # Surface Arrays für direktes Pixel-Setting
        self.spectrum_surface = pygame.Surface((self.surface_spectrum_width, self.surface_spectrum_height))
        self.spectrum_array = pygame.surfarray.pixels3d(self.spectrum_surface)
        
        # Vorberechne die Farbpalette
        self.color_palette = np.zeros((101, 3), dtype=np.uint8)
        for i in range(101):
            color = pygame.Color(0)
            color.hsva = (40, i, i, i)
            self.color_palette[i] = [color.r, color.g, color.b]
    
    @lru_cache(maxsize=None)
    def get_current_spectrum(self, time_ms: int) -> np.ndarray:
        """Optimierte Spektrum-Berechnung mit vorberechneten Matrizen"""
        frame_idx = int((time_ms / 1000.0) * self.sr / self.hop_length)
        frame_idx = min(frame_idx, self.precalc_spectrum.shape[0] - 1)
        return self.precalc_spectrum[frame_idx]

    def _precalculate_colors(self):
        """Vorberechnung der Farben für bessere Performance"""
        for brightness in range(101):
            color = pygame.Color(0)
            color.hsva = (40, brightness, brightness, brightness)
            self.color_cache[brightness] = (
                color.r,
                color.g,
                color.b
            )
    
    def draw_spectrum(self, time_ms: int):
        """Optimierte Zeichenroutine mit weniger draw-Aufrufen"""
        spectrum = self.get_current_spectrum(time_ms)
        
        # Vorberechne alle Positionen auf einmal
        positions = np.column_stack((
            np.arange(len(spectrum))[spectrum > 0],
            spectrum[spectrum > 0]
        ))
        
        if len(positions) == 0:
            return
            
        width_mult = self.surface_spectrum_width / positions[-1,0]
        
        # Erstelle ein numpy array für die gesamte Oberfläche
        surface_array = np.zeros((self.surface_spectrum_width, self.surface_spectrum_height, 3), dtype=np.uint8)
        surface_array.fill(30)  # grey15
        
        # Batch-Weise Rechtecke zeichnen
        batch_size = 100  # Experimentiere mit diesem Wert
        for i in range(0, len(positions) - 1, batch_size):
            batch = positions[i:i+batch_size]
            heights = (batch[:,1] * self.surface_spectrum_height).astype(int)
            x_starts = (batch[:,0] * width_mult).astype(int)
            x_ends = (x_starts[1:]).astype(int)
            
            # Zeichne alle Rechtecke dieser Batch auf einmal
            for x1, x2, h in zip(x_starts, x_ends, heights):
                if h > 0:
                    brightness = int(h / self.surface_spectrum_height * 100)
                    color = self.color_cache[brightness]
                    surface_array[x1:x2, :h] = color
                    
        # Update die Surface
        pygame.surfarray.blit_array(self.surface_spectrum, surface_array)
        self.window.blit(self.surface_spectrum, (0, 0))
    
    def draw_volume(self, time_ms: int):
        """Optimierte Version der Volumen-Visualisierung"""
        # Clear surface with numpy
        self.volume_array[:] = (30, 30, 30)  # grey15
        
        # Numpy-optimierte Berechnungen
        progress = time_ms / (len(self.y) / self.sr) / 1000.0
        progress_mask = np.arange(self.num_volume_samples) / self.num_volume_samples < progress
        
        heights = self.volume_samples * self.volume_height
        
        for i, (x_pos, height) in enumerate(zip(self.x_coords, heights)):
            if height > 0:
                color = (255, 165, 0) if progress_mask[i] else (77, 77, 77)  # orange : grey30
                x_start = int(x_pos)
                x_end = min(x_start + self.rect_width, self.volume_width)
                h = int(height)
                
                # Direktes Array-Filling
                self.volume_array[x_start:x_end, :h] = color
        
        # Aktualisiere die Surface
        pygame.surfarray.blit_array(self.surface_volume, self.volume_array)
        ypos = self.window.get_height() - self.volume_height
        self.window.blit(pygame.transform.flip(self.surface_volume, False, True), (0, ypos))

    

class ProfilingMusicPlayer(MusicPlayer):
    def __init__(self, music_dir: str = "musik"):
        super().__init__(music_dir)
        self.frame_times = []
        self.profiler = cProfile.Profile()
        
    def run(self):
        self.profiler.enable()
        last_time = time.time()
        
        try:
            super().run()
        finally:
            self.profiler.disable()
            stats = pstats.Stats(self.profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(30)  # Top 30 zeitintensivste Funktionen
            
            # Analysiere Frame-Zeiten
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            print(f"\nDurchschnittliche Frame-Zeit: {avg_frame_time*1000:.2f}ms")
            print(f"Min FPS: {1/max(self.frame_times):.1f}")
            print(f"Max FPS: {1/min(self.frame_times):.1f}")
            print(f"Avg FPS: {1/avg_frame_time:.1f}")
    
    def update(self):
        """Misst die Frame-Zeit"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time

if __name__ == "__main__":
    # Profiling aktivieren
    player = ProfilingMusicPlayer()
    player.run()