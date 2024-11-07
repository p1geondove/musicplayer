import librosa
import numpy as np
import pygame
from pygame import mixer_music
from time import perf_counter as time

class SpectrumAnalyzer:
    def __init__(self, audio_path:str, window:pygame.Surface, min_freq=10, max_freq=16000, num_bands=300):
        # Audio laden
        self.y, self.sr = librosa.load(audio_path,sr=None)
        self.audio_path = audio_path
        
        # FFT Parameter
        self.n_fft = 10000  # Fenstergröße für FFT
        self.hop_length = 512  # Schrittweite zwischen Fenstern
        
        # Frequenzbänder definieren
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.freq_bands = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands)
        
        # Pygame Setup
        self.window = window
        self.screen = pygame.Surface((window.get_width(),window.get_width()//5),pygame.SRCALPHA)
        
        # Zeitliche Synchronisation
        self.start_time = 0
        self.current_frame = 0
        
        # Spektrogramm vorberechnen
        self.D = librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length)
        self.S = np.abs(self.D)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Spektrum-Daten vorbereiten
        self.times = librosa.times_like(self.S, sr=self.sr, hop_length=self.hop_length)
        
    def get_current_spectrum(self, time_pos:float):
        # Aktuelle Wiedergabeposition bestimmen
        frame_idx = int(time_pos * self.sr / self.hop_length)
        frame_idx = min(frame_idx, self.S.shape[1] - 1)
        
        # Aktuelles Spektrum holen
        current_spectrum = self.S[:, frame_idx]
        
        # Interpolation auf gewünschte Frequenzbänder
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
        
        return band_magnitudes.astype(float)  # Explizite Konvertierung zu float
        
    def draw(self, time_pos):
        # Zeichne Frequenzbänder
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
        self.window.blit(pygame.transform.flip(self.screen,False,True),(0,0))

class VolumeVisualizer:
    def __init__(self, audio_path:str, window:pygame.Surface, num_samples=401):
        # Laden der Audiodatei
        self.y, self.sr = librosa.load(audio_path)
        self.num_samples = num_samples
        self.song_lenght = len(self.y) / self.sr
        
        # Berechnung der durchschnittlichen Amplituden
        self.samples = self._calculate_samples()
        
        # Pygame Setup
        self.window = window
        height = window.get_height()
        win_height = height // 8
        self.screen = pygame.Surface((window.get_width(),win_height))
        self.rect = pygame.Rect(0,height-win_height, window.get_width(), win_height)
        self.mouse_down = False

    def _calculate_samples(self) -> np.ndarray[float]:
        # Teile das Audio in num_samples Segmente
        segment_size = len(self.y) // self.num_samples
        samples = []
        
        for i in range(self.num_samples):
            start = i * segment_size
            end = start + segment_size
            segment = self.y[start:end]
            # Berechne durchschnittliche absolute Amplitude für das Segment
            amplitude = np.mean(np.abs(segment))
            samples.append(amplitude)
            
        # Normalisiere die Werte auf einen Bereich von 0 bis 1
        samples = np.array(samples)
        samples = samples / np.max(samples)

        return samples.astype(float)
    
    def draw(self, music_pos:float):
        self.screen.fill('grey15')  # Schwarzer Hintergrund
        width, height = self.screen.get_size()
        
        # Berechne x-Koordinaten für die Linien
        x_coords = np.linspace(0, width, self.num_samples)
        colors = ['orange' if x/self.num_samples < music_pos/self.song_lenght else 'grey30' for x in range(self.num_samples)]
        
        # Erstelle Liste von Punkten für pygame.draw.lines()
        points = []
        rect_width = width//self.num_samples+1
        for x_pos, h, color in zip(x_coords, self.samples*height, colors):
            rect = pygame.Rect(x_pos, 0, rect_width, h)
            pygame.draw.rect(self.screen, color, rect)
            # rect = pygame.Rect(x_pos, 0, rect_width, h)
            # pygame.draw.rect(self.screen, 'yellow', rect,1,1)

        ypos = self.window.get_height()-height
        self.window.blit(pygame.transform.flip(self.screen,False,True),(0,ypos))

    def handle_event(self, event:pygame.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.mouse_down = True
                music_pos = event.pos[0]/self.screen.get_width()*self.song_lenght
                mixer_music.set_pos(music_pos)
                self.draw(music_pos)
                return music_pos
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_down = False
        elif event.type == pygame.MOUSEMOTION and self.mouse_down:
            music_pos = event.pos[0]/self.screen.get_width()*self.song_lenght
            self.draw(music_pos)
            mixer_music.set_pos(music_pos)
            return music_pos

def main():
    width, height = 400, 800
    audio_path = "Realization MST v1.wav"
    pygame.init()
    pygame.mixer.init()
    mixer_music.load(audio_path)

    clock = pygame.time.Clock()
    window = pygame.display.set_mode((width,height),pygame.SRCALPHA)
    volume = VolumeVisualizer(audio_path, window)
    spectrum = SpectrumAnalyzer(audio_path, window)
    music_pos = 0
    start_time = time()
    pause_time = start_time-1
    last_time = start_time

    mixer_music.rewind()
    mixer_music.play()

    running = True
    while running:
        for event in pygame.event.get():
            if (x:=volume.handle_event(event)):
                music_pos = x
                start_time = time() - music_pos
                pause_time = start_time-1
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # mixer_music.pause() if mixer_music.get_busy() else mixer_music.unpause()
                    if mixer_music.get_busy():
                        mixer_music.pause()
                        pause_time = time()
                    else:
                        mixer_music.unpause()
                        music_pos += time() - pause_time
                        pause_time = start_time-1

        window.fill('grey15')
        if pause_time < start_time:
            music_pos = time() - start_time

        spectrum.draw(music_pos-.5) #-.5 weil da immer irgend ein offset ist ka woher der kommt
        volume.draw(music_pos)
        clock.tick()
        pygame.display.flip()
        
    pygame.quit()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()