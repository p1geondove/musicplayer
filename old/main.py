import librosa
import numpy as np
import pygame
import os

class AudioVisualizer:
    def __init__(self, audio_path, num_samples=100):
        # Laden der Audiodatei
        self.y, self.sr = librosa.load(audio_path)
        print(self.y)
        self.num_samples = num_samples
        
        # Berechnung der durchschnittlichen Amplituden
        self.samples = self._calculate_samples()
        
        # Pygame Setup
        pygame.init()
        self.width = 800
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Audio Visualizer")
        
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
    
    def draw(self):
        self.screen.fill((0, 0, 0))  # Schwarzer Hintergrund
        
        # Berechne x-Koordinaten für die Linien
        x_coords = np.linspace(0, self.width, self.num_samples)
        
        # Erstelle Liste von Punkten für pygame.draw.lines()
        points = []
        for i in range(self.num_samples):
            x = x_coords[i]
            # Skaliere die Amplitude auf die Bildschirmhöhe
            y = self.height/2 * (1 - self.samples[i])
            points.append((x, y))
            
        # Zeichne die Linie
        pygame.draw.lines(self.screen, (0, 255, 0), False, points, 2)
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            self.draw()
            pygame.time.wait(50)
            
        pygame.quit()

# Beispielnutzung
if __name__ == "__main__":
    audio_file = "Realization MST v1.wav"  # Pfad zu deiner Audiodatei
    visualizer = AudioVisualizer(audio_file)
    visualizer.run()
