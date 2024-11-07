# src/config.py
import os

# Basis-Verzeichnisse
DIR_BASE = os.path.dirname(os.path.dirname(__file__))
DIR_DATA = os.path.join(DIR_BASE, "data")
DIR_CACHE = os.path.join(DIR_DATA, "cache")
DIR_MUSIC = os.path.join(DIR_BASE, "musik")

# Dateipfade
CACHE_FILE = os.path.join(DIR_CACHE, "music_cache.json")

# Window Config
MIN_WIDTH = 800
MIN_HEIGHT = 600
WINDOW_TITLE = "Music Visualizer"

# Visualizer Config
HEIGHT_RATIO_SPECTRUM = 0.2  # statt //5
HEIGHT_RATIO_VOLUME = 0.1    # statt //10

# Audio Processing Config
FFT_SIZE = 8192  # war vorher n_fft
HOP_LENGTH = 256
FREQ_MIN = 10
FREQ_MAX = 16000

# UI Config
FONT_SIZE = 48
FONT_FACE = 'agavenerdfontmonoregular'
COLOR_TEXT = 'white'
COLOR_BACKGROUND = 'grey15'
COLOR_SPECTRUM_HUE = 40  # für deine HSV Farben im Spektrum

COLOR_WAVEFORM = 'white'
WAVEFORM_WIDTH_RATIO = 0.2
WAVEFORM_TIME_S = 0.05
PLAYBACK_MANAGER_DELAY_S = -0.2