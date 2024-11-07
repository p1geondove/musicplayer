import os
import librosa
from librosa.feature import rhythm
from dataclasses import dataclass

"""
filetree beispiel

song-craler.py
musik/
    song1/
        Linkin Park - In The End.wav
        cover.jpg
    egal/
        Justin Timberlake - Mirrors.mp3
        5986b1234907a15fd123c.png
"""

@dataclass
class Song:
    path:str        #path to the song
    cover:str       #path to the cover, later to be replaced with pygame surface
    length:float    #song lengt in seconds
    bpm:float       #bpm eg 161.5

    #next couple entries are not yet implemented
    release:str     #release date metadata from file, import datetime only if neccesarry
    album:str       #also from metadata
    genre:str
    tags:list[str]  #custom set tags eg ['techno','melodic','vocal'], ill decide that later


audio_extensions = {'wav','mp3','ogg'}
image_extensions = {'jpg','jpeg','png','bmp'}

class SongCrawler:
    def __init__(self):
        self.main_dir = os.path.join(os.getcwd(),'musik')
        self.music:list[Song] = []

        for path, dirs, files in next(os.walk(self.main_dir)):
            for file in files:
                audio_files = []
                image_files = []
                abs_path = os.path.join(path, file)

                for ext in audio_extensions:
                    if file.endswith(ext):
                        audio_files.append(abs_path)

                for ext in image_extensions:
                    if file.endswith(ext):
                        image_files.append(abs_path)
                
                if audio_files:
                    audio_file = audio_files[0]
                else: continue

                if image_files:
                    cover = audio_files[0]
                else: continue

                y, sr = librosa.load(audio_file, sr=None)
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
                length = len(y) / sr

                song = Song(
                    path=audio_file,
                    cover=cover,
                    length=length,
                    bpm=tempo,
                )

                self.music.append(song)
                
a = SongCrawler()