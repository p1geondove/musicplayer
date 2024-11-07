
if __name__ == "__main__":
    audio_file = "Raxeller - SHATTERED SILENCE - 03 BLINDED BY HATE.wav"  # Pfad zu deiner Audiodatei
    
    print('importing librosa')
    import librosa
    
    print('importing rytm')
    from librosa.feature import rhythm
    
    print('loading file')
    y, sr = librosa.load(audio_file, sr=None)
    
    print(f'calcing onset_strength\tsample rate:{sr}Hz')
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    print(f'calcing tempo')
    tempo = rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    print(f'{tempo:.1f} BPM')