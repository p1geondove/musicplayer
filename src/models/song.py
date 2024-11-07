from dataclasses import dataclass
from pygame import Surface

@dataclass
class Song:
    path: str
    title: str
    artist: str
    album: str
    genre: str
    year: str
    length: float
    bpm: float
    cover: Surface | None
    
    def get_display_name(self) -> str:
        artist = self.artist or "Unknown Artist"
        title = self.title or "Unknown Title"
        return f"{artist} - {title}"

    def get_str(self) -> list[str]:
        _min, _sec = divmod(self.length,60)
        return [
            self.title,
            self.artist,
            self.album,
            self.genre,
            self.year,
            f'{int(_min):1d}:{int(_sec):02d}',
            f'{self.bpm:.1f}'
        ]

@dataclass
class SongMetadata:
    """Separate Klasse für die zu cachenden Metadaten"""
    path: str
    title: str
    artist: str
    album: str
    genre: str
    year: str
    length: float
    bpm: float