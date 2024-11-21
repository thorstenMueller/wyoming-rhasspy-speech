import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rhasspy_speech import KaldiTranscriber

ARPA = "arpa"
GRAMMAR = "grammar"
LANG_TYPES = (ARPA, GRAMMAR)


@dataclass
class TranscriberSettings:
    is_enabled: bool
    max_active: int
    lattice_beam: float
    acoustic_scale: float
    beam: float


@dataclass
class AppSettings:
    train_dir: Path
    tools_dir: Path
    models_dir: Path

    # VAD
    vad_enabled: bool
    vad_threshold: float
    before_speech_seconds: float

    # Speex
    speex_enabled: bool
    speex_noise_suppression: int
    speex_auto_gain: int

    # Edit distance
    word_norm_distance_threshold: float
    char_norm_distance_threshold: float

    # Transcribers
    transcriber_settings: Dict[str, TranscriberSettings] = field(default_factory=dict)

    # Home Assistant
    hass_token: Optional[str] = None
    hass_websocket_uri: str = "homeassistant.local"
    hass_ingress: bool = False


@dataclass
class AppState:
    settings: AppSettings

    # model_id -> words
    skip_words: Dict[str, List[str]] = field(default_factory=dict)

    # model_id -> lang_type -> transcriber
    transcribers: Dict[str, Dict[str, KaldiTranscriber]] = field(default_factory=dict)
    transcribers_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
