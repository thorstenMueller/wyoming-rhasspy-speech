from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rhasspy_speech.const import LangSuffix


@dataclass
class AppSettings:
    train_dir: Path
    tools_dir: Path
    models_dir: Path

    volume_multiplier: float

    # VAD
    vad_enabled: bool
    vad_threshold: float
    before_speech_seconds: float

    # Speex
    speex_enabled: bool
    speex_noise_suppression: int
    speex_auto_gain: int

    # Edit distance
    norm_distance_threshold: float

    # Transcribers
    max_active: int
    lattice_beam: float
    acoustic_scale: float
    beam: float
    nbest: int
    streaming: bool

    decode_mode: LangSuffix
    arpa_rescore_order: Optional[int]

    # Home Assistant
    hass_token: Optional[str] = None
    hass_websocket_uri: str = "homeassistant.local"
    hass_ingress: bool = False


@dataclass
class AppState:
    settings: AppSettings

    # model_id -> words
    skip_words: Dict[str, List[str]] = field(default_factory=dict)
