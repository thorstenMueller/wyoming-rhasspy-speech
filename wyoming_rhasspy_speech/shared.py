import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rhasspy_speech import KaldiTranscriber

ARPA = "arpa"
GRAMMAR = "grammar"
LANG_TYPES = (ARPA, GRAMMAR)


@dataclass
class AppSettings:
    train_dir: Path
    tools_dir: Path
    models_dir: Path
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
