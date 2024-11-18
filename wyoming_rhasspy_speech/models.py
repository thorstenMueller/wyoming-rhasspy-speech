from dataclasses import dataclass
from typing import Dict, Optional

URL_FORMAT = "https://huggingface.co/rhasspy/rhasspy-speech/resolve/main/models/{model_id}?download=true"


@dataclass
class Model:
    id: str
    language: str
    language_code: str
    attribution: str
    url: str
    version: Optional[str] = None


MODELS: Dict[str, Model] = {
    m.id: m
    for m in [
        Model(
            id="en_US-rhasspy",
            language="English, United States",
            language_code="en_US",
            attribution="Rhasspy",
            url=URL_FORMAT.format(model_id="en_US-rhasspy"),
        ),
        Model(
            id="fr_FR-rhasspy",
            language="French, France",
            language_code="fr_FR",
            attribution="Rhasspy",
            url=URL_FORMAT.format(model_id="fr_FR-rhasspy"),
        ),
    ]
}
