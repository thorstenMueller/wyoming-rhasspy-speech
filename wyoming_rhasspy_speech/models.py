from dataclasses import dataclass
from typing import Dict, Optional

URL_FORMAT = "http://localhost:8000/{model_id}.tar.gz"


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
