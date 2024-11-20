#!/usr/bin/env python3
import argparse
import asyncio
import logging
import re
import sqlite3
import time
from functools import partial
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

from pyring_buffer import RingBuffer
from pysilero_vad import SileroVoiceActivityDetector
from pyspeex_noise import AudioProcessor as SpeexAudioProcessor
from rhasspy_speech import KaldiTranscriber
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .edit_distance import edit_distance
from .shared import ARPA, GRAMMAR, LANG_TYPES, AppSettings, AppState
from .web_server import get_app

_LOGGER = logging.getLogger()
_DIR = Path(__file__).parent

RATE = 16000
WIDTH = 2
CHANNELS = 1
BYTES_10MS = 320


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--train-dir", required=True, help="Directory to write trained model files"
    )
    parser.add_argument(
        "--tools-dir", required=True, help="Directory with kaldi, openfst, etc."
    )
    parser.add_argument(
        "--models-dir", required=True, help="Directory with speech models"
    )
    parser.add_argument(
        "--hass-token", help="Long-lived access token for Home Assistant"
    )
    parser.add_argument(
        "--hass-websocket-uri",
        default="ws://homeassistant.local:8123/api/websocket",
        help="URI of Home Assistant websocket API",
    )
    parser.add_argument(
        "--hass-ingress",
        action="store_true",
        help="Web server is behind Home Assistant ingress proxy",
    )
    parser.add_argument("--web-server-host", default="localhost")
    parser.add_argument("--web-server-port", type=int, default=8099)
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    state = AppState(
        settings=AppSettings(
            train_dir=Path(args.train_dir),
            tools_dir=Path(args.tools_dir),
            models_dir=Path(args.models_dir),
            hass_token=args.hass_token,
            hass_websocket_uri=args.hass_websocket_uri,
            hass_ingress=args.hass_ingress,
        )
    )

    _LOGGER.info("Ready")

    # Run Flask server in a separate thread
    flask_app = get_app(state)
    Thread(
        target=flask_app.run,
        kwargs={
            "host": args.web_server_host,
            "port": args.web_server_port,
            "debug": args.debug,
            "use_reloader": False,
        },
        daemon=True,
    ).start()

    wyoming_server = AsyncServer.from_uri(args.uri)

    try:
        await wyoming_server.run(partial(RhasspySpeechEventHandler, args, state))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class RhasspySpeechEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        state: AppState,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.client_id = str(time.monotonic_ns())
        self.converter = AudioChunkConverter(rate=RATE, width=WIDTH, channels=CHANNELS)

        self.model_id: Optional[str] = None
        self.state = state
        self.audio_queues: Dict[str, asyncio.Queue[Optional[bytes]]] = {
            lang_type: asyncio.Queue() for lang_type in LANG_TYPES
        }
        self.transcribe_tasks: Dict[str, asyncio.Task] = {}

        self.vad_buffer = bytes()
        self.vad = SileroVoiceActivityDetector()
        self.vad_bytes_per_chunk = self.vad.chunk_bytes()
        self.vad_threshold = 0.5
        self.before_speech_seconds = 0.7
        self.before_speech_buffer = RingBuffer(
            int(self.before_speech_seconds * RATE * WIDTH * CHANNELS)
        )
        self.is_speech_started = False

        self.speex = SpeexAudioProcessor(4000, -30)
        self.speex_audio_buffer = bytes()

        _LOGGER.debug("Client connected: %s", self.client_id)

    async def _audio_stream(self, lang_type: str):
        audio_queue = self.audio_queues[lang_type]
        while True:
            chunk = await audio_queue.get()
            if not chunk:
                break

            yield chunk

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.get_info().event())
            _LOGGER.debug("Sent info to client: %s", self.client_id)
            return True

        if AudioStart.is_type(event.type):
            if not self.model_id:
                _LOGGER.error("No model selected")
                return False

            # Cancel running tasks
            for lang_type, transcribe_task in self.transcribe_tasks.values():
                self.audio_queues[lang_type].put_nowait(None)
                transcribe_task.cancel()
            self.transcribe_tasks.clear()

            self.vad.reset()
            self.vad_buffer = bytes()
            self.before_speech_buffer = RingBuffer(
                int(self.before_speech_seconds * RATE * WIDTH * CHANNELS)
            )
            self.is_speech_started = False
            self.speex_audio_buffer = bytes()

            await self.state.transcribers_lock.acquire()
            try:
                transcribers = self.state.transcribers.get(self.model_id)
                if transcribers is None:
                    transcribers = {
                        lang_type: KaldiTranscriber(
                            model_dir=self.state.settings.models_dir
                            / self.model_id
                            / "model",
                            graph_dir=self.state.settings.train_dir
                            / self.model_id
                            / f"graph_{lang_type}",
                            kaldi_bin_dir=self.state.settings.tools_dir
                            / "kaldi"
                            / "bin",
                        )
                        for lang_type in LANG_TYPES
                    }
                    self.state.transcribers[self.model_id] = transcribers

                _LOGGER.debug("Starting transcribers")

                self.audio_queues = {
                    lang_type: asyncio.Queue() for lang_type in LANG_TYPES
                }
                self.transcribe_tasks = {
                    lang_type: asyncio.create_task(
                        transcribers[lang_type].transcribe_stream_async(
                            self._audio_stream(lang_type), RATE, WIDTH, CHANNELS
                        )
                    )
                    for lang_type in LANG_TYPES
                }
            except Exception:
                self.state.transcribers_lock.release()
                raise
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            chunk = self.converter.convert(chunk)

            if self.is_speech_started:
                # Clean audio with speex
                self.speex_audio_buffer += chunk.audio
                clean_audio = bytes()
                audio_idx = 0
                while (audio_idx + BYTES_10MS) < len(self.speex_audio_buffer):
                    clean_audio += self.speex.Process10ms(
                        self.speex_audio_buffer[audio_idx : audio_idx + BYTES_10MS]
                    ).audio
                    audio_idx += BYTES_10MS

                self.speex_audio_buffer = self.speex_audio_buffer[audio_idx:]
                if clean_audio:
                    for audio_queue in self.audio_queues.values():
                        audio_queue.put_nowait(clean_audio)
            else:
                self.before_speech_buffer.put(chunk.audio)

                # Detect start of speech with silero VAD
                self.vad_buffer += chunk.audio
                while len(self.vad_buffer) >= self.vad_bytes_per_chunk:
                    vad_chunk = self.vad_buffer[: self.vad_bytes_per_chunk]
                    speech_prob = self.vad.process_chunk(vad_chunk)
                    if speech_prob > self.vad_threshold:
                        self.is_speech_started = True

                        # Buffered audio will be cleaned when next chunk arrives
                        self.speex_audio_buffer += self.before_speech_buffer.getvalue()
                        break

                    self.vad_buffer = self.vad_buffer[self.vad_bytes_per_chunk :]

        elif AudioStop.is_type(event.type):
            assert self.model_id
            start_time = time.monotonic()

            try:
                texts = {}
                for lang_type in LANG_TYPES:
                    self.audio_queues[lang_type].put_nowait(None)
                    texts[lang_type] = await self.transcribe_tasks[lang_type]
            finally:
                self.state.transcribers_lock.release()

            _LOGGER.debug(
                "Transcripts for client %s in %s second(s): %s",
                self.client_id,
                time.monotonic() - start_time,
                texts,
            )

            text = ""
            # text_arpa, text_grammar = f" {texts[ARPA]} ", f" {texts[GRAMMAR]} "
            text_arpa, text_grammar = texts[ARPA].strip(), texts[GRAMMAR].strip()

            if text_arpa == text_grammar:
                text = text_grammar
            elif text_arpa or text_grammar:
                skip_words = self.state.skip_words.get(self.model_id, [])
                words_arpa = text_arpa.split()
                words_grammar = text_grammar.split()
                distance = edit_distance(
                    words_arpa, words_grammar, skip_words=skip_words
                )
                norm_distance = distance / max(len(words_arpa), len(words_grammar))
                if norm_distance < 0.5:
                    text = text_grammar
                else:
                    distance = edit_distance(text_arpa, text_grammar)
                    norm_distance = distance / max(len(text_arpa), len(text_grammar))
                    if norm_distance < 0.5:
                        text = text_grammar

            # Get output text
            with sqlite3.Connection(
                self.state.settings.train_dir / self.model_id / "sentences.db"
            ) as conn:
                cur = conn.execute(
                    "SELECT output FROM sentences WHERE input = ? LIMIT 1", (text,)
                )
                for row in cur:
                    text = row[0]
                    _LOGGER.debug("Output text: %s", text)
                    break

            await self.write_event(Transcript(text=text).event())

            return True
        elif Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.name:
                self.model_id = transcribe.name
            elif transcribe.language:
                for model in sorted(
                    self.get_info().asr[0].models,
                    key=lambda m: len(m.name),
                    reverse=True,
                ):
                    if model.name.startswith(transcribe.language):
                        self.model_id = model.name
                        break
        else:
            _LOGGER.debug("Unexpected event: type=%s, data=%s", event.type, event.data)

        return True

    async def disconnect(self) -> None:
        _LOGGER.debug("Client disconnected: %s", self.client_id)

    def get_info(self) -> Info:
        models: List[AsrModel] = []
        for model_dir in self.state.settings.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = model_dir.name
            trained_model_dir = self.state.settings.train_dir / model_id
            if not trained_model_dir.is_dir():
                _LOGGER.warning("Skipping model %s (not trained)", model_id)
                continue

            language = model_id.split("-")[0]
            models.append(
                AsrModel(
                    name=model_id,
                    description=model_id,
                    attribution=Attribution(name="", url=""),
                    installed=True,
                    version=None,
                    languages=[language],
                )
            )

        if not models:
            _LOGGER.warning("No trained models found.")

        return Info(
            asr=[
                AsrProgram(
                    name="rhasspy-speech",
                    description="A fixed input speech-to-text system based on Kaldi",
                    attribution=Attribution(
                        name="synesthesiam",
                        url="https://github.com/synesthesiam/rhasspy-speech",
                    ),
                    installed=True,
                    version="1.0.0",
                    models=models,
                )
            ],
        )


def remove_skip_word(text: str, skip_word: str) -> str:
    skip_word_pattern = re.compile(
        r"(?<=\W)(" + re.escape(skip_word) + r")(?=\W)",
        re.IGNORECASE,
    )
    text = skip_word_pattern.sub(" ", f" {text} ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
