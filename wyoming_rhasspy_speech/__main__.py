#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import re
import sqlite3
import tempfile
import time
import wave
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
from .shared import (
    ARPA,
    GRAMMAR,
    LANG_TYPES,
    AppSettings,
    AppState,
    TranscriberSettings,
)
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
    # Home Assistant
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
    # Web server
    parser.add_argument("--web-server-host", default="localhost")
    parser.add_argument("--web-server-port", type=int, default=8099)
    # VAD
    parser.add_argument(
        "--no-vad", action="store_true", help="Disable voice activity detection"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Threshold for VAD (default: 0.5)",
    )
    parser.add_argument(
        "--before-speech-seconds",
        type=float,
        default=0.7,
        help="Seconds of audio to keep before speech is detected (default: 0.7)",
    )
    # Speex
    parser.add_argument(
        "--no-speex", action="store_true", help="Disable audio cleaning with Speex"
    )
    parser.add_argument(
        "--speex-noise-suppression",
        type=int,
        default=-30,
        help="Noise suppression level (default: -30)",
    )
    parser.add_argument(
        "--speex-auto-gain",
        type=int,
        default=4000,
        help="Auto gain level (default: 4000)",
    )
    # Edit distance
    parser.add_argument("--word-norm-distance-threshold", type=float, default=0.5)
    parser.add_argument("--char-norm-distance-threshold", type=float, default=0.5)
    # Transcribers
    for lang_type in LANG_TYPES:
        parser.add_argument(f"--no-{lang_type}", action="store_true")
        parser.add_argument(f"--no-{lang_type}-streaming", action="store_true")
        parser.add_argument(f"--{lang_type}-max-active", type=int, default=7000)
        parser.add_argument(f"--{lang_type}-lattice-beam", type=float, default=8.0)
        parser.add_argument(f"--{lang_type}-acoustic-scale", type=float, default=1.0)
        parser.add_argument(f"--{lang_type}-beam", type=float, default=24.0)
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    state = AppState(
        settings=AppSettings(
            train_dir=Path(args.train_dir),
            tools_dir=Path(args.tools_dir),
            models_dir=Path(args.models_dir),
            # VAD
            vad_enabled=(not args.no_vad),
            vad_threshold=args.vad_threshold,
            before_speech_seconds=args.before_speech_seconds,
            # Speex
            speex_enabled=(not args.no_speex),
            speex_noise_suppression=args.speex_noise_suppression,
            speex_auto_gain=args.speex_auto_gain,
            # Edit distance
            word_norm_distance_threshold=args.word_norm_distance_threshold,
            char_norm_distance_threshold=args.char_norm_distance_threshold,
            # Transcribers
            transcriber_settings={
                lang_type: TranscriberSettings(
                    is_streaming=(not getattr(args, f"no_{lang_type}_streaming")),
                    max_active=getattr(args, f"{lang_type}_max_active"),
                    lattice_beam=getattr(args, f"{lang_type}_lattice_beam"),
                    acoustic_scale=getattr(args, f"{lang_type}_acoustic_scale"),
                    beam=getattr(args, f"{lang_type}_beam"),
                )
                for lang_type in LANG_TYPES
                if not getattr(args, f"no_{lang_type}")
            },
            # Home Assistant
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
        self.transcribe_tasks: Dict[str, asyncio.Task] = {}

        settings = self.state.settings
        self.streaming_audio_queues: Dict[str, asyncio.Queue[Optional[bytes]]] = {
            lang_type: asyncio.Queue()
            for lang_type, lang_settings in settings.transcriber_settings.items()
            if lang_settings.is_streaming
        }
        self.wav_audio_buffers: Dict[str, bytes] = {
            lang_type: bytes()
            for lang_type, lang_settings in settings.transcriber_settings.items()
            if not lang_settings.is_streaming
        }

        # VAD
        self.vad: Optional[SileroVoiceActivityDetector] = None
        self.vad_bytes_per_chunk: int = 0
        self.vad_buffer = bytes()
        self.vad_threshold = settings.vad_threshold
        self.before_speech_seconds = settings.before_speech_seconds
        self.before_speech_buffer: Optional[RingBuffer] = None
        if settings.vad_enabled:
            self.vad = SileroVoiceActivityDetector()
            self.vad_bytes_per_chunk = self.vad.chunk_bytes()
            self.before_speech_buffer = RingBuffer(
                int(self.before_speech_seconds * RATE * WIDTH * CHANNELS)
            )
        self.is_speech_started = False

        # Speex
        self.speex: Optional[SpeexAudioProcessor] = None
        self.speex_audio_buffer = bytes()
        if settings.speex_enabled:
            self.speex = SpeexAudioProcessor(
                settings.speex_auto_gain, settings.speex_noise_suppression
            )

        _LOGGER.debug("Client connected: %s", self.client_id)

    async def _audio_stream(self, lang_type: str):
        audio_queue = self.streaming_audio_queues[lang_type]
        while True:
            chunk = await audio_queue.get()
            if not chunk:
                break

            yield chunk

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.get_info().event())
            return True

        if AudioStart.is_type(event.type):
            if not self.model_id:
                _LOGGER.error("No model selected")
                return False

            # Cancel running tasks
            for lang_type, transcribe_task in self.transcribe_tasks.values():
                self.streaming_audio_queues[lang_type].put_nowait(None)
                transcribe_task.cancel()
            self.transcribe_tasks.clear()

            if self.vad is not None:
                # Reset VAD
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
                            max_active=lang_settings.max_active,
                            lattice_beam=lang_settings.lattice_beam,
                            acoustic_scale=lang_settings.acoustic_scale,
                            beam=lang_settings.beam,
                        )
                        for lang_type, lang_settings in self.state.settings.transcriber_settings.items()
                    }
                    self.state.transcribers[self.model_id] = transcribers

                _LOGGER.debug("Starting transcribers")

                # Clear queues
                for audio_queue in self.streaming_audio_queues.values():
                    while not audio_queue.empty():
                        audio_queue.get_nowait()

                # Clear WAV buffers
                for lang_type in self.wav_audio_buffers:
                    self.wav_audio_buffers[lang_type] = bytes()

                self.transcribe_tasks = {
                    lang_type: asyncio.create_task(
                        transcribers[lang_type].transcribe_stream_async(
                            self._audio_stream(lang_type), RATE, WIDTH, CHANNELS
                        )
                    )
                    for lang_type in self.streaming_audio_queues
                }
            except Exception:
                self.state.transcribers_lock.release()
                raise
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            chunk = self.converter.convert(chunk)

            if (self.vad is None) or self.is_speech_started:
                if self.speex is not None:
                    # Clean audio with speex
                    self.speex_audio_buffer += chunk.audio
                    audio_to_transcribe = bytes()
                    audio_idx = 0
                    while (audio_idx + BYTES_10MS) < len(self.speex_audio_buffer):
                        audio_to_transcribe += self.speex.Process10ms(
                            self.speex_audio_buffer[audio_idx : audio_idx + BYTES_10MS]
                        ).audio
                        audio_idx += BYTES_10MS

                    self.speex_audio_buffer = self.speex_audio_buffer[audio_idx:]
                else:
                    # Not cleaned
                    if self.speex_audio_buffer:
                        audio_to_transcribe = self.speex_audio_buffer + chunk.audio
                        self.speex_audio_buffer = bytes()
                    else:
                        audio_to_transcribe = chunk.audio

                if audio_to_transcribe:
                    for audio_queue in self.streaming_audio_queues.values():
                        audio_queue.put_nowait(audio_to_transcribe)

                    for lang_type in self.wav_audio_buffers:
                        self.wav_audio_buffers[lang_type] += audio_to_transcribe
            else:
                # VAD
                if self.before_speech_buffer is not None:
                    self.before_speech_buffer.put(chunk.audio)

                # Detect start of speech
                self.vad_buffer += chunk.audio
                while len(self.vad_buffer) >= self.vad_bytes_per_chunk:
                    vad_chunk = self.vad_buffer[: self.vad_bytes_per_chunk]
                    speech_prob = self.vad.process_chunk(vad_chunk)
                    if speech_prob > self.vad_threshold:
                        self.is_speech_started = True

                        # Buffered audio will be cleaned when next chunk arrives
                        if self.before_speech_buffer is not None:
                            self.speex_audio_buffer += (
                                self.before_speech_buffer.getvalue()
                            )

                        break

                    self.vad_buffer = self.vad_buffer[self.vad_bytes_per_chunk :]

        elif AudioStop.is_type(event.type):
            assert self.model_id
            start_time = time.monotonic()

            try:
                # Tell transcribers to stop
                for lang_type in self.streaming_audio_queues:
                    self.streaming_audio_queues[lang_type].put_nowait(None)

                # Gather transcriptions
                fut_to_lang_type: Dict[asyncio.Future, str] = {
                    task: lang_type for lang_type, task in self.transcribe_tasks.items()
                }

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = "/tmp"
                    for lang_type, audio_buffer in self.wav_audio_buffers.items():
                        wav_path = os.path.join(temp_dir, f"{lang_type}.wav")
                        wav_file: wave.Wave_write = wave.open(wav_path, "wb")
                        with wav_file:
                            wav_file.setframerate(RATE)
                            wav_file.setsampwidth(WIDTH)
                            wav_file.setnchannels(CHANNELS)
                            wav_file.writeframes(audio_buffer)

                        task = asyncio.create_task(
                            self.state.transcribers[self.model_id][
                                lang_type
                            ].transcribe_wav_async(wav_path)
                        )
                        fut_to_lang_type[task] = lang_type

                texts: Dict[str, str] = {}
                results = await asyncio.gather(*(fut_to_lang_type.keys()))
                for lang_type, result in zip(fut_to_lang_type.values(), results):
                    texts[lang_type] = result or ""
            finally:
                self.state.transcribers_lock.release()

            _LOGGER.debug(
                "Transcripts for client %s in %s second(s): %s",
                self.client_id,
                time.monotonic() - start_time,
                texts,
            )

            text = ""
            if len(texts) == 1:
                # Only one choice
                text = next(iter(texts.values()), "")
            elif len(texts) > 1:
                # Check ARPA against grammar
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
                    if norm_distance < self.state.settings.word_norm_distance_threshold:
                        text = text_grammar
                    else:
                        distance = edit_distance(text_arpa, text_grammar)
                        norm_distance = distance / max(
                            len(text_arpa), len(text_grammar)
                        )
                        if (
                            norm_distance
                            < self.state.settings.char_norm_distance_threshold
                        ):
                            text = text_grammar

            # Get output text
            with sqlite3.Connection(
                self.state.settings.train_dir / self.model_id / "sentences.db"
            ) as conn:
                cur = conn.execute(
                    "SELECT output FROM sentences WHERE input = ? LIMIT 1", (text,)
                )
                for row in cur:
                    text = row[0].strip()
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
