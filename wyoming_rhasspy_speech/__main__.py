#!/usr/bin/env python3
import argparse
import array
import asyncio
import logging
import tempfile
import time
import wave
from collections import defaultdict
from functools import partial
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple

from pyring_buffer import RingBuffer
from pysilero_vad import SileroVoiceActivityDetector
from pyspeex_noise import AudioProcessor as SpeexAudioProcessor
from rhasspy_speech.const import LangSuffix
from rhasspy_speech.tools import KaldiTools
from rhasspy_speech.transcribe_stream import KaldiNnet3StreamTranscriber
from rhasspy_speech.transcribe_wav import KaldiNnet3WavTranscriber
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from .shared import AppSettings, AppState
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
    # Audio
    parser.add_argument("--volume-multiplier", type=float, default=1.0)
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
        "--speex", action="store_true", help="Enable audio cleaning with Speex"
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
    parser.add_argument("--norm-distance-threshold", type=float, default=0.15)
    # Transcribers
    parser.add_argument("--max-active", type=int, default=7000)
    parser.add_argument("--lattice-beam", type=float, default=8.0)
    parser.add_argument("--acoustic-scale", type=float, default=0.5)
    parser.add_argument("--beam", type=float, default=24.0)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    #
    parser.add_argument(
        "--decode-mode",
        choices=("grammar", "arpa", "arpa_rescore"),
        default="arpa",
    )
    parser.add_argument("--arpa-rescore-order", type=int, default=5)
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
            # Audio
            volume_multiplier=args.volume_multiplier,
            # VAD
            vad_enabled=(not args.no_vad),
            vad_threshold=args.vad_threshold,
            before_speech_seconds=args.before_speech_seconds,
            # Speex
            speex_enabled=args.speex,
            speex_noise_suppression=args.speex_noise_suppression,
            speex_auto_gain=args.speex_auto_gain,
            # Edit distance
            norm_distance_threshold=args.norm_distance_threshold,
            # Transcribers
            max_active=args.max_active,
            lattice_beam=args.lattice_beam,
            acoustic_scale=args.acoustic_scale,
            beam=args.beam,
            nbest=args.nbest,
            streaming=args.streaming,
            #
            decode_mode=LangSuffix(args.decode_mode),
            arpa_rescore_order=args.arpa_rescore_order,
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
        self.model_suffix: Optional[str] = None
        self.model_train_dir: Optional[Path] = None
        self.model_data_dir: Optional[Path] = None
        self.state = state
        self.transcriber: Optional[KaldiNnet3WavTranscriber] = None
        self.transcribe_task: Optional[asyncio.Task] = None

        settings = self.state.settings

        self.is_streaming = self.state.settings.streaming
        if self.state.settings.decode_mode == LangSuffix.GRAMMAR:
            # Strict grammar
            self.graph_dir_name = "graph_grammar"
        else:
            # Language model
            self.graph_dir_name = "graph_arpa"

        # Non-streaming
        self.audio_buffer = bytes()

        # Streaming
        self.audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()

        # Audio
        self.volume_multiplier: Optional[float] = None
        if settings.volume_multiplier != 1.0:
            self.volume_multiplier = settings.volume_multiplier

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

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.get_info().event())
            return True

        if AudioStart.is_type(event.type):
            if not self.model_id:
                _LOGGER.error("No model selected")
                return False

            self.model_train_dir = self.state.settings.model_train_dir(
                self.model_id, self.model_suffix
            )
            self.model_data_dir = self.state.settings.model_data_dir(self.model_id)

            # Empty queue
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            if self.is_streaming:
                # Streaming audio
                transcriber = KaldiNnet3StreamTranscriber(
                    model_dir=self.model_data_dir,
                    graph_dir=self.model_train_dir / self.graph_dir_name,
                    tools=KaldiTools.from_tools_dir(self.state.settings.tools_dir),
                    max_active=self.state.settings.max_active,
                    lattice_beam=self.state.settings.lattice_beam,
                    acoustic_scale=self.state.settings.acoustic_scale,
                    beam=self.state.settings.beam,
                )

                if self.state.settings.decode_mode == LangSuffix.ARPA_RESCORE:
                    # Streaming with rescoring
                    self.transcribe_task = asyncio.create_task(
                        transcriber.async_transcribe_rescore(
                            self.audio_stream(),
                            old_lang_dir=self.model_train_dir / "data" / "lang_arpa",
                            new_lang_dir=self.model_train_dir
                            / "data"
                            / "lang_arpa_rescore",
                            nbest=self.state.settings.nbest,
                        )
                    )
                else:
                    # Streaming without rescoring
                    self.transcribe_task = asyncio.create_task(
                        transcriber.async_transcribe(
                            self.audio_stream(), nbest=self.state.settings.nbest
                        )
                    )
            else:
                # Non-streaming
                self.transcriber = KaldiNnet3WavTranscriber(
                    model_dir=self.model_data_dir,
                    graph_dir=self.model_train_dir / self.graph_dir_name,
                    tools=KaldiTools.from_tools_dir(self.state.settings.tools_dir),
                    max_active=self.state.settings.max_active,
                    lattice_beam=self.state.settings.lattice_beam,
                    acoustic_scale=self.state.settings.acoustic_scale,
                    beam=self.state.settings.beam,
                )

            if self.vad is not None:
                # Reset VAD
                self.vad.reset()
                self.vad_buffer = bytes()
                self.before_speech_buffer = RingBuffer(
                    int(self.before_speech_seconds * RATE * WIDTH * CHANNELS)
                )
                self.is_speech_started = False
                self.speex_audio_buffer = bytes()

            self.audio_buffer = bytes()

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            chunk = self.converter.convert(chunk)

            if self.volume_multiplier is not None:
                chunk.audio = multiply_volume(chunk.audio, self.volume_multiplier)

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
                    if self.is_streaming:
                        self.audio_queue.put_nowait(audio_to_transcribe)
                    else:
                        self.audio_buffer += audio_to_transcribe
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
            assert self.model_train_dir is not None
            assert self.model_data_dir is not None

            start_time = time.monotonic()
            texts: List[str] = []

            if self.is_streaming:
                assert self.transcribe_task is not None

                # End stream and get transcript(s)
                self.audio_queue.put_nowait(None)
                texts = await self.transcribe_task
                self.transcribe_task = None
            else:
                assert self.transcriber is not None

                with tempfile.NamedTemporaryFile("wb+", suffix=".wav") as temp_file:
                    wav_path = temp_file.name
                    wav_writer: wave.Wave_write = wave.open(wav_path, "wb")
                    with wav_writer:
                        wav_writer.setframerate(16000)
                        wav_writer.setsampwidth(2)
                        wav_writer.setnchannels(1)
                        wav_writer.writeframes(self.audio_buffer)

                    if self.state.settings.decode_mode == LangSuffix.ARPA_RESCORE:
                        texts = await self.transcriber.async_transcribe_rescore(
                            wav_path,
                            old_lang_dir=self.model_train_dir / "data" / "lang_arpa",
                            new_lang_dir=self.model_train_dir
                            / "data"
                            / "lang_arpa_rescore",
                            nbest=self.state.settings.nbest,
                        )
                    else:
                        texts = await self.transcriber.async_transcribe(
                            wav_path, nbest=self.state.settings.nbest
                        )

                    self.transcriber = None

            _LOGGER.debug(
                "Transcripts for client %s in %s second(s): %s",
                self.client_id,
                time.monotonic() - start_time,
                texts,
            )

            text = ""
            if texts:
                text = texts[0]

            await self.write_event(Transcript(text=text).event())

            return True
        elif Transcribe.is_type(event.type):
            self.model_id, self.model_suffix = None, None
            self.model_data_dir = None
            self.model_train_dir = None

            transcribe = Transcribe.from_event(event)
            _LOGGER.debug(transcribe)

            if transcribe.name:
                name_parts = transcribe.name.split("/", maxsplit=1)
                if len(name_parts) == 2:
                    self.model_id, self.model_suffix = name_parts
                else:
                    self.model_id, self.model_suffix = transcribe.name, None
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

    async def audio_stream(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break

            yield chunk

    async def disconnect(self) -> None:
        _LOGGER.debug("Client disconnected: %s", self.client_id)

    def get_info(self) -> Info:
        # [(model_id, suffix)]
        trained_models: List[Tuple[str, Optional[str]]] = []

        for model_dir in self.state.settings.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = model_dir.name
            trained_model_dir = self.state.settings.model_train_dir(model_id)
            if trained_model_dir.is_dir():
                trained_models.append((model_id, None))

            suffixes = self.state.settings.get_suffixes(model_id)
            for suffix in suffixes:
                trained_model_dir = self.state.settings.model_train_dir(
                    model_id, suffix
                )
                if trained_model_dir.is_dir():
                    trained_models.append((model_id, suffix))

        if not trained_models:
            _LOGGER.warning("No trained models found.")

        # program -> language -> (model_id, suffix)
        language_support = defaultdict(dict)
        for model_id, suffix in trained_models:
            # en_US-rhasspy -> rhasspy
            language, program = model_id.split("-", maxsplit=1)
            if suffix:
                program = f"{program}-{suffix}"

            language_support[program][language] = (model_id, suffix)

        return Info(
            asr=[
                AsrProgram(
                    name=program,
                    description="A fixed input speech-to-text system based on Kaldi",
                    attribution=Attribution(
                        name="synesthesiam",
                        url="https://github.com/synesthesiam/rhasspy-speech",
                    ),
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name=model_id
                            if (suffix is None)
                            else f"{model_id}/{suffix}",
                            description=model_id,
                            attribution=Attribution(name="", url=""),
                            installed=True,
                            version=None,
                            languages=[language],
                        )
                        for language, (model_id, suffix) in languages.items()
                    ],
                )
                for program, languages in language_support.items()
            ],
        )


def multiply_volume(chunk: bytes, volume_multiplier: float) -> bytes:
    """Multiplies 16-bit PCM samples by a constant."""

    def _clamp(val: float) -> float:
        """Clamp to signed 16-bit."""
        return max(-32768, min(32767, val))

    return array.array(
        "h",
        (int(_clamp(value * volume_multiplier)) for value in array.array("h", chunk)),
    ).tobytes()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
