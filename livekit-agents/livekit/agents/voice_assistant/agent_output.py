from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterable, Callable, Union

from livekit import rtc

from .. import llm, tokenize, utils
from .. import transcription as agent_transcription
from .. import tts as text_to_speech
from .agent_playout import AgentPlayout, PlayoutHandle
from .log import logger

SpeechSource = Union[AsyncIterable[str], str]


class SynthesisHandle:
    def __init__(
        self,
        *,
        speech_id: str,
        speech_source: SpeechSource,
        agent_playout: AgentPlayout,
        tts: text_to_speech.TTS,
        transcription_fwd: agent_transcription.TTSSegmentsForwarder,
    ) -> None:
        self._speech_source, self._agent_playout, self._tts, self._tr_fwd = (
            speech_source,
            agent_playout,
            tts,
            transcription_fwd,
        )
        self._buf_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._play_handle: PlayoutHandle | None = None
        self._interrupt_fut = asyncio.Future[None]()
        self._speech_id = speech_id

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def tts_forwarder(self) -> agent_transcription.TTSSegmentsForwarder:
        return self._tr_fwd

    @property
    def validated(self) -> bool:
        return self._play_handle is not None

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def play_handle(self) -> PlayoutHandle | None:
        return self._play_handle

    def play(self) -> PlayoutHandle:
        """Validate the speech for playout"""
        if self.interrupted:
            raise RuntimeError("synthesis was interrupted")

        self._play_handle = self._agent_playout.play(
            self._speech_id, self._buf_ch, transcription_fwd=self._tr_fwd
        )
        return self._play_handle

    def interrupt(self) -> None:
        """Interrupt the speech"""
        if self.interrupted:
            return

        logger.debug(
            "interrupting synthesis/playout",
            extra={"speech_id": self.speech_id},
        )

        if self._play_handle is not None:
            self._play_handle.interrupt()

        self._interrupt_fut.set_result(None)


class AgentOutput:
    def __init__(
        self,
        *,
        room: rtc.Room,
        agent_playout: AgentPlayout,
        llm: llm.LLM,
        tts: text_to_speech.TTS,
    ) -> None:
        self._room, self._agent_playout, self._llm, self._tts = (
            room,
            agent_playout,
            llm,
            tts,
        )
        self._tasks = set[asyncio.Task[Any]]()

    @property
    def playout(self) -> AgentPlayout:
        return self._agent_playout

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    def synthesize(
        self,
        *,
        speech_id: str,
        transcript: SpeechSource,
        transcription: bool,
        transcription_speed: float,
        sentence_tokenizer: tokenize.SentenceTokenizer,
        word_tokenizer: tokenize.WordTokenizer,
        hyphenate_word: Callable[[str], list[str]],
    ) -> SynthesisHandle:
        def _will_forward_transcription(
            fwd: agent_transcription.TTSSegmentsForwarder,
            transcription: rtc.Transcription,
        ):
            if not transcription:
                transcription.segments = []

            return transcription

        transcription_fwd = agent_transcription.TTSSegmentsForwarder(
            room=self._room,
            participant=self._room.local_participant,
            speed=transcription_speed,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            will_forward_transcription=_will_forward_transcription,
        )

        handle = SynthesisHandle(
            speech_source=transcript,
            agent_playout=self._agent_playout,
            tts=self._tts,
            transcription_fwd=transcription_fwd,
            speech_id=speech_id,
        )

        task = asyncio.create_task(self._synthesize_task(handle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return handle

    @utils.log_exceptions(logger=logger)
    async def _synthesize_task(self, handle: SynthesisHandle) -> None:
        """Synthesize speech from the source"""
        if isinstance(handle._speech_source, str):
            co = _str_synthesis_task(handle._speech_source, handle)
        else:
            co = _stream_synthesis_task(handle._speech_source, handle)

        synth = asyncio.create_task(co)
        synth.add_done_callback(lambda _: handle._buf_ch.close())
        try:
            _ = await asyncio.wait(
                [synth, handle._interrupt_fut], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await utils.aio.gracefully_cancel(synth)


@utils.log_exceptions(logger=logger)
async def _str_synthesis_task(text: str, handle: SynthesisHandle) -> None:
    """synthesize speech from a string"""
    if not handle.tts_forwarder.closed:
        handle.tts_forwarder.push_text(text)
        handle.tts_forwarder.mark_text_segment_end()

    start_time = time.time()
    first_frame = True

    try:
        async for audio in handle._tts.synthesize(text):
            # print(f'_str_synthesis_task: received frame {text}')
            if first_frame:
                first_frame = False
                logger.debug(
                    "received first TTS frame",
                    extra={
                        "speech_id": handle.speech_id,
                        "elapsed": round(time.time() - start_time, 3),
                        "streamed": False,
                    },
                )

            frame = audio.frame

            handle._buf_ch.send_nowait(frame)
            if not handle.tts_forwarder.closed:
                handle.tts_forwarder.push_audio(frame)

    finally:
        if not handle.tts_forwarder.closed:
            handle.tts_forwarder.mark_audio_segment_end()


@utils.log_exceptions(logger=logger)
async def _stream_synthesis_task(
    streamed_text: AsyncIterable[str], handle: SynthesisHandle
) -> None:
    """synthesize speech from streamed text"""

    @utils.log_exceptions(logger=logger)
    async def _read_generated_audio_task():
        start_time = time.time()
        first_frame = True
        async for audio in tts_stream:
            if first_frame:
                first_frame = False
                logger.debug(
                    "first TTS frame",
                    extra={
                        "speech_id": handle.speech_id,
                        "elapsed": round(time.time() - start_time, 3),
                        "streamed": True,
                    },
                )

            if not handle._tr_fwd.closed:
                handle._tr_fwd.push_audio(audio.frame)

            handle._buf_ch.send_nowait(audio.frame)

        if handle._tr_fwd and not handle._tr_fwd.closed:
            # mark_audio_segment_end must be called *after* mart_text_segment_end
            handle._tr_fwd.mark_audio_segment_end()

    # otherwise, stream the text to the TTS
    tts_stream = handle._tts.stream()
    read_atask: asyncio.Task | None = None
    gpt_streamed_text = ''

    try:
        async for seg in streamed_text:
            gpt_streamed_text += seg
            if not handle.tts_forwarder.closed:
                # print(f'==>_stream_synthesis_task tts_forwarder_seg:{seg}')
                handle.tts_forwarder.push_text(seg)

            if read_atask is None:
                # print(f'==>_stream_synthesis_task create read_atask:{seg}')
                # start the task when we receive the first text segment (so start_time is more accurate)
                read_atask = asyncio.create_task(_read_generated_audio_task())
            # print(f'==> _stream_synthesis_task tts_stream_seg:{seg}')
            tts_stream.push_text(seg)

        print(f'_stream_synthesis_task 结束streamed_text {gpt_streamed_text}')
        if not handle.tts_forwarder.closed:
            handle.tts_forwarder.mark_text_segment_end()

        tts_stream.end_input()

        if read_atask is not None:
            await read_atask
    finally:
        if read_atask is not None:
            await utils.aio.gracefully_cancel(read_atask)

        await tts_stream.aclose()
