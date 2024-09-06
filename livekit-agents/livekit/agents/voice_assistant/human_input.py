from __future__ import annotations

import asyncio
import json
from typing import List, Literal

from livekit import rtc

from .. import stt as speech_to_text
from .. import transcription, utils
from .. import vad as voice_activity_detection
from .. import llm as large_language_model
from .log import logger

EventTypes = Literal[
    "start_of_speech",
    "vad_inference_done",
    "end_of_speech",
    "final_transcript",
    "interim_transcript",
]


class HumanInput(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        room: rtc.Room,
        vad: voice_activity_detection.VAD,
        stt: speech_to_text.STT,
        llm: large_language_model.LLM,
        participant: rtc.RemoteParticipant,
        transcription: bool,
    ) -> None:
        super().__init__()
        self._room, self._vad, self._stt, self._llm, self._participant, self._transcription = (
            room,
            vad,
            stt,
            llm,
            participant,
            transcription,
        )
        self._subscribed_track: rtc.RemoteAudioTrack | None = None
        self._subscribed_track_video: rtc.RemoteVideoTrack | None = None
        self._recognize_atask: asyncio.Task[None] | None = None

        self._closed = False
        self._speaking = False
        self._speech_probability = 0.0

        self._room.on("track_published", self._subscribe_to_microphone)
        self._room.on("track_subscribed", self._subscribe_to_microphone)
        self._subscribe_to_microphone()

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("HumanInput already closed")

        self._closed = True
        self._room.off("track_published", self._subscribe_to_microphone)
        self._room.off("track_subscribed", self._subscribe_to_microphone)
        self._speaking = False

        if self._recognize_atask is not None:
            await utils.aio.gracefully_cancel(self._recognize_atask)

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def speaking_probability(self) -> float:
        return self._speech_probability

    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """
        Subscribe to the participant microphone if found and not already subscribed.
        Do nothing if no track is found.
        """
        print(f'self._participant.track_publications.values() {self._participant.track_publications.values()}')
        # for publication in self._participant.track_publications.values():
        #     print(f'publication.source: {publication.source}')
        #     # SOURCE_CAMERA
        #     if publication.source != rtc.TrackSource.SOURCE_CAMERA:
        #         continue

        #     if not publication.subscribed:
        #         publication.set_subscribed(True)
        #     if (
        #         publication.track is not None
        #         and publication.track != self._subscribed_track_video
        #     ):
        #         self._subscribed_track_video = publication.track  # type: ignore
        #     pass
        for publication in self._participant.track_publications.values():
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                continue

            if not publication.subscribed:
                publication.set_subscribed(True)

            if (
                publication.track is not None
                and publication.track != self._subscribed_track
            ):
                self._subscribed_track = publication.track  # type: ignore
                if self._recognize_atask is not None:
                    self._recognize_atask.cancel()
                
                if self._subscribed_track_video is not None:
                    self._recognize_atask = asyncio.create_task(
                    self._recognize_task(rtc.AudioStream(self._subscribed_track), rtc.VideoStream(self._subscribed_track_video))  # type: ignore
                )
                else:
                    self._recognize_atask = asyncio.create_task(
                    self._recognize_task(rtc.AudioStream(self._subscribed_track), None)  # type: ignore
                )
                    
                break

    @utils.log_exceptions(logger=logger)
    async def _recognize_task(self, audio_stream: rtc.AudioStream, video_stream: rtc.VideoStream) -> None:
        """
        Receive the frames from the user audio stream and detect voice activity.
        """
        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream()

        def _will_forward_transcription(
            fwd: transcription.STTSegmentsForwarder, transcription: rtc.Transcription
        ):
            if not self._transcription:
                transcription.segments = []

            return transcription

        stt_forwarder = transcription.STTSegmentsForwarder(
            room=self._room,
            participant=self._participant,
            track=self._subscribed_track,
            will_forward_transcription=_will_forward_transcription,
        )

        async def _audio_stream_co() -> None:
            # forward the audio stream to the VAD and STT streams
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)
                vad_stream.push_frame(ev.frame)
        
        async def _video_stream_co() -> None:
            # if video_stream is not None:
            #     async for ev in video_stream:
            #         self._llm.push_video(ev.frame)
            #         pass
            async def get_human_video_track(room: rtc.Room):
                track_future = asyncio.Future[rtc.RemoteVideoTrack]()

                def on_sub(track: rtc.Track, *_):
                    if isinstance(track, rtc.RemoteVideoTrack):
                        track_future.set_result(track)

                room.on("track_subscribed", on_sub)

                remote_video_tracks: List[rtc.RemoteVideoTrack] = []
                for _, p in room.remote_participants.items():
                    for _, t_pub in p.track_publications.items():
                        if t_pub.track is not None and isinstance(
                            t_pub.track, rtc.RemoteVideoTrack
                        ):
                            remote_video_tracks.append(t_pub.track)

                if len(remote_video_tracks) > 0:
                    track_future.set_result(remote_video_tracks[0])

                video_track = await track_future
                room.off("track_subscribed", on_sub)
                return video_track
            # ConnectionState.CONN_CONNECTED
            while self._room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                # print('=========================>11')
                video_track = await get_human_video_track(self._room)
                # print('=========================>22')
                async for event in rtc.VideoStream(video_track):
                    self._llm.push_video(event.frame)
                    # print('=========================>33')

        async def _vad_stream_co() -> None:
            TURINGASR_START_MSG: str = json.dumps({"isSpeaking":"start"})
            TURINGASR_END_MSG: str = json.dumps({"isSpeaking":"end"})
            async for ev in vad_stream:
                if ev.type == voice_activity_detection.VADEventType.START_OF_SPEECH:
                    self._speaking = True
                    await stt_stream._turing_asr_send(TURINGASR_START_MSG)
                    self.emit("start_of_speech", ev)
                elif ev.type == voice_activity_detection.VADEventType.INFERENCE_DONE:
                    self._speech_probability = ev.probability
                    self.emit("vad_inference_done", ev)
                elif ev.type == voice_activity_detection.VADEventType.END_OF_SPEECH:
                    await stt_stream._turing_asr_send(TURINGASR_END_MSG)
                    self._speaking = False
                    self.emit("end_of_speech", ev)

        async def _stt_stream_co() -> None:
            async for ev in stt_stream:
                stt_forwarder.update(ev)

                if ev.type == speech_to_text.SpeechEventType.FINAL_TRANSCRIPT:
                    self.emit("final_transcript", ev)
                elif ev.type == speech_to_text.SpeechEventType.INTERIM_TRANSCRIPT:
                    self.emit("interim_transcript", ev)

        tasks = [
            asyncio.create_task(_audio_stream_co()),
            asyncio.create_task(_video_stream_co()),
            asyncio.create_task(_vad_stream_co()),
            asyncio.create_task(_stt_stream_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

            await stt_forwarder.aclose()
            await stt_stream.aclose()
            await vad_stream.aclose()
