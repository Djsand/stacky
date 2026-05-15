from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..danish import spoken_danish_system_prompt


@dataclass(frozen=True)
class PipecatPipelineSpec:
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    idle_timeout_secs: int = 300


class PipecatUnavailable(RuntimeError):
    pass


def build_pipecat_pipeline(
    *,
    transport: Any,
    stt: Any,
    llm: Any,
    tts: Any,
    system_instruction: str,
    spec: PipecatPipelineSpec | None = None,
) -> tuple[Any, Any]:
    """Build the Pipecat pipeline shape Stacky expects.

    The concrete STT/TTS services are intentionally injected because Danish
    quality needs hardware testing. The pipeline itself is fixed: transport
    input, STT, VAD-backed user aggregation, LLM, TTS, transport output, and
    assistant aggregation.
    """

    spec = spec or PipecatPipelineSpec()
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
    except ImportError as exc:
        raise PipecatUnavailable("Install stacky[voice] to build the Pipecat runtime.") from exc

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": system_instruction + "\n\n" + spoken_danish_system_prompt(),
            }
        ]
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=spec.audio_in_sample_rate,
            audio_out_sample_rate=spec.audio_out_sample_rate,
        ),
        idle_timeout_secs=spec.idle_timeout_secs,
    )
    return pipeline, task
