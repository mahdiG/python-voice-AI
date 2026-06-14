# LOCAL AUDIO-TO-AUDIO AI PIPELINE

This application implements a low-latency, completely offline voice-to-voice
AI pipeline prototype built to run on standard consumer hardware. It bypasses
traditional multi-model orchestration by feeding raw audio directly to the LLM
backend for speech understanding and reasoning, streaming structural text outputs
cleanly to a downstream TTS pipeline.

This prototype serves as an early architectural exploration for a completely
on-device, serverless AI Life OS, with a long-term goal of porting this pipeline
layout to mobile targets using LiteRT-LM's Kotlin and Flutter APIs.

## 🏗️ ARCHITECTURAL OVERVIEW

Traditional speech agents run an expensive three-model cascade:
STT Engine -> LLM -> TTS Engine. This project eliminates the standalone
Speech-to-Text phase entirely, reducing latency and avoiding data serialization overhead.

[User Audio Input]
│
▼
┌──────────────────────────────────────────────┐
│ Hybrid VAD Layer (WebRTC VAD + RMS Floor) │ ◄── Blocks fan & environmental noise
└──────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ LiteRT-LM Core Engine (Gemma 4 2B Native) │ ◄── Direct audio file ingestion
└──────────────────────────────────────────────┘
│ (Asynchronous Token Stream)
▼
┌──────────────────────────────────────────────┐
│ Dynamic Phrase Slicing Parser │ ◄── High-speed phrase/clause chunking
└──────────────────────────────────────────────┘
│ (Eager Queued Text Blocks)
▼
┌──────────────────────────────────────────────┐
│ Kokoro 82M Speech Synthesis │ ◄── Parallel TTS Worker Thread
└──────────────────────────────────────────────┘
│ (Raw Audio Arrays)
▼
┌──────────────────────────────────────────────┐
│ Acoustic Post-Processor (Signal Stitching) │ ◄── Slices trailing dead air/gaps
└──────────────────────────────────────────────┘
│
▼
[Hardware Audio Driver Output (PyAudio)]

## 🛠️ THE TECH STACK

- Core LLM: Gemma 4 2B (Instruction-tuned, .litertlm layout) optimized for edge deployments.
- Execution Runtime: LiteRT-LM with speculative decoding enabled for low-latency text token prefilling and generation.
- Speech Synthesis: Kokoro 82M (KPipeline) configured for fast, high-fidelity conversational audio generation.
- Audio I/O Framework: PyAudio (PortAudio bindings) for streaming raw system audio input/output.
- Voice Activity Detection: webrtcvad (WebRTC Voice Activity Detector) for hardware-level chunk management.

## ⚡ ENGINEERING & LATENCY OPTIMIZATIONS

1. Native Audio Ingest (STT-Less Layer)
   Because the pipeline uses multi-modal native processing through litert_lm.Content.AudioFile,
   the stack drops an entire heavy transformer layer out of the inference graph. The single LLM
   processes vocal metrics and contextual logic simultaneously.

2. Fan Noise Resilient Hybrid VAD
   When running intense compute loops locally, cooling fans spin up and can trick typical VAD
   thresholds into staying locked "open." This architecture pairs WebRTC VAD (Aggression level 3)
   with an adaptive root-mean-square (RMS) energy floor (RMS_VOLUME_THRESHOLD = 0.0001) to block
   hardware hum from corrupting speech boundaries.

3. Dynamic Look-Ahead Phrase Slicing
   Waiting for an LLM to generate an entire sentence before passing text to the speech pipeline
   ruins conversational pacing. The token parser uses a two-tier streaming pipeline:
   - Terminal Processing: Looks ahead for '.' or '?' but enforces a character floor
     (MINIMUM_TERMINAL_CHARACTER_LENGTH = 6) to protect short conversational statements like
     "Huh?" or "Yeah" from being dicalized into fragments.
   - Clause Slicing: Dispatches mid-sentence clauses immediately when structural punctuation is hit,
     provided they cross a character constraint window, keeping the voice output flowing while
     the model completes downstream decoding steps.

4. Acoustic Signal Stitching & Dead Air Trimming
   Consecutive streaming speech fragments naturally contain trailing silent dead air right after
   a punctuation mark is synthesized. The background execution loop runs generated voice arrays
   through a NumPy signal cropper (trim_trailing_dead_air_from_audio_signal) to crop unvoiced
   signals before they enter the PyAudio hardware consumer queue.

5. Multi-Threaded Queue Topologies
   Hardware blockages are eliminated by decoupling processing domains across concurrent execution units:
   1. Main Execution Loop: Orchestrates user recording loops and handles engine execution.
   2. TTS Synthesis Worker: Asynchronously polls incoming text queues, feeding localized weights into Kokoro.
   3. Hardware Audio Consumer: Feeds continuous paFloat32 streams directly to the system speaker arrays
      without stalling upstream generation blocks.
