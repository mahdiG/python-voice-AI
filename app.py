#!/usr/bin/env python3
"""
Application entry point for a low-latency, voice-to-voice AI pipeline.
Integrates PyAudio input monitoring, Voice Activity Detection, LiteRT-LM processing, and Kokoro TTS streaming.
Uses an adaptive multi-stage background worker architecture to handle long-form streaming text.
"""

import os
import shutil
import logging
import tempfile
import threading
import queue
import collections
import numpy as np
import pyaudio
import soundfile as sf
import litert_lm
import webrtcvad
import torch  
from kokoro import KPipeline

# ==============================================================================
# PIPELINE TUNING & CONFIGURATION REGISTRY
# ==============================================================================

# Hardware Acceleration Toggles
USE_GPU_FOR_LLM = False              # Evaluates the LiteRT-LM core model on graphics hardware
USE_GPU_FOR_TTS = False              # Set to True to pass CUDA execution contexts directly to Kokoro

# Low-Latency Audio Stream Topography
RECORDING_SAMPLE_RATE = 16000        # Input capture rate optimized for human speech processing
PLAYBACK_SAMPLE_RATE = 24000         # Native acoustic output sample delivery rate for Kokoro-82M
PLAYBACK_CHUNK_SIZE = 512            # Frame step dimension for audio driver buffer writing
VAD_FRAME_SIZE = 480                 # Audio slice width (equivalent to exactly 30 milliseconds at 16kHz)

# Dynamic Phrase Slicing Floor Parameters (Latency vs. Human Quality Balance)
MINIMUM_TERMINAL_CHARACTER_LENGTH = 6   # Protects tiny statements like "Huh?" from being diced down
MINIMUM_CLAUSE_CHARACTER_LENGTH = 35    # Prevents isolating structural expressions like "Jarvis," 
MAXIMUM_UNBROKEN_CHARACTER_COUNT = 130  # Safety fallback limit that forces clean whitespace breaks

# Environmental Noise Gate & Silence Configurations
VAD_AGGRESSION_MODE = 3                 # Aggressive noise rejection floor factor for WebRTC VAD (0 to 3)
RMS_VOLUME_THRESHOLD = 0.0001           # Secondary energy baseline gate to ignore room or device fan hum
MAXIMUM_ALLOWED_SILENCE_CHUNKS = 14     # Count of sequential silent frames before executing LLM processing (420ms)

# File System Workspace Locations
LOCAL_MODEL_FILE = "./gemma-4-E2B-it.litertlm"
RUNTIME_CACHE_DIRECTORY = "/tmp/litert-lm-cache"


# ==============================================================================
# APPLICATION SETUP AND SYSTEM RESOURCE ALLOCATION
# ==============================================================================

# Constrain deep learning backends to step cleanly alongside local CPU allocations
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)


def prepare_environment_directories() -> None:
    """Wipes and recreates the isolated cache tree to prevent state corruption."""
    if os.path.exists(RUNTIME_CACHE_DIRECTORY):
        try:
            shutil.rmtree(RUNTIME_CACHE_DIRECTORY)
        except Exception as directory_error:
            logging.warning(f"Could not clear temporary cache: {directory_error}")
            
    os.makedirs(RUNTIME_CACHE_DIRECTORY, exist_ok=True)


def setup_audio_hardware():
    """Initializes and returns the system's core audio recording and playback infrastructure."""
    audio_manager = pyaudio.PyAudio()
    
    recording_stream = audio_manager.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RECORDING_SAMPLE_RATE,
        input=True,
        frames_per_buffer=VAD_FRAME_SIZE,
        start=False  
    )
    
    playback_stream = audio_manager.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=PLAYBACK_SAMPLE_RATE,
        output=True,
        frames_per_buffer=PLAYBACK_CHUNK_SIZE
    )
    
    return audio_manager, recording_stream, playback_stream


# ==============================================================================
# HUMAN VOICE DETECTION AND RECORDING WORKFLOWS
# ==============================================================================

def capture_user_speech_with_voice_activity_detection(recording_stream, temporary_audio_path: str) -> None:
    """
    Listens continuously for human speech. Uses a hybrid approach combining WebRTC VAD
    with an RMS energy floor to prevent laptop fan noise from locking the loop open.
    """
    speech_detector = webrtcvad.Vad(VAD_AGGRESSION_MODE) 
    
    history_ring_buffer = collections.deque(maxlen=30) 
    is_actively_recording = False
    recorded_voice_frames = []
    silence_chunk_counter = 0
    
    print("\n[System]: 🟢 Listening... (Speak naturally, AI will reply when you pause)")
    recording_stream.start_stream()
    
    while True:
        try:
            raw_audio_bytes = recording_stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            rms_volume = np.sqrt(np.mean(audio_data ** 2))
            
            if rms_volume > RMS_VOLUME_THRESHOLD:
                is_current_frame_speech = speech_detector.is_speech(raw_audio_bytes, RECORDING_SAMPLE_RATE)
                print(f"rms={rms_volume:.4f} vad_speech={is_current_frame_speech} silence_count={silence_chunk_counter}", end='\r')
            else:
                is_current_frame_speech = False 
            
            if not is_actively_recording:
                history_ring_buffer.append((raw_audio_bytes, is_current_frame_speech))
                number_of_speech_frames = len([frame for frame, is_speech in history_ring_buffer if is_speech])
                if number_of_speech_frames > 12: 
                    is_actively_recording = True
                    print("[System]: 🔴 Recording active...                      ")
                    for buffer_frame, _ in history_ring_buffer:
                        recorded_voice_frames.append(buffer_frame)
                    history_ring_buffer.clear()
            else:
                recorded_voice_frames.append(raw_audio_bytes)
                if not is_current_frame_speech:
                    silence_chunk_counter += 1
                else:
                    silence_chunk_counter = 0
                    
                if silence_chunk_counter > MAXIMUM_ALLOWED_SILENCE_CHUNKS:
                    print("[System]: ⏸️ Silence detected. Processing...          ")
                    break
        except IOError:
            continue
            
    recording_stream.stop_stream()
    
    if recorded_voice_frames:
        merged_audio_bytes = b''.join(recorded_voice_frames)
        integer_audio_array = np.frombuffer(merged_audio_bytes, dtype=np.int16)
        normalized_audio_signal = integer_audio_array.astype(np.float32) / 32768.0
        sf.write(temporary_audio_path, normalized_audio_signal, RECORDING_SAMPLE_RATE)


# ==============================================================================
# LANGUAGE PROCESSING AND STREAMING TOKEN PARSER
# ==============================================================================

def stream_large_language_model_text_response(conversation_session, audio_file_path: str):
    """Sends the multi-modal audio file to the engine and yields raw text tokens asynchronously."""
    prompt_package = litert_lm.Contents.of(
        "You are participating in a fluid, natural voice conversation. Keep responses brief.",
        litert_lm.Content.AudioFile(absolute_path=audio_file_path)
    )
    
    print("\nAI Response: ", end="", flush=True)
    for data_chunk in conversation_session.send_message_async(prompt_package):
        text_token = data_chunk["content"][0]["text"]
        yield text_token


def process_text_stream_into_queues(text_generator, text_queue: queue.Queue) -> None:
    """
    Accumulates streaming LLM text tokens and breaks them down into natural speech chunks.
    Balances low latency with high vocal quality by enforcing a text floor on sub-clauses
    while allowing clean terminal thoughts to dispatch quickly.
    """
    accumulated_text_buffer = ""
    
    TERMINAL_PUNCTUATION = {".", "?", "!"}
    CLAUSE_PUNCTUATION = {",", ";", ":", "—"}

    for text_token in text_generator:
        accumulated_text_buffer += text_token
        print(text_token, end="", flush=True)
        
        earliest_punctuation_index = -1
        is_terminal_boundary = False
        
        for current_index, character in enumerate(accumulated_text_buffer):
            if character in TERMINAL_PUNCTUATION:
                earliest_punctuation_index = current_index
                is_terminal_boundary = True
                break
            elif character in CLAUSE_PUNCTUATION:
                if current_index >= MINIMUM_CLAUSE_CHARACTER_LENGTH:
                    earliest_punctuation_index = current_index
                    is_terminal_boundary = False
                    break
                    
        if earliest_punctuation_index != -1:
            if is_terminal_boundary and earliest_punctuation_index < MINIMUM_TERMINAL_CHARACTER_LENGTH:
                continue
                
            completed_phrase = accumulated_text_buffer[:earliest_punctuation_index + 1].strip()
            
            if len(completed_phrase) > 1:
                text_queue.put(completed_phrase)
                accumulated_text_buffer = accumulated_text_buffer[earliest_punctuation_index + 1:]
                
        elif len(accumulated_text_buffer) > MAXIMUM_UNBROKEN_CHARACTER_COUNT:
            last_word_boundary_index = accumulated_text_buffer.rfind(" ")
            if last_word_boundary_index > 40:
                completed_phrase = accumulated_text_buffer[:last_word_boundary_index].strip()
                text_queue.put(completed_phrase)
                accumulated_text_buffer = accumulated_text_buffer[last_word_boundary_index:]

    final_remaining_text = accumulated_text_buffer.strip()
    if final_remaining_text:
        text_queue.put(final_remaining_text)
    print()


# ==============================================================================
# AUDIO GENERATION AND HARDWARE EMISSION PIPELINES
# ==============================================================================

def trim_trailing_dead_air_from_audio_signal(audio_signal: np.ndarray, volume_threshold: float = 0.01) -> np.ndarray:
    """Slices off trailing silence from punctuation marks to ensure fragments stitch cleanly."""
    active_sound_indices = np.where(np.abs(audio_signal) > volume_threshold)[0]
    if len(active_sound_indices) > 0:
        last_active_index = active_sound_indices[-1]
        safe_end_index = min(len(audio_signal), last_active_index + 500)
        return audio_signal[:safe_end_index]
    return audio_signal


def synthesize_text_to_audio_worker(tts_pipeline, text_queue: queue.Queue, audio_queue: queue.Queue) -> None:
    """Continuously polls for text fragments, synthesizes them, and forwards raw bytes to playback."""
    while True:
        text_fragment = text_queue.get()
        
        if text_fragment is None:
            audio_queue.put(None)
            text_queue.task_done()
            break
            
        try:
            speech_generator = tts_pipeline(text_fragment, voice="af_heart", speed=1.0)
            for graphemes, phonemes, generated_audio_data in speech_generator:
                if generated_audio_data is not None and len(generated_audio_data) > 0:
                    if hasattr(generated_audio_data, "cpu"):
                        audio_array = generated_audio_data.cpu().numpy()
                    else:
                        audio_array = generated_audio_data
                    
                    optimized_audio_array = trim_trailing_dead_air_from_audio_signal(audio_array)
                    raw_pcm_bytes = optimized_audio_array.astype(np.float32).tobytes()
                    audio_queue.put(raw_pcm_bytes)
        except Exception as tts_error:
            logging.error(f"TTS synthesis generation fault: {tts_error}")
            
        text_queue.task_done()


def playback_hardware_consumer_worker(playback_stream, audio_queue: queue.Queue, playback_active_event: threading.Event) -> None:
    """Plays back generated audio fragments smoothly until a termination signal is received."""
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            audio_queue.task_done()
            break
            
        playback_active_event.set()
        try:
            playback_stream.write(audio_chunk)
        except Exception as playback_error:
            logging.error(f"Error during hardware audio emission: {playback_error}")
            
        audio_queue.task_done()
    playback_active_event.clear()


# ==============================================================================
# PIPELINE COORDINATION ENGINE
# ==============================================================================

def execute_conversation_pipeline(model_path: str) -> None:
    """Orchestrates contexts, initializations, and the core dialogue execution loop."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Target model bundle not found at location: {model_path}")

    torch.set_num_threads(4)

    print("[System]: Initializing speech pipelines...")
    prepare_environment_directories()
    
    # Assign target compute context based directly on the hoisted configuration choice
    target_tts_device = "cuda" if USE_GPU_FOR_TTS else "cpu"
    tts_pipeline = KPipeline(lang_code="a", device=target_tts_device)
    
    audio_manager, recording_stream, playback_stream = setup_audio_hardware()
    
    selected_main_backend = litert_lm.Backend.GPU() if USE_GPU_FOR_LLM else litert_lm.Backend.CPU()
    # gemma doesn't support gpu audio backend
    selected_audio_backend = litert_lm.Backend.CPU()
    # selected_audio_backend = litert_lm.Backend.GPU() if USE_GPU_FOR_MULTIMODAL_AUDIO else litert_lm.Backend.CPU()

    with litert_lm.Engine(
        model_path=model_path,
        backend=selected_main_backend,
        audio_backend=selected_audio_backend,
        enable_speculative_decoding=True,
        cache_dir=RUNTIME_CACHE_DIRECTORY
    ) as litert_engine:
        
        system_rules = [litert_lm.Message.system("You are a short, conversational, and direct voice assistant.")]
        
        with litert_engine.create_conversation(messages=system_rules) as conversation_session:
            print("[System]: Pipeline active. Talk to your AI companion.")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
                temporary_audio_path = temporary_file.name
                
            try:
                while True:
                    capture_user_speech_with_voice_activity_detection(recording_stream, temporary_audio_path)
                    
                    if os.path.getsize(temporary_audio_path) < 1000:
                        continue
                    
                    text_processing_queue = queue.Queue()
                    audio_playback_queue = queue.Queue()
                    playback_active_condition = threading.Event()
                    
                    tts_synthesis_thread = threading.Thread(
                        target=synthesize_text_to_audio_worker,
                        args=(tts_pipeline, text_processing_queue, audio_playback_queue)
                    )
                    tts_synthesis_thread.start()
                    
                    audio_playback_thread = threading.Thread(
                        target=playback_hardware_consumer_worker,
                        args=(playback_stream, audio_playback_queue, playback_active_condition)
                    )
                    audio_playback_thread.start()
                    
                    llm_text_generator = stream_large_language_model_text_response(conversation_session, temporary_audio_path)
                    process_text_stream_into_queues(llm_text_generator, text_processing_queue)
                    
                    text_processing_queue.put(None)
                    
                    tts_synthesis_thread.join()
                    audio_playback_thread.join()
                    
            except KeyboardInterrupt:
                print("\n[System]: Shutting down conversational pipeline smoothly.")
            finally:
                if os.path.exists(temporary_audio_path):
                    os.remove(temporary_audio_path)
                recording_stream.stop_stream()
                recording_stream.close()
                playback_stream.stop_stream()
                playback_stream.close()
                audio_manager.terminate()


if __name__ == "__main__":
    execute_conversation_pipeline(model_path=LOCAL_MODEL_FILE)