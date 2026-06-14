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
from kokoro import KPipeline

# Configure clean logging visibility
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)

# Configuration Constants
RECORDING_SAMPLE_RATE = 16000  # Standard input sample rate for speech models
PLAYBACK_SAMPLE_RATE = 24000   # Native output sample rate for Kokoro-82M
PLAYBACK_CHUNK_SIZE = 1024
VAD_FRAME_SIZE = 480           # 30ms frame size required by WebRTC VAD at 16000Hz

# Isolated path for engine execution caches
RUNTIME_CACHE_DIRECTORY = "/tmp/litert-lm-cache"


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


def capture_user_speech_with_voice_activity_detection(recording_stream, temporary_audio_path: str) -> None:
    """
    Listens continuously for human speech. Uses a hybrid approach combining WebRTC VAD
    with an RMS energy floor to prevent laptop fan noise from locking the loop open.
    """
    speech_detector = webrtcvad.Vad(3) # Maximum aggressiveness against noise
    
    # --- NOISE FLOOR CONFIGURATION ---
    RMS_VOLUME_THRESHOLD = 0.0001
    
    history_ring_buffer = collections.deque(maxlen=30) # ~0.9 seconds of audio history
    is_actively_recording = False
    recorded_voice_frames = []
    silence_chunk_counter = 0
    
    # OPTIMIZATION: Reduced from 55 to 25 (~750ms). Drastically cuts trailing response latency.
    MAXIMUM_ALLOWED_SILENCE_CHUNKS = 25 
    
    print("\n[System]: 🟢 Listening... (Speak naturally, AI will reply when you pause)")
    recording_stream.start_stream()
    
    while True:
        try:
            raw_audio_bytes = recording_stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
            
            # Convert raw bytes to a floating-point array to measure absolute volume
            audio_data = np.frombuffer(raw_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            rms_volume = np.sqrt(np.mean(audio_data ** 2))
            
            # Hybrid check: Must pass WebRTC VAD AND be louder than the laptop's baseline fan noise
            if rms_volume > RMS_VOLUME_THRESHOLD:
                is_current_frame_speech = speech_detector.is_speech(raw_audio_bytes, RECORDING_SAMPLE_RATE)
                print(f"rms={rms_volume:.4f} vad_speech={is_current_frame_speech} silence_count={silence_chunk_counter}", end='\r')
            else:
                is_current_frame_speech = False # Force silence if it's just background hum
            
            if not is_actively_recording:
                history_ring_buffer.append((raw_audio_bytes, is_current_frame_speech))
                
                # Check if we have enough recent speech to trigger recording
                number_of_speech_frames = len([frame for frame, is_speech in history_ring_buffer if is_speech])
                if number_of_speech_frames > 12: # ~0.36 seconds of voice
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


def stream_large_language_model_text_response(conversation_session, audio_file_path: str):
    """Sends the multi-modal audio file to the engine and yields raw text tokens asynchronously."""
    prompt_package = litert_lm.Contents.of(
        "You are participating in a fluid, natural vocal conversation. Keep answers concise.",
        litert_lm.Content.AudioFile(absolute_path=audio_file_path)
    )
    
    print("\nAI Response: ", end="", flush=True)
    for data_chunk in conversation_session.send_message_async(prompt_package):
        text_token = data_chunk["content"][0]["text"]
        yield text_token


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
            for _, _, generated_audio_data in speech_generator:
                if generated_audio_data is not None and len(generated_audio_data) > 0:
                    if hasattr(generated_audio_data, "cpu"):
                        audio_array = generated_audio_data.cpu().numpy()
                    else:
                        audio_array = generated_audio_data
                    
                    raw_pcm_bytes = audio_array.astype(np.float32).tobytes()
                    audio_queue.put(raw_pcm_bytes)
        except Exception as tts_error:
            logging.error(f"TTS synthesis generation fault: {tts_error}")
            
        text_queue.task_done()


def process_text_stream_into_queues(text_generator, text_queue: queue.Queue) -> None:
    """
    Accumulates tokens and extracts chunks eagerly. 
    Uses a highly responsive multi-tier fallback mechanism to feed the TTS pipeline instantly.
    """
    accumulated_text_buffer = ""
    sentence_endings = [".", "?", "!"]
    clause_endings = [",", ";", ":", "—"]
    is_first_chunk = True
    
    for text_token in text_generator:
        accumulated_text_buffer += text_token
        print(text_token, end="", flush=True)
        
        # OPTIMIZATION: Hyper-aggressive chunking for the absolute first phrase 
        # to guarantee an immediate vocal response start.
        if is_first_chunk:
            highest_punctuation_index = -1
            all_delimiters = sentence_endings + clause_endings
            for marker in all_delimiters:
                marker_index = accumulated_text_buffer.find(marker)
                if marker_index != -1 and (highest_punctuation_index == -1 or marker_index < highest_punctuation_index):
                    highest_punctuation_index = marker_index
            
            # Send immediately if minor/major punctuation is found early, OR we hit a clean word boundary at ~60 chars
            if highest_punctuation_index != -1 and highest_punctuation_index > 10:
                first_phrase = accumulated_text_buffer[:highest_punctuation_index + 1].strip()
                accumulated_text_buffer = accumulated_text_buffer[highest_punctuation_index + 1:]
                if first_phrase:
                    text_queue.put(first_phrase)
                    is_first_chunk = False
            elif len(accumulated_text_buffer) > 60:
                last_space_index = accumulated_text_buffer.rfind(" ")
                if last_space_index > 20:
                    first_phrase = accumulated_text_buffer[:last_space_index].strip()
                    accumulated_text_buffer = accumulated_text_buffer[last_space_index:]
                    text_queue.put(first_phrase)
                    is_first_chunk = False
            continue

        # Standard processing pipeline for subsequent text blocks
        highest_sentence_index = -1
        for marker in sentence_endings:
            marker_index = accumulated_text_buffer.rfind(marker)
            if marker_index > highest_sentence_index:
                highest_sentence_index = marker_index
                
        if highest_sentence_index != -1:
            complete_phrase = accumulated_text_buffer[:highest_sentence_index + 1].strip()
            accumulated_text_buffer = accumulated_text_buffer[highest_sentence_index + 1:]
            if complete_phrase:
                text_queue.put(complete_phrase)
                
        # Lowered threshold from 200 to 100 characters for smoother, streaming human cadence
        elif len(accumulated_text_buffer) > 100:
            highest_clause_index = -1
            for marker in clause_endings:
                marker_index = accumulated_text_buffer.rfind(marker)
                if marker_index > highest_clause_index:
                    highest_clause_index = marker_index
                    
            if highest_clause_index != -1:
                complete_phrase = accumulated_text_buffer[:highest_clause_index + 1].strip()
                accumulated_text_buffer = accumulated_text_buffer[highest_clause_index + 1:]
                if complete_phrase:
                    text_queue.put(complete_phrase)
            
            # Absolute fallback safety threshold lowered to 150 characters
            elif len(accumulated_text_buffer) > 150:
                last_space_index = accumulated_text_buffer.rfind(" ")
                if last_space_index > 30:
                    phrasal_chunk = accumulated_text_buffer[:last_space_index].strip()
                    accumulated_text_buffer = accumulated_text_buffer[last_space_index:]
                    text_queue.put(phrasal_chunk)
            
    remaining_text = accumulated_text_buffer.strip()
    if remaining_text:
        text_queue.put(remaining_text)
    print()


def execute_conversation_pipeline(model_path: str) -> None:
    """Orchestrates contexts, initializations, and the core dialogue execution loop."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Target model bundle not found at location: {model_path}")

    print("[System]: Initializing speech pipelines...")
    prepare_environment_directories()
    
    tts_pipeline = KPipeline(lang_code="a")
    audio_manager, recording_stream, playback_stream = setup_audio_hardware()
    
    with litert_lm.Engine(
        model_path=model_path,
        # backend=litert_lm.Backend.GPU(),
        backend=litert_lm.Backend.CPU(),
        audio_backend=litert_lm.Backend.CPU(),
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
    LOCAL_MODEL_FILE = "./gemma-4-E2B-it.litertlm"
    execute_conversation_pipeline(model_path=LOCAL_MODEL_FILE)