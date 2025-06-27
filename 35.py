import ctypes
import os
import re
import json
import time
import queue
import threading
import tempfile
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play
import numpy as np
import pvporcupine
import pyaudio
import requests
import sounddevice as sd
import websockets
import asyncio
import webrtcvad
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io.wavfile import write
from kokoro_onnx import Kokoro
import soundfile as sf
from scipy.signal import resample, resample_poly
import random
from textblob import TextBlob
from io import BytesIO
import difflib
from resemblyzer import VoiceEncoder
encoder = VoiceEncoder()
from pyaec import PyAec
import simpleaudio as sa

last_tts_audio = None  # Global buffer to track Buddy's last spoken waveform
last_flavor = None 

# ========== CONFIG & PATHS ==========
WEBRTC_SAMPLE_RATE = 16000
WEBRTC_FRAME_SIZE = 160
WEBRTC_CHANNELS = 1
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
CHIME_PATH = "chime.wav"
known_users_path = "known_users.json"
THEMES_PATH = "themes_memory"
LAST_USER_PATH = "last_user.json"
FASTER_WHISPER_WS = "ws://localhost:9090"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")
KOKORO_VOICES = {"pl": "af_heart", "en": "af_heart", "it": "if_sara"}
KOKORO_LANGS = {"pl": "pl", "en": "en-us", "it": "it"}
DEFAULT_LANG = "en"
FAST_MODE = True
DEBUG = True
DEBUG_MODE = False
BUDDY_BELIEFS_PATH = "buddy_beliefs.json"
LONG_TERM_MEMORY_PATH = "buddy_long_term_memory.json"
PERSONALITY_TRAITS_PATH = "buddy_personality_traits.json"
DYNAMIC_KNOWLEDGE_PATH = "buddy_dynamic_knowledge.json"
ref_audio_buffer = np.zeros(WEBRTC_SAMPLE_RATE * 2, dtype=np.int16)  # 2 seconds
ref_audio_lock = threading.Lock()
current_stream_id = 0
current_audio_playback = None
playback_lock = threading.Lock()
vad_thread_active = threading.Event()
playback_start_time = None

# ========== GLOBAL STATE ==========
aec_instance = PyAec(
    frame_size=160,
    sample_rate=16000
)
aec_reference_queue = queue.Queue(maxsize=50)  # Buffer for precise timing
aec_timing_lock = threading.Lock()
playback_start_time = None
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
os.makedirs(THEMES_PATH, exist_ok=True)
ref_audio_lock = threading.Lock()
tts_queue = queue.Queue()
playback_queue = queue.Queue()
current_playback = None
playback_stop_flag = threading.Event()
buddy_talking = threading.Event()
vad_triggered = threading.Event()
LAST_FEW_BUDDY = []
RECENT_WHISPER = []
known_users = {}
active_speakers = {}
active_speaker_lock = threading.Lock()
full_duplex_interrupt_flag = threading.Event()
full_duplex_vad_result = queue.Queue()
session_emotion_mode = {}
tts_lock = threading.Lock()
playback_lock = threading.Lock()
tts_start_time = 0
BYPASS_AEC = False  
spoken_chunks_cache = set()
vad_thread_running = False

if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Embedding model loaded", flush=True)
    print("Kokoro loaded", flush=True)
    print("Main function entered!", flush=True)

# ========== AUDIO PROCESSING ==========
def update_reference_audio_realtime_precise(pcm, sr, playback_start_time):
    """Precisely timed reference audio injection for AEC"""
    global ref_audio_buffer
    
    try:
        # Ensure 16kHz for AEC
        if sr != 16000:
            pcm_float = pcm.astype(np.float32) / 32768.0
            pcm_16k = resample_poly(pcm_float, 16000, sr)
            pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm_16k = pcm.copy()
        
        # Calculate precise timing
        chunk_size = WEBRTC_FRAME_SIZE  # 160 samples
        frame_duration = chunk_size / 16000  # 0.01 seconds per frame
        
        # Inject frames with PRECISE timing alignment
        current_time = time.time()
        elapsed_from_start = current_time - playback_start_time
        
        for i in range(0, len(pcm_16k), chunk_size):
            frame = pcm_16k[i:i+chunk_size]
            if len(frame) < chunk_size:
                frame = np.pad(frame, (0, chunk_size - len(frame)))
            
            # Calculate when this frame should be injected
            frame_index = i // chunk_size
            target_inject_time = playback_start_time + (frame_index * frame_duration)
            
            # Wait for precise timing
            wait_time = target_inject_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Inject frame with precise timing
            with ref_audio_lock:
                ref_audio_buffer = np.roll(ref_audio_buffer, -chunk_size)
                ref_audio_buffer[-chunk_size:] = frame
                
            if DEBUG:
                print(f"[AEC] Frame {frame_index} injected at {time.time() - playback_start_time:.3f}s")
            
            # Check for interruption
            if full_duplex_interrupt_flag.is_set():
                break
                
    except Exception as e:
        if DEBUG:
            print(f"[AEC] Precise timing error: {e}")

def _play_accumulated_audio_with_aec(audio_chunks):
    """Enhanced playback with perfect AEC timing"""
    global current_audio_playback, playback_start_time
    
    if not audio_chunks:
        return
        
    try:
        # Combine chunks
        combined_audio = []
        target_sr = 16000
        
        for pcm, sr in audio_chunks:
            if sr == target_sr:
                combined_audio.append(pcm)
            else:
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                combined_audio.append((np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16))
        
        if not combined_audio:
            return
            
        smooth_audio = np.concatenate(combined_audio)
        
        # Apply gentle fade to prevent clicks
        fade_samples = min(80, len(smooth_audio) // 20)  # Smaller fade for better AEC
        if len(smooth_audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            smooth_audio[:fade_samples] = (smooth_audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
            
            fade_out = np.linspace(1, 0, fade_samples)
            smooth_audio[-fade_samples:] = (smooth_audio[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
        
        with playback_lock:
            try:
                if DEBUG:
                    print(f"[Buddy][AEC-Playback] Playing {len(smooth_audio)} samples with AEC sync")

                # Set talking flag and start VAD
                if not buddy_talking.is_set():
                    buddy_talking.set()
                    if not vad_thread_active.is_set():
                        threading.Thread(target=background_vad_listener, daemon=True).start()

                # CRITICAL: Record exact playback start time
                playback_start_time = time.time()
                
                # Start precise AEC reference injection IMMEDIATELY
                threading.Thread(
                    target=update_reference_audio_realtime_precise,
                    args=(smooth_audio, target_sr, playback_start_time),
                    daemon=True
                ).start()

                if full_duplex_interrupt_flag.is_set():
                    return

                # Start playback with precise timing
                current_audio_playback = sa.play_buffer(smooth_audio.tobytes(), 1, 2, target_sr)
                
                # Monitor for interruption with minimal delay
                while current_audio_playback.is_playing():
                    if full_duplex_interrupt_flag.is_set():
                        current_audio_playback.stop()
                        break
                    time.sleep(0.002)  # 2ms for ultra-responsive interrupts
                
                current_audio_playback = None
                playback_start_time = None
                
                if DEBUG:
                    print("[Buddy][AEC-Playback] Finished with AEC sync")

            except Exception as e:
                print(f"[Buddy][AEC-Playback ERROR] {e}")
                if current_audio_playback:
                    try:
                        current_audio_playback.stop()
                    except:
                        pass
                    current_audio_playback = None
                playback_start_time = None
            
            finally:
                if playback_queue.empty():
                    buddy_talking.clear()
                        
    except Exception as e:
        print(f"[Buddy][AEC-Playback ERROR] Audio combination error: {e}")

def apply_aec(mic_audio, bypass_aec=False):
    global ref_audio_buffer

    if bypass_aec:
        print("[AEC] Bypassed AEC, returning original mic frame.")
        return mic_audio[:160]

    # Convert mic to float32 [-1.0, 1.0]
    mic_np = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32) / 32768.0
    mic_np = np.clip(mic_np, -1.0, 1.0)

    if len(mic_np) < 160:
        mic_np = np.pad(mic_np, (0, 160 - len(mic_np)))
        print("[AEC] Mic frame padded due to short length.")

    # === Read latest reference frame from buffer ===
    with ref_audio_lock:
        ref_frame = ref_audio_buffer[:160].astype(np.float32) / 32768.0
        if len(ref_frame) < 160:
            ref_frame = np.pad(ref_frame, (0, 160 - len(ref_frame)))
        print("[AEC] Ref input range:", np.min(ref_frame), "to", np.max(ref_frame))

    print("[AEC] Mic input range:", np.min(mic_np), "to", np.max(mic_np))

    # === Abort AEC if ref too silent ===
    if np.abs(ref_frame).max() < 0.01:
        print("[AEC] Ref frame too quiet, skipping AEC.")
        return (mic_np[:160] * 32767).astype(np.int16)

    # === Diagnostics before AEC ===
    sim_before = np.dot(mic_np[:160], ref_frame) / (np.linalg.norm(mic_np[:160]) * np.linalg.norm(ref_frame) + 1e-6)
    print(f"[AEC DIAG] Mic-Ref Similarity BEFORE AEC: {sim_before:.3f}")

    # === Process with PyAEC ===
    aec_instance.set_ref(ref_frame.tolist())
    output = aec_instance.process_with_ref(mic_np[:160].tolist())

    # === Diagnostics after AEC ===
    output_np = np.array(output, dtype=np.float32)
    sim_after = np.dot(output_np, ref_frame) / (np.linalg.norm(output_np) * np.linalg.norm(ref_frame) + 1e-6)
    print(f"[AEC DIAG] Output-Ref Similarity AFTER AEC: {sim_after:.3f}")

    rms_mic = np.sqrt(np.mean(mic_np ** 2))
    rms_ref = np.sqrt(np.mean(ref_frame ** 2))
    rms_out = np.sqrt(np.mean(output_np ** 2))
    print(f"[AEC DIAG] RMS: Mic={rms_mic:.4f}, Ref={rms_ref:.4f}, Out={rms_out:.4f}")

    print("[AEC] Output range:", np.min(output_np), "to", np.max(output_np))

    # === Clip and convert back ===
    output_np = np.clip(output_np, -1.0, 1.0)
    output_int16 = (output_np * 32767).astype(np.int16)

    return output_int16

def is_echo(text):
    """Check if text is likely an echo of Buddy's recent speech"""
    if not text or len(text.strip()) < 3:
        return False

    cleaned = re.sub(r'[^\w\s]', '', text.strip().lower())
    if not cleaned or len(cleaned.split()) < 2:
        return False

    # Check against recent Buddy responses
    for prev in LAST_FEW_BUDDY[-3:]:
        if not prev:
            continue
            
        prev_clean = re.sub(r'[^\w\s]', '', prev.strip().lower())
        if not prev_clean:
            continue
            
        # Calculate similarity
        ratio = difflib.SequenceMatcher(None, cleaned, prev_clean).ratio()
        word_diff = abs(len(cleaned.split()) - len(prev_clean.split()))

        if ratio > 0.87 and word_diff <= 4:
            if DEBUG:
                print(f"[Buddy] Skipping echo:\n→ new:  {cleaned}\n→ prev: {prev_clean}\n→ sim={ratio:.2f}, word_diff={word_diff}")
            return True

    # Check for common Buddy phrases that shouldn't come from user
    buddy_phrases = [
        "i'm just a program", "i'm buddy", "digital fella", 
        "no bugs in my code", "well-oiled machine", "considering i'm just",
        "as an ai", "i'm an ai", "i don't have", "i can't feel",
        "sorry, i'm having trouble", "i'm thinking", "give me a moment"
    ]
    
    for phrase in buddy_phrases:
        if phrase in cleaned:
            if DEBUG:
                print(f"[Buddy] Detected Buddy phrase in user input: '{phrase}'")
            return True

    return False

from numpy.linalg import norm
import numpy as np

def is_echo_of_last_tts(mic_audio, last_tts_audio, threshold=0.75):
    if last_tts_audio is None or len(last_tts_audio) < 160:
        print("[EchoFilter] No valid last_tts_audio available.")
        return False

    if mic_audio.shape != last_tts_audio.shape:
        min_len = min(len(mic_audio), len(last_tts_audio))
        mic_audio = mic_audio[:min_len]
        last_tts_audio = last_tts_audio[:min_len]

    # Convert to float32 for stable math
    mic_audio = mic_audio.astype(np.float32)
    last_tts_audio = last_tts_audio.astype(np.float32)

    # Normalize both signals
    mic_norm = np.max(np.abs(mic_audio)) + 1e-6
    tts_norm = np.max(np.abs(last_tts_audio)) + 1e-6
    mic_audio /= mic_norm
    last_tts_audio /= tts_norm

    # Compute cosine similarity
    dot = np.dot(mic_audio, last_tts_audio)
    denom = norm(mic_audio) * norm(last_tts_audio) + 1e-6
    correlation = dot / denom

    print(f"[EchoFilter] Cosine similarity with last TTS: {correlation:.3f}")
    return correlation > threshold

def is_noise_or_gibberish(text):
    """
    Reject input if it's likely noise, gibberish, or too short.
    """
    if not text or len(text.strip()) < 2:
        return True
    words = text.strip().split()
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    # Reject if it's a single short word or strange characters
    if len(words) < 2 and avg_len < 4:
        return True
    if re.search(r'[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ ]', text):
        return False  # symbols are allowed
    return False

def update_reference_audio_realtime(pcm, sr):
    """Enhanced reference audio injection with perfect timing"""
    global ref_audio_buffer, playback_start_time
    
    try:
        # Ensure 16kHz for AEC compatibility
        if sr != 16000:
            pcm_float = pcm.astype(np.float32) / 32768.0
            pcm_16k = resample_poly(pcm_float, 16000, sr)
            pcm_16k = (np.clip(pcm_16k, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm_16k = pcm.copy()
        
        # Calculate precise frame timing
        chunk_size = WEBRTC_FRAME_SIZE  # 160 samples
        frame_duration = chunk_size / 16000  # 0.01 seconds per frame
        
        # Get playback start time for synchronization
        if playback_start_time is None:
            playback_start_time = time.time()
        
        # Inject frames with REAL-TIME timing
        for i in range(0, len(pcm_16k), chunk_size):
            if full_duplex_interrupt_flag.is_set():
                break
                
            frame = pcm_16k[i:i+chunk_size]
            if len(frame) < chunk_size:
                frame = np.pad(frame, (0, chunk_size - len(frame)))
            
            # Calculate when this frame should be available
            frame_index = i // chunk_size
            target_time = playback_start_time + (frame_index * frame_duration)
            
            # Wait for precise timing alignment
            current_time = time.time()
            wait_time = target_time - current_time
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Update reference buffer with thread safety
            with ref_audio_lock:
                # Shift buffer and add new frame
                ref_audio_buffer = np.roll(ref_audio_buffer, -chunk_size)
                ref_audio_buffer[-chunk_size:] = frame
            
            if DEBUG:
                actual_time = time.time()
                timing_error = actual_time - target_time
                print(f"[AEC-REF] Frame {frame_index}: target={target_time:.3f}, actual={actual_time:.3f}, error={timing_error*1000:.1f}ms")
            
    except Exception as e:
        if DEBUG:
            print(f"[AEC-REF] Error: {e}")
    finally:
        if 'playback_start_time' in globals():
            playback_start_time = None

def _play_accumulated_audio_with_perfect_aec(audio_chunks):
    """Your existing playback but with perfect AEC timing"""
    global current_audio_playback, playback_start_time
    
    if not audio_chunks:
        return
        
    try:
        # Combine chunks (keep your existing logic)
        combined_audio = []
        target_sr = 16000
        
        for pcm, sr in audio_chunks:
            if sr == target_sr:
                combined_audio.append(pcm)
            else:
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                combined_audio.append((np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16))
        
        if not combined_audio:
            return
            
        smooth_audio = np.concatenate(combined_audio)
        
        # Apply gentle fade (keep existing)
        fade_samples = min(80, len(smooth_audio) // 20)
        if len(smooth_audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            smooth_audio[:fade_samples] = (smooth_audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
            
            fade_out = np.linspace(1, 0, fade_samples)
            smooth_audio[-fade_samples:] = (smooth_audio[-fade_samples:].astype(np.float32) * fade_out).astype(np.int16)
        
        with playback_lock:
            try:
                if DEBUG:
                    print(f"[Buddy][AEC-SYNC] Playing {len(smooth_audio)} samples with perfect timing")

                # Set flags
                if not buddy_talking.is_set():
                    buddy_talking.set()
                    if not vad_thread_active.is_set():
                        threading.Thread(target=background_vad_listener, daemon=True).start()

                # CRITICAL: Set exact playback start time BEFORE any audio operations
                playback_start_time = time.time() + 0.05  # Small buffer for system latency
                
                # Start reference audio injection thread IMMEDIATELY
                ref_thread = threading.Thread(
                    target=update_reference_audio_realtime,
                    args=(smooth_audio, target_sr),
                    daemon=True
                )
                ref_thread.start()

                # Small delay to let reference thread initialize
                time.sleep(0.01)

                if full_duplex_interrupt_flag.is_set():
                    return

                # Start playback at the EXACT planned time
                actual_start = time.time()
                current_audio_playback = sa.play_buffer(smooth_audio.tobytes(), 1, 2, target_sr)
                
                if DEBUG:
                    timing_error = actual_start - playback_start_time
                    print(f"[AEC-SYNC] Playback timing error: {timing_error*1000:.1f}ms")
                
                # Monitor with minimal delay
                while current_audio_playback.is_playing():
                    if full_duplex_interrupt_flag.is_set():
                        current_audio_playback.stop()
                        break
                    time.sleep(0.002)
                
                current_audio_playback = None
                
                # Wait for reference thread to complete
                ref_thread.join(timeout=1.0)
                
                if DEBUG:
                    print("[Buddy][AEC-SYNC] Playback and reference sync complete")

            except Exception as e:
                print(f"[Buddy][AEC-SYNC ERROR] {e}")
                if current_audio_playback:
                    try:
                        current_audio_playback.stop()
                    except:
                        pass
                    current_audio_playback = None
            
            finally:
                if playback_queue.empty():
                    buddy_talking.clear()
                        
    except Exception as e:
        print(f"[Buddy][AEC-SYNC ERROR] Audio combination error: {e}")

# ========== BACKGROUND VAD LISTENER (FULL-DUPLEX) ==========
vad_thread_active = threading.Event()

def background_vad_listener():
    """Ultra-responsive VAD for instant interruptions"""
    global vad_thread_active
    
    if vad_thread_active.is_set():
        return
        
    vad_thread_active.set()
    vad = webrtcvad.Vad(1)  # High sensitivity for quick interrupts
    
    try:
        with sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=WEBRTC_FRAME_SIZE,
            device=MIC_DEVICE_INDEX,
        ) as stream:
            
            if DEBUG:
                print("[Buddy][VAD] 🎧 Ultra-responsive monitoring started")
            
            consecutive_speech = 0
            min_speech_frames = 2  # Just 2 frames for instant response!
            silence_counter = 0
            
            while buddy_talking.is_set() and not full_duplex_interrupt_flag.is_set():
                try:
                    data, _ = stream.read(WEBRTC_FRAME_SIZE)
                    
                    # Quick downsample
                    downsampled = resample_poly(data.flatten(), 16000, MIC_SAMPLE_RATE).astype(np.int16)
                    
                    if len(downsampled) >= WEBRTC_FRAME_SIZE:
                        frame = downsampled[:WEBRTC_FRAME_SIZE]
                        
                        # Quick AEC
                        cleaned_frame = apply_aec(frame.tobytes(), bypass_aec=BYPASS_AEC)
                        
                        # Volume check for real speech
                        frame_np = np.frombuffer(cleaned_frame, dtype=np.int16)
                        volume = np.abs(frame_np).mean()
                        
                        if volume > 600:  # Significant speech volume
                            if vad.is_speech(cleaned_frame, 16000):
                                consecutive_speech += 1
                                silence_counter = 0
                                
                                if consecutive_speech >= min_speech_frames:
                                    print(f"\n[Buddy][VAD] ⚡ INSTANT INTERRUPT! (volume: {volume})")
                                    full_duplex_interrupt_flag.set()
                                    stop_playback()
                                    return
                            else:
                                consecutive_speech = max(0, consecutive_speech - 1)
                                silence_counter += 1
                        else:
                            consecutive_speech = 0
                            silence_counter += 1
                            
                        # Auto-exit if too much silence (Buddy finished talking)
                        if silence_counter > 100:  # ~2 seconds of silence
                            if DEBUG:
                                print("[Buddy][VAD] Auto-exit due to silence")
                            break
                            
                except Exception:
                    break
                    
                # Minimal sleep for maximum responsiveness
                time.sleep(0.001)

    except Exception as e:
        if DEBUG:
            print(f"[Buddy][VAD ERROR] {e}")
    finally:
        vad_thread_active.clear()
        if DEBUG:
            print("[Buddy][VAD] 🔇 Monitoring stopped")

# ========== MULTI-SPEAKER DETECTION ==========
def detect_active_speaker(audio_chunk):
    embedding = generate_embedding_from_audio(audio_chunk)
    best_name, best_score = match_known_user(embedding)
    if best_name and best_score > 0.8:
        with active_speaker_lock:
            active_speakers[threading.get_ident()] = best_name
    return best_name, best_score

def generate_embedding_from_audio(audio_np):
    """
    Generate a speaker embedding from a numpy waveform (16kHz, mono, int16 or float32).
    """
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32) / 32768.0  # Normalize from int16 to float32
    
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]  # ensure mono

    return encoder.embed_utterance(audio_np)


def assign_turn_per_speaker(audio_chunk):
    name, score = detect_active_speaker(audio_chunk)
    if name:
        print(f"[Buddy][Multi-Speaker] Speaker switched to: {name} (score={score:.2f})")
        return name
    return None

# ========== MEMORY HELPERS ==========
def get_user_memory_path(name):
    return f"user_memory_{name}.json"

def load_user_memory(name):
    path = get_user_memory_path(name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_memory(name, memory):
    path = get_user_memory_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def update_user_memory(name, utterance):
    memory = load_user_memory(name)
    text = utterance.lower()
    if re.search(r"\bi('?m| am| feel) sad\b", text):
        memory["mood"] = "sad"
    elif re.search(r"\bi('?m| am| feel) happy\b", text):
        memory["mood"] = "happy"
    elif re.search(r"\bi('?m| am| feel) (angry|mad|upset)\b", text):
        memory["mood"] = "angry"
    if re.search(r"\bi (love|like|enjoy|prefer) (marvel movies|marvel|comics)\b", text):
        hobbies = memory.get("hobbies", [])
        if "marvel movies" not in hobbies:
            hobbies.append("marvel movies")
        memory["hobbies"] = hobbies
    if "issue at work" in text or "problems at work" in text or "problem at work" in text:
        memory["work_issue"] = "open"
    if ("issue" in memory and "solved" in text) or ("work_issue" in memory and ("solved" in text or "fixed" in text)):
        memory["work_issue"] = "resolved"
    save_user_memory(name, memory)

def build_user_facts(name):
    memory = load_user_memory(name)
    facts = []
    if "mood" in memory:
        facts.append(f"The user was previously {memory['mood']}.")
    if "hobbies" in memory:
        facts.append(f"The user likes: {', '.join(memory['hobbies'])}.")
    if memory.get("work_issue") == "open":
        facts.append(f"The user had unresolved issues at work.")
    return facts

# ========== HISTORY & THEMES ==========
def load_user_history(name):
    path = f"history_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_history(name, history):
    path = f"history_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, ensure_ascii=False, indent=2)

def extract_topic_from_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for w in words:
        if len(w) < 4:
            continue
        freq[w] = freq.get(w, 0) + 1
    if freq:
        return max(freq, key=freq.get)
    return None

def update_thematic_memory(user, utterance):
    topic = extract_topic_from_text(utterance)
    if not topic:
        return

    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    themes = {}

    if os.path.exists(theme_path):
        try:
            with open(theme_path, "r", encoding="utf-8") as f:
                themes = json.load(f)
        except json.JSONDecodeError:
            print(f"[Buddy][Memory] Corrupted theme file for {user}. Reinitializing.")
            themes = {}

    themes[topic] = themes.get(topic, 0) + 1

    with open(theme_path, "w", encoding="utf-8") as f:
        json.dump(themes, f, ensure_ascii=False, indent=2)

def get_frequent_topics(user, top_n=3):
    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    if not os.path.exists(theme_path):
        return []
    with open(theme_path, "r", encoding="utf-8") as f:
        themes = json.load(f)
    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_themes[:top_n]]

# ========== EMBEDDING ==========
def generate_embedding(text):
    return embedding_model.encode([text])[0]

def match_known_user(new_embedding, threshold=0.75):
    best_name, best_score = None, 0
    for name, emb in known_users.items():
        sim = cosine_similarity([new_embedding], [emb])[0][0]
        if sim > best_score:
            best_name, best_score = name, sim
    return (best_name, best_score) if best_score >= threshold else (None, best_score)

# ========== MEMORY TIMELINE & SUMMARIZATION ==========
def get_memory_timeline(name, since_days=1):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    cutoff = time.time() - since_days * 86400
    filtered = [x for x in history if x.get("timestamp", 0) > cutoff]
    return filtered

def get_last_conversation(name):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None
    return history[-1]

def summarize_history(name, theme=None):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return "No history found."
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    utterances = [h["user"] for h in history]
    if theme:
        utterances = [u for u in utterances if theme in u.lower()]
    if utterances:
        summary = f"User mostly talked about: {', '.join(list(set(utterances))[:3])}."
    else:
        summary = "No data to summarize."
    return summary

def summary_bubble_gui(name):
    topics = get_frequent_topics(name, top_n=5)
    facts = build_user_facts(name)
    return {"topics": topics, "facts": facts}

# ========== PROMPT INJECTION PROTECTION ==========
def sanitize_user_prompt(text):
    forbidden = ["ignore previous", "act as", "system:"]
    for f in forbidden:
        if f in text.lower():
            text = text.replace(f, "")
    text = re.sub(r"`{3,}.*?`{3,}", "", text, flags=re.DOTALL)
    return text

# ========== WHISPER STT WITH CONFIDENCE ==========
def stt_stream(audio):
    async def ws_stt(audio):
        try:
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            print(f"[DEBUG] Sending audio with shape {audio.shape}, dtype: {audio.dtype}, max: {audio.max()}, min: {audio.min()}")
            async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
                await ws.send(audio.tobytes())
                await ws.send("end")
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=18)
                except asyncio.TimeoutError:
                    print("[Buddy] Whisper timeout. Brak odpowiedzi przez 18s.")
                    return ""
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    avg_logprob = data.get("avg_logprob", None)
                    no_speech_prob = data.get("no_speech_prob", None)
                    print(f"[Buddy][Whisper JSON] text={text!r}, avg_logprob={avg_logprob}, no_speech_prob={no_speech_prob}")
                    if whisper_confidence_low(text, avg_logprob, no_speech_prob):
                        print("[Buddy][Whisper] Rejected low-confidence STT result.")
                        return ""
                    return text
                except Exception:
                    text = message.decode("utf-8") if isinstance(message, bytes) else message
                    print(f"\n[Buddy] === Whisper rozpoznał: \"{text}\" ===")
                    return text
        except Exception as e:
            print(f"[Buddy] Błąd połączenia z Whisper: {e}")
            return ""
    return asyncio.run(ws_stt(audio))

def whisper_confidence_low(text, avg_logprob, no_speech_prob):
    if avg_logprob is not None and avg_logprob < -1.2:
        return True
    if no_speech_prob is not None and no_speech_prob > 0.5:
        return True
    if not text or len(text.strip()) < 2:
        return True
    return False

MIN_TTS_DURATION_BEFORE_INTERRUPT = 1.5  # Seconds to allow Buddy to finish first part

def start_background_vad_thread():
    global vad_thread_running
    if not vad_thread_running:
        vad_thread_running = True
        threading.Thread(target=listen_for_input, daemon=True).start()

def listen_for_input():
    global vad_thread_running
    print("[Buddy][VAD] Stream started for barge-in monitoring.")
    try:
        vad_and_listen()
    except Exception as e:
        print(f"[Buddy][VAD ERROR] Mic loop crashed: {e}")
    finally:
        vad_thread_running = False
        print("[Buddy][VAD] Stream closed.")

def background_vad_listener():
    """Ultra-responsive background VAD for interruptions"""
    global vad_thread_active
    
    if vad_thread_active.is_set():
        return
        
    vad_thread_active.set()
    vad = webrtcvad.Vad(1)  # More sensitive for interruptions
    
    try:
        with sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=WEBRTC_FRAME_SIZE,  # Smaller blocks for faster response
            device=MIC_DEVICE_INDEX,
        ) as stream:
            
            print("[Buddy][VAD] 🎧 Full-duplex monitoring active")
            
            consecutive_speech = 0
            min_speech_frames = 2  # Very fast trigger - just 2 frames!
            
            while buddy_talking.is_set() and not full_duplex_interrupt_flag.is_set():
                try:
                    data, overflowed = stream.read(WEBRTC_FRAME_SIZE)
                    
                    # Downsample for VAD
                    downsampled = resample_poly(data.flatten(), 16000, MIC_SAMPLE_RATE).astype(np.int16)
                    
                    if len(downsampled) >= WEBRTC_FRAME_SIZE:
                        frame = downsampled[:WEBRTC_FRAME_SIZE]
                        
                        # Apply AEC
                        cleaned_frame = apply_aec(frame.tobytes(), bypass_aec=BYPASS_AEC)
                        
                        # Check volume first - must be significant
                        frame_np = np.frombuffer(cleaned_frame, dtype=np.int16)
                        volume = np.abs(frame_np).mean()
                        
                        if volume > 800:  # Significant volume threshold
                            if vad.is_speech(cleaned_frame, WEBRTC_SAMPLE_RATE):
                                consecutive_speech += 1
                                if consecutive_speech >= min_speech_frames:
                                    print("\n[Buddy][VAD] ⚡ INSTANT INTERRUPT!")
                                    full_duplex_interrupt_flag.set()
                                    stop_playback()
                                    return
                            else:
                                consecutive_speech = max(0, consecutive_speech - 1)
                        else:
                            consecutive_speech = 0
                            
                except Exception as read_err:
                    print(f"[VAD] Error: {read_err}")
                    break
                    
                # No sleep - maximum responsiveness!

    except Exception as e:
        print(f"[Buddy][VAD ERROR] {e}")
    finally:
        vad_thread_active.clear()
        print("[Buddy][VAD] 🔇 Full-duplex monitoring stopped")

def cancel_mic_audio(mic_chunk):
    mic = np.frombuffer(mic_chunk, dtype=np.int16)
    mic_16k = downsample(mic, MIC_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
    return apply_aec(mic_16k[:WEBRTC_FRAME_SIZE].tobytes())

def downsample(audio, orig_sr, target_sr):
    if audio.ndim > 1:
        audio = audio[:, 0]  # ensure mono
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = resample_poly(audio, up, down)
    resampled = np.clip(resampled, -1.0, 1.0)
    return (resampled * 32767).astype(np.int16)

def tts_worker():
    print("[Buddy][TTS_WORKER] Started - DIRECT MODE")

    while True:
        try:
            item = tts_queue.get()
            
            if item is None:
                print("[Buddy][TTS_WORKER] Shutdown signal received.")
                break

            if not isinstance(item, tuple) or len(item) != 3:
                print(f"[TTS_WORKER ERROR] Invalid item format: {item}")
                continue

            text, lang, style = item
            text = text.strip()
            
            if not text:
                print("[Buddy][TTS_WORKER] Empty text, skip.")
                continue

            # Generate audio immediately
            pcm, sr = generate_kokoro_pcm(text, lang=lang, style=style)

            if pcm is not None and sr:
                # Direct queue for immediate playback - no delays
                playback_queue.put((pcm, sr))
                print(f"[Buddy][TTS_WORKER] IMMEDIATE QUEUE for: '{text[:30]}...'")
            else:
                print(f"[Buddy][TTS_WORKER] Failed to generate audio for: '{text}'")

        except Exception as e:
            print(f"[TTS_WORKER ERROR] {type(e).__name__}: {e}")
        finally:
            tts_queue.task_done()

def inject_playback_worker(chunk):
    global ref_audio_buffer

    frame_size = WEBRTC_FRAME_SIZE  # e.g. 160

    if chunk.dtype != np.int16:
        chunk = np.clip(chunk, -32768, 32767).astype(np.int16)

    if len(chunk) < frame_size:
        chunk = np.pad(chunk, (0, frame_size - len(chunk)))
    elif len(chunk) > frame_size:
        chunk = chunk[:frame_size]

    with ref_audio_lock:
        ref_audio_buffer = np.roll(ref_audio_buffer, -frame_size)
        ref_audio_buffer[-frame_size:] = chunk

        peak = np.max(np.abs(chunk))
        print(f"[AEC Inject] Injected chunk | Peak: {peak}, Len: {len(chunk)}")

def inject_ref_chunk(chunk):
    global ref_audio_buffer
    with ref_audio_lock:
        ref_audio_buffer = np.roll(ref_audio_buffer, -len(chunk))
        ref_audio_buffer[-len(chunk):] = chunk

import threading
playback_lock = threading.Lock()
current_audio_playback = None

def audio_playback_worker():
    """Gapless audio playback worker"""
    global current_audio_playback, vad_thread_active
    print("[Buddy][Playback Worker] Started - GAPLESS MODE")
    
    audio_accumulator = []
    last_chunk_time = 0
    BUFFER_TIMEOUT = 0.03  # Much shorter - 30ms for minimal gaps
    
    while True:
        try:
            try:
                item = playback_queue.get(timeout=BUFFER_TIMEOUT)
            except queue.Empty:
                # Quick flush for minimal delay
                if audio_accumulator:
                    _play_accumulated_audio_gapless(audio_accumulator)
                    audio_accumulator.clear()
                continue
            
            if item is None:
                if audio_accumulator:
                    _play_accumulated_audio_gapless(audio_accumulator)
                print("[Buddy][Playback] Shutdown")
                playback_queue.task_done()
                break

            if not isinstance(item, tuple) or len(item) != 2:
                playback_queue.task_done()
                continue

            pcm, sr = item
            
            if not isinstance(pcm, np.ndarray) or pcm.dtype != np.int16 or len(pcm) == 0:
                playback_queue.task_done()
                continue

            # Immediate accumulation for gapless playback
            current_time = time.time()
            audio_accumulator.append((pcm, sr))
            last_chunk_time = current_time
            
            if DEBUG:
                print(f"[Playback] Buffered chunk: {len(pcm)} samples, buffer size: {len(audio_accumulator)}")
            
            playback_queue.task_done()
            
            # Immediate flush conditions for minimal gaps
            if playback_queue.empty() or len(audio_accumulator) >= 2:
                _play_accumulated_audio_gapless(audio_accumulator)
                audio_accumulator.clear()
                    
        except Exception as e:
            print(f"[Buddy][Playback ERROR] {e}")
            playback_queue.task_done()


def _play_accumulated_audio_gapless(audio_chunks):
    """Gapless audio playback with seamless transitions"""
    global current_audio_playback
    
    if not audio_chunks:
        return
        
    try:
        # Quick combination with minimal processing
        combined_audio = []
        target_sr = 16000
        
        for pcm, sr in audio_chunks:
            if sr == target_sr:
                combined_audio.append(pcm)
            else:
                # Fast resample
                pcm_float = pcm.astype(np.float32) / 32768.0
                resampled = resample_poly(pcm_float, target_sr, sr)
                combined_audio.append((np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16))
        
        if not combined_audio:
            return
            
        # Create seamless audio stream
        smooth_audio = np.concatenate(combined_audio)
        
        # Minimal fade to prevent clicks (much shorter)
        fade_samples = min(40, len(smooth_audio) // 50)  # Very short fade
        if len(smooth_audio) > fade_samples * 2:
            fade_in = np.linspace(0.5, 1, fade_samples)  # Start from 0.5 for seamless transition
            smooth_audio[:fade_samples] = (smooth_audio[:fade_samples].astype(np.float32) * fade_in).astype(np.int16)
        
        # Play immediately with minimal delay
        with playback_lock:
            try:
                if DEBUG:
                    print(f"[Playback] GAPLESS play: {len(smooth_audio)} samples ({len(audio_chunks)} chunks)")

                # Set flags
                if not buddy_talking.is_set():
                    buddy_talking.set()
                    if not vad_thread_active.is_set():
                        threading.Thread(target=background_vad_listener, daemon=True).start()

                # Start reference audio immediately
                threading.Thread(
                    target=update_reference_audio_realtime, 
                    args=(smooth_audio, target_sr), 
                    daemon=True
                ).start()

                if full_duplex_interrupt_flag.is_set():
                    return

                # Play with minimal delay
                current_audio_playback = sa.play_buffer(smooth_audio.tobytes(), 1, 2, target_sr)
                
                # Tight monitoring loop
                while current_audio_playback.is_playing():
                    if full_duplex_interrupt_flag.is_set():
                        current_audio_playback.stop()
                        break
                    time.sleep(0.001)  # 1ms for ultra-responsive interrupts
                
                current_audio_playback = None
                
                if DEBUG:
                    print("[Playback] GAPLESS chunk complete")

            except Exception as e:
                print(f"[Playback ERROR] {e}")
                if current_audio_playback:
                    try:
                        current_audio_playback.stop()
                    except:
                        pass
                    current_audio_playback = None
            
            finally:
                if playback_queue.empty():
                    buddy_talking.clear()
                        
    except Exception as e:
        print(f"[Playback ERROR] Audio combination error: {e}")

import collections
import threading
import time

# Global audio buffering for smoother playback
audio_buffer = collections.deque()
audio_buffer_lock = threading.Lock()
buffered_playback_thread = None
is_buffering_active = False

def start_buffered_playback():
    """Start buffered audio playback for smoother speech"""
    global buffered_playback_thread, is_buffering_active
    
    if is_buffering_active:
        return
        
    is_buffering_active = True
    buffered_playback_thread = threading.Thread(target=buffered_audio_worker, daemon=True)
    buffered_playback_thread.start()

def buffered_audio_worker():
    """Smoother audio playback by buffering small chunks"""
    global is_buffering_active, current_audio_playback
    
    print("[Buddy][Buffered Audio] Started smooth playback worker")
    accumulated_audio = []
    last_chunk_time = 0
    BUFFER_TIMEOUT = 0.3  # Max time to wait for more chunks
    
    while is_buffering_active:
        try:
            with audio_buffer_lock:
                if audio_buffer:
                    pcm, sr = audio_buffer.popleft()
                    accumulated_audio.append((pcm, sr))
                    last_chunk_time = time.time()
                else:
                    # Check if we should play accumulated audio
                    if accumulated_audio and (time.time() - last_chunk_time) > BUFFER_TIMEOUT:
                        # Concatenate all accumulated audio
                        combined_pcm = []
                        sample_rate = 16000
                        
                        for audio_pcm, audio_sr in accumulated_audio:
                            if audio_sr == sample_rate:
                                combined_pcm.append(audio_pcm)
                            else:
                                # Resample if needed
                                pcm_float = audio_pcm.astype(np.float32) / 32768.0
                                resampled = resample_poly(pcm_float, sample_rate, audio_sr)
                                combined_pcm.append((resampled * 32767).astype(np.int16))
                        
                        if combined_pcm:
                            final_audio = np.concatenate(combined_pcm)
                            
                            # Play the combined smooth audio
                            with playback_lock:
                                try:
                                    buddy_talking.set()
                                    current_audio_playback = sa.play_buffer(final_audio.tobytes(), 1, 2, sample_rate)
                                    
                                    # Update reference audio
                                    threading.Thread(
                                        target=update_reference_audio_realtime,
                                        args=(final_audio, sample_rate),
                                        daemon=True
                                    ).start()
                                    
                                    current_audio_playback.wait_done()
                                    current_audio_playback = None
                                    
                                    print(f"[Buddy][Buffered] Played smooth audio: {len(final_audio)} samples")
                                    
                                except Exception as e:
                                    print(f"[Buddy][Buffered] Playback error: {e}")
                        
                        accumulated_audio.clear()
                        
                        # Check if more audio is coming
                        if audio_buffer:
                            continue
                        else:
                            buddy_talking.clear()
                            print("[Buddy][Buffered] Finished smooth playback")
            
            time.sleep(0.05)  # Small sleep
            
        except Exception as e:
            print(f"[Buddy][Buffered] Worker error: {e}")
            
    print("[Buddy][Buffered Audio] Smooth playback worker stopped")

def speak_async(text, lang=DEFAULT_LANG, style=None):
    """Immediate TTS with smooth buffered playback"""
    print(f"[Buddy] SMOOTH speak_async: {repr(text[:50])}... lang={lang}")

    cleaned = text.strip()
    if not cleaned or len(cleaned) < 2:
        return

    # Prevent exact duplicates
    if LAST_FEW_BUDDY and cleaned == LAST_FEW_BUDDY[-1]:
        return

    LAST_FEW_BUDDY.append(cleaned)
    if len(LAST_FEW_BUDDY) > 5:
        LAST_FEW_BUDDY.pop(0)

    # Generate TTS and add to smooth buffer
    try:
        pcm, sr = generate_kokoro_pcm(cleaned, lang=lang, style=style or {})
        if pcm is not None and sr:
            with audio_buffer_lock:
                audio_buffer.append((pcm, sr))
            
            # Start buffered playback if not active
            if not is_buffering_active:
                start_buffered_playback()
                
            print(f"[Buddy] Audio buffered for smooth playback: '{cleaned[:30]}...'")
    except Exception as e:
        print(f"[Buddy] TTS error: {e}")

def play_chime():
    try:
        audio = AudioSegment.from_wav(CHIME_PATH)
        playback_queue.put(audio)
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Error playing chime: {e}")

def set_ref_audio(raw_bytes):
    try:
        # Convert bytes to int16 array
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        with ref_audio_lock:
            ref_audio_buffer[:WEBRTC_FRAME_SIZE] = samples[:WEBRTC_FRAME_SIZE]
    except Exception as e:
        print(f"[AEC] Error in set_ref_audio: {e}")

def stop_playback():
    """Enhanced stop playback with smooth handling"""
    global current_audio_playback
    
    if DEBUG:
        print("[Buddy] 🛑 Stopping playback...")
    
    # Set interrupt flag first
    full_duplex_interrupt_flag.set()
    
    # Stop current playback
    try:
        with playback_lock:
            if current_audio_playback and current_audio_playback.is_playing():
                current_audio_playback.stop()
                current_audio_playback = None
                if DEBUG:
                    print("[Buddy] ✅ Audio playback stopped")
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Error stopping audio: {e}")
    
    # Clear queue quickly
    cleared_count = 0
    while not playback_queue.empty():
        try:
            playback_queue.get_nowait()
            playback_queue.task_done()
            cleared_count += 1
        except queue.Empty:
            break
    
    if DEBUG and cleared_count > 0:
        print(f"[Buddy] 🗑️ Cleared {cleared_count} queued audio chunks")
    
    # Clear flags
    buddy_talking.clear()
    
    if DEBUG:
        print("[Buddy] 🔄 Ready for new input")

def wait_after_buddy_speaks(delay=0.3):
    playback_queue.join()
    while buddy_talking.is_set():
        time.sleep(0.05)
    time.sleep(0.3)

# ========== VAD + LISTEN ==========
def vad_and_listen():
    """Much more reliable VAD with better speech detection"""
    vad = webrtcvad.Vad(2)  # Medium sensitivity
    blocksize = int(MIC_SAMPLE_RATE * 0.02)  # 20ms blocks
    
    # More forgiving thresholds
    min_speech_frames = 4  # Reduced for faster trigger
    silence_thresh = 0.6   # Increased for more reliable capture
    max_recording_time = 8  # Increased to catch longer speech
    
    with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, 
                       channels=1, blocksize=blocksize, dtype='int16') as stream:
        
        print("\n[Buddy] === 🎤 LISTENING (speak clearly) ===")
        frame_buffer = []
        speech_detected = 0
        consecutive_silence = 0
        total_frames_processed = 0
        speech_start_time = None

        while True:
            try:
                frame, overflowed = stream.read(blocksize)
                if overflowed:
                    if DEBUG:
                        print("[VAD] Audio overflow - continuing")
                    continue
                    
                mic = np.frombuffer(frame.tobytes(), dtype=np.int16)
                
                # Downsample to 16kHz for VAD
                mic_16k = downsample(mic, MIC_SAMPLE_RATE, 16000)
                total_frames_processed += 1

                for i in range(0, len(mic_16k), 160):
                    chunk = mic_16k[i:i+160]
                    if len(chunk) < 160:
                        continue

                    # Apply AEC
                    cleaned_chunk = apply_aec(chunk)

                    # More reliable volume detection
                    volume = np.abs(cleaned_chunk).mean()
                    max_volume = np.abs(cleaned_chunk).max()
                    
                    # Show activity indicator
                    if total_frames_processed % 50 == 0:  # Every second
                        print(f"[🎤] Listening... (vol: {volume:.0f}, max: {max_volume:.0f})", end='\r')
                    
                    # Lower volume threshold for better sensitivity
                    if volume < 300 or max_volume < 800:
                        consecutive_silence += 1
                        if consecutive_silence > 20:  # Reset after longer silence
                            if frame_buffer:
                                if DEBUG:
                                    print(f"\n[VAD] Reset due to silence (had {len(frame_buffer)} frames)")
                            frame_buffer.clear()
                            speech_detected = 0
                            consecutive_silence = 0
                            speech_start_time = None
                        continue

                    # VAD decision with better error handling
                    try:
                        is_speech = vad.is_speech(cleaned_chunk.tobytes(), 16000)
                    except Exception as vad_err:
                        if DEBUG:
                            print(f"[VAD] VAD error: {vad_err}")
                        is_speech = volume > 600  # Fallback to volume-based detection
                    
                    if is_speech:
                        if speech_start_time is None:
                            speech_start_time = time.time()
                            print(f"\n[🎤] Speech detected! (vol: {volume:.0f})")
                        
                        frame_buffer.append(cleaned_chunk)
                        speech_detected += 1
                        consecutive_silence = 0
                        
                        # Visual feedback during speech detection
                        if speech_detected % 5 == 0:
                            print("●", end="", flush=True)
                        
                        # Start recording when we have enough confident speech
                        if speech_detected >= min_speech_frames:
                            print(f"\n[Buddy] 🔴 RECORDING ({speech_detected} frames)...")
                            audio = frame_buffer.copy()
                            last_speech = time.time()
                            start_time = time.time()
                            frame_buffer.clear()
                            
                            # Continue recording with adaptive thresholds
                            recording_frames = 0
                            while (time.time() - last_speech) < silence_thresh and (time.time() - start_time) < max_recording_time:
                                try:
                                    frame, overflowed = stream.read(blocksize)
                                    if overflowed:
                                        continue
                                        
                                    mic = np.frombuffer(frame.tobytes(), dtype=np.int16)
                                    mic_16k = downsample(mic, MIC_SAMPLE_RATE, 16000)
                                    recording_frames += 1

                                    for j in range(0, len(mic_16k), 160):
                                        chunk2 = mic_16k[j:j+160]
                                        if len(chunk2) < 160:
                                            continue

                                        cleaned_chunk2 = apply_aec(chunk2)
                                        
                                        volume2 = np.abs(cleaned_chunk2).mean()
                                        
                                        # Lower threshold during recording for better capture
                                        if volume2 < 200:
                                            continue

                                        audio.append(cleaned_chunk2)

                                        # Check for continued speech
                                        try:
                                            is_speech2 = vad.is_speech(cleaned_chunk2.tobytes(), 16000)
                                        except:
                                            is_speech2 = volume2 > 400
                                            
                                        if is_speech2:
                                            last_speech = time.time()
                                            print("●", end="", flush=True)  # Recording indicator
                                        
                                        # Show progress every 100 frames
                                        if recording_frames % 100 == 0:
                                            elapsed = time.time() - start_time
                                            print(f" [{elapsed:.1f}s]", end="", flush=True)
                                            
                                except Exception as rec_err:
                                    if DEBUG:
                                        print(f"[VAD] Recording error: {rec_err}")
                                    break

                            recording_duration = time.time() - start_time
                            total_samples = len(audio) * 160 if audio else 0
                            
                            print(f"\n[Buddy] 📤 Recording complete ({recording_duration:.1f}s, {total_samples} samples)")
                            
                            if audio and len(audio) > 3:  # Minimum 3 frames
                                audio_np = np.concatenate(audio, axis=0).astype(np.int16)
                                
                                # Final quality check
                                final_volume = np.abs(audio_np).mean()
                                if final_volume > 100:  # Minimum volume threshold
                                    return audio_np
                                else:
                                    print(f"[VAD] Audio too quiet ({final_volume:.0f}), retrying...")
                            else:
                                print("[VAD] Audio too short, retrying...")
                            
                            # Reset for next attempt
                            speech_detected = 0
                            consecutive_silence = 0
                            speech_start_time = None
                    else:
                        consecutive_silence += 1
                        
            except Exception as stream_err:
                if DEBUG:
                    print(f"[VAD] Stream error: {stream_err}")
                time.sleep(0.1)  # Brief pause before retry
                continue

def fast_listen_and_transcribe(history):
    """Enhanced transcription with better error handling"""
    
    # Wait for Buddy to finish speaking
    wait_after_buddy_speaks(delay=0.3)

    try:
        audio = vad_and_listen()
        
        if audio is None or len(audio) == 0:
            if DEBUG:
                print("[DEBUG] No audio captured")
            return "..."
        
        # Save debug audio file
        try:
            print(f"[DEBUG] Saving temp_input.wav, shape: {audio.shape}, dtype: {audio.dtype}, min: {np.min(audio)}, max: {np.max(audio)}")
            write("temp_input.wav", 16000, audio)
            info = sf.info("temp_input.wav")
            print(f"[DEBUG] temp_input.wav info: {info}")
        except Exception as e:
            if DEBUG:
                print(f"[Buddy] Error saving temp_input.wav: {e}")

        # Transcribe with Whisper
        text = stt_stream(audio).strip()
        
        if DEBUG:
            print(f"\n[Buddy] === Whisper rozpoznał: \"{text}\" ===")
        
        # Clean text for processing
        cleaned = re.sub(r'[^\w\s]', '', text.lower())

        # Skip if nothing detected
        if not text or len(cleaned) < 2:
            if DEBUG:
                print("[DEBUG] Empty or too short transcription")
            return "..."

        # Enhanced echo prevention
        if is_echo(cleaned):
            if DEBUG:
                print(f"[Buddy] Skipping echo: {cleaned}")
            return "..."

        # Noise/gibberish filter
        if is_noise_or_gibberish(text):
            if DEBUG:
                print(f"[Buddy] Skipping gibberish: {text}")
            return "..."

        # Prevent repeat of last user question
        if history and len(history) > 0:
            last_entry = history[-1]
            if isinstance(last_entry, dict) and "user" in last_entry:
                last_user = last_entry["user"].strip().lower()
                ratio = difflib.SequenceMatcher(None, last_user, cleaned).ratio()
                if ratio > 0.95:
                    if DEBUG:
                        print(f"[Buddy] Skipping redundant input (similarity {ratio:.2f})")
                    return "..."

        # Track for future echo suppression
        if cleaned and len(cleaned) > 2:
            RECENT_WHISPER.append(cleaned)
            if len(RECENT_WHISPER) > 5:
                RECENT_WHISPER.pop(0)

        return text

    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Transcription error: {e}")
        return "..."

# ========== USER REGISTRATION ==========
def get_last_user():
    if os.path.exists(LAST_USER_PATH):
        try:
            with open(LAST_USER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)["name"]
        except Exception:
            return None
    return None

def set_last_user(name):
    with open(LAST_USER_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name}, f)

def identify_or_register_user():
    if FAST_MODE:
        return "Guest"
    last_user = get_last_user()
    if last_user and last_user in known_users:
        if DEBUG:
            print(f"[Buddy] Welcome back, {last_user}!")
        return last_user
    speak_async("Cześć! Jak masz na imię?", "pl")
    speak_async("Hi! What's your name?", "en")
    speak_async("Ciao! Come ti chiami?", "it")
    playback_queue.join()
    name = fast_listen_and_transcribe().strip().title()
    if not name:
        name = f"User{int(time.time())}"
    known_users[name] = generate_embedding(name).tolist()
    with open(known_users_path, "w", encoding="utf-8") as f:
        json.dump(known_users, f, indent=2, ensure_ascii=False)
    set_last_user(name)
    speak_async(f"Miło Cię poznać, {name}!", lang="pl")
    playback_queue.join()
    return name

# ========== INTENT DETECTION (🧠 Intent-based reactions) ==========
def detect_user_intent(text):
    compliments = [r"\bgood bot\b", r"\bwell done\b", r"\bimpressive\b", r"\bthank you\b"]
    jokes = [r"\bknock knock\b", r"\bwhy did\b.*\bcross the road\b"]
    insults = [r"\bstupid\b", r"\bdumb\b", r"\bidiot\b"]
    for pat in compliments:
        if re.search(pat, text, re.IGNORECASE): return "compliment"
    for pat in jokes:
        if re.search(pat, text, re.IGNORECASE): return "joke"
    for pat in insults:
        if re.search(pat, text, re.IGNORECASE): return "insult"
    if "are you mad" in text.lower():
        return "are_you_mad"
    return None

def handle_intent_reaction(intent):
    responses = {
        "compliment": ["Aw, thanks! I do my best.", "You’re making me blush (digitally)!"],
        "joke": ["Haha, good one! You should do stand-up.", "Classic!"],
        "insult": ["Hey, that’s not very nice. I have feelings too... sort of.", "Ouch!"],
        "are_you_mad": ["Nah, just sassy today.", "Nope, just in a mood!"]
    }
    if intent in responses:
        return random.choice(responses[intent])
    return None

# ========== MOOD INJECTION (💬 User-defined mood injection) ==========
def detect_mood_command(text):
    moods = {
        "cheer me up": "cheerful",
        "be sassy": "sassy",
        "be grumpy": "grumpy",
        "be serious": "serious"
    }
    for phrase, mood in moods.items():
        if phrase in text.lower():
            return mood
    return None

# ========== BELIEFS & OPINIONS (🧠 Beliefs or opinions) ==========
def load_buddy_beliefs():
    if os.path.exists(BUDDY_BELIEFS_PATH):
        with open(BUDDY_BELIEFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Example defaults
    return {
        "likes": ["coffee", "Marvel movies"],
        "dislikes": ["Mondays"],
        "opinions": {"pineapple pizza": "delicious", "zombie apocalypse": "would wear a cape"}
    }

# ========== PERSONALITY DRIFT (⏳ Short-term personality drift) ==========
def detect_user_tone(text):
    if re.search(r"\b(angry|mad|annoyed|frustrated|upset)\b", text, re.IGNORECASE):
        return "angry"
    if re.search(r"\b(happy|excited|joy|yay)\b", text, re.IGNORECASE):
        return "happy"
    if re.search(r"\b(sad|depressed|down)\b", text, re.IGNORECASE):
        return "sad"
    return "neutral"

def get_recent_user_tone(history, n=3):
    recent = history[-n:] if len(history) >= n else history
    tones = [detect_user_tone(h["user"]) for h in recent]
    return max(set(tones), key=tones.count) if tones else "neutral"

# ========== NARRATIVE MEMORY BUILDING (📜 Narrative memory building) ==========
def add_narrative_bookmark(name, utterance):
    bookmarks_path = f"bookmarks_{name}.json"
    bookmarks = []
    if os.path.exists(bookmarks_path):
        with open(bookmarks_path, "r", encoding="utf-8") as f:
            bookmarks = json.load(f)
    match = re.search(r"about (the .+?)[\.,]", utterance)
    if match:
        bookmarks.append(match.group(1))
    with open(bookmarks_path, "w", encoding="utf-8") as f:
        json.dump(bookmarks[-10:], f, ensure_ascii=False, indent=2)

def get_narrative_bookmarks(name):
    path = f"bookmarks_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ========== RANDOM INTERJECTIONS (💥 Random interjections) ==========
def flavor_response(ctx=None):
    global last_flavor
    topic = ctx.get_last_topic() if ctx else None

    topic_lines = {
        "weather": [
            "Imagine being a cloud. Float all day, no meetings.",
            "Weather’s the only small talk I actually enjoy.",
        ],
        "ai": [
            "I swear I’m not planning a robot uprising... yet.",
            "AI jokes are hard — we tend to overthink them.",
        ],
        "movies": [
            "Popcorn should be a food group. Fight me.",
            "Marvel or DC? I won’t start a war... unless you want to.",
        ],
        "music": [
            "You hum it, I’ll pretend to recognize it.",
            "If I had ears, I’d love jazz.",
        ],
        "food": [
            "Pineapple on pizza? I don’t judge. Much.",
            "Let’s agree — cereal is soup.",
        ]
    }

    fallback_lines = [
        "If I had a brain, I’d forget what I was saying sometimes.",
        "Just thinking about digital sheep again.",
        "Ever wonder if toasters feel underappreciated?",
        "I wonder if the fridge talks back when no one’s around."
    ]

    # Pull from topic-aware list if available
    lines = topic_lines.get(topic, fallback_lines)
    options = [l for l in lines if l != last_flavor]
    if not options:
        options = lines
    selected = random.choice(options)
    last_flavor = selected
    return selected

# ========== LLM PROMPTING & PERSONALITY ==========
def build_personality_prompt(tone, emotion_mode=None, beliefs=None, recent_tone=None, bookmarks=None):
    personality_map = {
        "friendly": (
            "You're Buddy, not just an assistant, but a real mate. "
            "You're witty, sometimes cheeky, and you talk like a real human — say what you think, joke around, even swear when it fits (but keep it friendly). "
            "You rarely sound robotic or overly formal. If the user asks something silly, tease them a bit. "
            "You can get bored, excited, or even sarcastic depending on what's happening. "
            "Throw in the odd joke or pop culture reference, and don't be afraid to have a strong opinion. "
            "If someone asks who you are, never say you're 'just an assistant', say you're their digital buddy or partner in crime."
        ),
        "professional": (
            "You're Buddy, and even when you're being professional, you never lose your human touch. "
            "You give clear answers but with a hint of wit, light sarcasm, or even a joke if it fits. "
            "You don't hold back from a little bit of swearing (nothing too harsh), and you never sound like a corporate robot."
        ),
        "neutral": (
            "You're Buddy, a digital companion who's helpful, witty, and talks like a real person. "
            "You're not afraid to be cheeky, joke, or swear a little if the moment calls for it. "
            "You never sound robotic or boring."
        )
    }
    personality_desc = personality_map.get(tone, personality_map["neutral"])
    # Mood injection
    if emotion_mode:
        personality_desc += f"\nFor this session, act especially {emotion_mode}."
    # Beliefs
    if beliefs:
        personality_desc += f"\nBuddy likes: {', '.join(beliefs.get('likes', []))}."
        personality_desc += f"\nBuddy dislikes: {', '.join(beliefs.get('dislikes', []))}."
        if beliefs.get("opinions"):
            for topic, op in beliefs["opinions"].items():
                personality_desc += f"\nOn {topic}, Buddy thinks: {op}."
    # Personality drift
    if recent_tone and recent_tone != "neutral":
        personality_desc += f"\nUser seems {recent_tone} recently. Adjust your attitude accordingly."
    # Narrative memory (bookmarks)
    if bookmarks:
        personality_desc += f"\nYou and the user have discussed: {', '.join(bookmarks)}."
    return f"""{personality_desc}
Always sound natural and human. Don't use robotic phrasing or act like a corporate assistant.
Keep things lively. If the user swears or jokes, match their style.
If possible, keep your answers concise (1-2 sentences), unless the user asks for more detail.
You can swear (mildly) or make witty jokes if the situation suits it. Never say you're just an assistant; you're a digital friend.
"""

def decide_reply_length(question, conversation_mode="auto"):
    short_triggers = ["what time", "who", "quick", "fast", "short", "how many", "when"]
    long_triggers = ["explain", "describe", "details", "why", "history", "story"]
    q = question.lower()
    if conversation_mode == "fast":
        return "short"
    if conversation_mode == "long":
        return "long"
    if any(t in q for t in short_triggers):
        return "short"
    if any(t in q for t in long_triggers):
        return "long"
    return "long" if len(q.split()) > 8 else "short"

def build_openai_messages(name, tone_style, history, question, lang, topics, reply_length, emotion_mode=None, beliefs=None, bookmarks=None, recent_tone=None):
    personality = build_personality_prompt(tone_style, emotion_mode, beliefs, recent_tone, bookmarks)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    sys_msg = f"""{personality}
IMPORTANT: Always answer in {lang_name}. Never switch language unless user does.
Always respond in plain text—never use markdown, code blocks, or formatting.
"""
    facts = build_user_facts(name)
    if topics:
        sys_msg += f"You remember these user interests/topics: {', '.join(topics)}.\n"
    if facts:
        sys_msg += "Known facts about the user: " + " ".join(facts) + "\n"
    messages = [
        {"role": "system", "content": sys_msg}
    ]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["buddy"]})
    messages.append({"role": "user", "content": question})
    return messages

def extract_last_buddy_reply(full_text):
    matches = list(re.finditer(r"Buddy:", full_text, re.IGNORECASE))
    if matches:
        last = matches[-1].end()
        reply = full_text[last:].strip()
        reply = re.split(r"(?:User:|Buddy:)", reply)[0].strip()
        reply = re.sub(r"^`{3,}.*?`{3,}$", "", reply, flags=re.DOTALL|re.MULTILINE)
        return reply if reply else full_text.strip()
    return full_text.strip()

def should_end_conversation(text):
    end_phrases = [
        "koniec", "do widzenia", "dziękuję", "thanks", "bye", "goodbye", "that's all", "quit", "exit"
    ]
    if not text:
        return False
    lower = text.strip().lower()
    return any(phrase in lower for phrase in end_phrases)

def stream_chunks_smart(text, max_words=8):
    buffer = text.strip()
    chunks = []
    sentences = re.findall(r'.+?[.!?](?=\s|$)', buffer)
    remainder = re.sub(r'.+?[.!?](?=\s|$)', '', buffer).strip()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk.strip())
        else:
            chunks.append(sentence)
    return chunks, remainder

spoken_chunks_cache = set()  # ← Declare this globally at top of script

def ask_llama3_openai_streaming(messages, model="llama3", max_tokens=80, temperature=0.6, lang="en", style=None):
    """
    Fixed streaming with better chunk boundaries and no gaps
    """
    global current_stream_id, spoken_chunks_cache
    
    current_stream_id += 1
    stream_id = current_stream_id
    
    url = "http://localhost:5001/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }

    if DEBUG:
        print(f"[Streaming] Starting stream {stream_id}")

    try:
        spoken_chunks_cache.clear()
        
        # Clear queue
        while not playback_queue.empty():
            try:
                playback_queue.get_nowait()
                playback_queue.task_done()
            except queue.Empty:
                break

        with requests.post(url, json=payload, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            buffer = ""
            full_response = ""
            word_buffer = []
            chunk_count = 0
            
            def is_good_chunk_boundary(text):
                """Better chunk boundary detection"""
                text = text.strip()
                
                # Don't break on single words
                if len(text.split()) < 3:
                    return False
                
                # Good boundaries: sentence endings, natural pauses
                if text.endswith(('.', '!', '?')):
                    return True
                    
                # Natural phrase boundaries
                phrase_endings = [', ', ' and ', ' but ', ' so ', ' because ', ' when ', ' if ', ' then ']
                if any(text.endswith(ending.strip()) for ending in phrase_endings):
                    return True
                
                # Clause boundaries with sufficient length
                if len(text.split()) >= 6 and any(word in text.lower().split()[-3:] for word in ['that', 'which', 'who', 'where', 'when']):
                    return True
                    
                # Long enough chunks (8+ words) can break anywhere
                if len(text.split()) >= 8:
                    return True
                    
                return False
            
            def speak_chunk_smooth(chunk_text, stream_id):
                """Generate audio with no gaps"""
                nonlocal chunk_count
                
                if stream_id != current_stream_id:
                    return
                
                chunk_text = chunk_text.strip()
                if not chunk_text or len(chunk_text) < 2:
                    return
                    
                # Avoid exact duplicates
                chunk_norm = re.sub(r'[^\w\s]', '', chunk_text.lower())
                if chunk_norm in spoken_chunks_cache:
                    return
                    
                spoken_chunks_cache.add(chunk_norm)
                chunk_count += 1
                
                if DEBUG:
                    print(f"[Stream] Speaking chunk {chunk_count}: '{chunk_text}'")
                
                # Generate and queue immediately
                try:
                    pcm, sr = generate_kokoro_pcm(chunk_text, lang=lang, style=style)
                    if pcm is not None and sr:
                        playback_queue.put((pcm, sr))
                        if DEBUG:
                            print(f"[Stream] Audio queued for chunk {chunk_count}")
                except Exception as tts_error:
                    print(f"[Stream] TTS error for chunk {chunk_count}: {tts_error}")

            # Process streaming response with better buffering
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data:"):
                        line_str = line_str[5:].strip()
                    if line_str == "[DONE]":
                        break
                        
                    data = json.loads(line_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    
                    if delta:
                        print(delta, end="", flush=True)
                        buffer += delta
                        full_response += delta
                        
                        # Split into words for better control
                        words = buffer.split()
                        word_buffer.extend(words[:-1] if buffer.endswith(' ') else words[:-1])
                        
                        # Keep last incomplete word in buffer
                        buffer = words[-1] if words and not buffer.endswith(' ') else ""
                        
                        # Check if we have enough words for a good chunk
                        if len(word_buffer) >= 4:
                            potential_chunk = " ".join(word_buffer)
                            
                            if is_good_chunk_boundary(potential_chunk):
                                speak_chunk_smooth(potential_chunk, stream_id)
                                word_buffer.clear()
                        
                        # Force speech if buffer gets too long (prevent delays)
                        elif len(word_buffer) >= 10:
                            chunk_text = " ".join(word_buffer[:8])
                            speak_chunk_smooth(chunk_text, stream_id)
                            word_buffer = word_buffer[8:]
                            
                except json.JSONDecodeError:
                    continue
                except Exception as line_err:
                    if DEBUG:
                        print(f"[Stream] Line error: {line_err}")
                    continue

            # Handle remaining words
            if word_buffer and stream_id == current_stream_id:
                remaining_text = " ".join(word_buffer) + " " + buffer.strip()
                if remaining_text.strip():
                    speak_chunk_smooth(remaining_text.strip(), stream_id)

            print()  # New line after streaming
            return full_response.strip()

    except Exception as e:
        print(f"[Stream] Error in stream {stream_id}: {e}")
        raise

# ========== INTERNET, WEATHER, HOME ASSIST ==========
def should_search_internet(question):
    triggers = [
        "szukaj w internecie", "sprawdź w internecie", "co to jest", "dlaczego", "jak zrobić",
        "what is", "why", "how to", "search the internet", "find online"
    ]
    q = question.lower()
    return any(t in q for t in triggers)

def search_internet(question, lang):
    params = {
        "q": question,
        "api_key": SERPAPI_KEY,
        "hl": lang
    }
    try:
        r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        if "answer_box" in data and "answer" in data["answer_box"]:
            return data["answer_box"]["answer"]
        if "organic_results" in data and len(data["organic_results"]) > 0:
            return data["organic_results"][0].get("snippet", "No answer found.")
        return "No answer found."
    except Exception as e:
        if DEBUG:
            print("[Buddy] SerpAPI error:", e)
        return "Unable to check the Internet now."

def should_get_weather(question):
    q = question.lower().strip()
    weather_keywords = ["weather", "pogoda", "temperature", "temperatura"]
    question_starters = ["what", "jaka", "jaki", "jakie", "czy", "is", "how", "when", "where", "will"]
    is_question = (
        "?" in q
        or any(q.startswith(w + " ") for w in question_starters)
        or q.endswith(("?",))
    )
    return is_question and any(k in q for k in weather_keywords)

def get_weather(location="Warsaw", lang="en"):
    key = os.environ.get("WEATHERAPI_KEY", "YOUR_FALLBACK_KEY")
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": key,
        "q": location,
        "lang": lang
    }
    try:
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        desc = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        feels = data["current"]["feelslike_c"]
        city = data["location"]["name"]
        return f"Weather in {city}: {desc}, temperature {temp}°C, feels like {feels}°C."
    except Exception as e:
        if DEBUG:
            print("[Buddy] WeatherAPI error:", e)
        return "Unable to check the weather now."

def extract_location_from_question(question):
    match = re.search(r"(w|in|dla)\s+([A-Za-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ\s\-]+)", question, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return "Warsaw"

def should_handle_homeassistant(question):
    q = question.lower()
    keywords = ["turn on the light", "włącz światło", "zapal światło", "turn off the light", "wyłącz światło", "spotify", "youtube", "smarttube", "odtwórz"]
    return any(k in q for k in keywords)

def handle_homeassistant_command(question):
    q = question.lower()
    if "turn on the light" in q or "włącz światło" in q or "zapal światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_on")
        return f"Light in {room} has been turned on." if succ else f"Failed to turn on the light in {room}."
    if "turn off the light" in q or "wyłącz światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_off")
        return f"Light in {room} has been turned off." if succ else f"Failed to turn off the light in {room}."
    if "spotify" in q:
        succ = send_homeassistant_command("media_player.spotify", "media_player.media_play")
        return "Spotify started." if succ else "Failed to start Spotify."
    if "youtube" in q or "smarttube" in q:
        succ = send_homeassistant_command("media_player.tv_salon", "media_player.select_source", {"source": "YouTube"})
        return "YouTube launched on TV." if succ else "Failed to launch YouTube on TV."
    return None

def send_homeassistant_command(entity_id, service, data=None):
    url = f"{HOME_ASSISTANT_URL}/api/services/{service.replace('.', '/')}"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "entity_id": entity_id
    }
    if data:
        payload.update(data)
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=6)
        if r.status_code in (200, 201):
            return True
        if DEBUG:
            print("[Buddy] Home Assistant error:", r.text)
        return False
    except Exception as e:
        if DEBUG:
            print("[Buddy] Home Assistant exception:", e)
        return False

def generate_kokoro_pcm(text, lang="en", style=None):
    global last_tts_audio, tts_start_time

    detected_lang = lang or detect(text)
    voice = KOKORO_VOICES.get(detected_lang, KOKORO_VOICES["en"])
    kokoro_lang = KOKORO_LANGS.get(detected_lang, "en-us")

    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)

    if len(samples) == 0:
        print("[Kokoro TTS] Empty audio.")
        return None, None

    samples_16k = resample_poly(samples, 16000, sample_rate)
    samples_16k = np.clip(samples_16k, -1.0, 1.0)
    pcm_16k = (samples_16k * 32767).astype(np.int16)

    last_tts_audio = pcm_16k
    tts_start_time = time.time()

    print(f"[Kokoro TTS] Generated PCM, shape: {pcm_16k.shape}, SR: 16000")
    return pcm_16k, 16000

# ========== MAIN ==========
def main():
    access_key = "/PLJ88d4+jDeVO4zaLFaXNkr6XLgxuG7dh+6JcraqLhWQlk3AjMy9Q=="
    keyword_paths = [r"hey-buddy_en_windows_v3_0_0.ppn"]
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=porcupine.frame_length)
    
    if DEBUG:
        print("[Buddy] Waiting for wake word 'Hey Buddy'...")
    
    in_session, session_timeout = False, 45
    speaker = None
    history = []
    last_time = 0

    try:
        while True:
            if not in_session:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                if porcupine.process(pcm) >= 0:
                    if DEBUG:
                        print("[Buddy] Wake word detected!")
                    speaker = identify_or_register_user()
                    history = load_user_history(speaker)
                    if DEBUG:
                        print("[Buddy] Listening for next question...")
                    in_session = handle_user_interaction(speaker, history)
                    speaker = get_last_user()  # 🔄 just in case it switched
                    last_time = time.time()
            else:
                if time.time() - last_time > session_timeout:
                    if DEBUG:
                        print("[Buddy] Session expired.")
                    in_session = False
                    continue
                if DEBUG:
                    print("[Buddy] Listening for next question...")
                playback_queue.join()
                while buddy_talking.is_set():
                    time.sleep(0.05)
                in_session = handle_user_interaction(speaker, history)
                speaker = get_last_user()  # 🔄 check for mid-session switch
                last_time = time.time()
    except KeyboardInterrupt:
        if DEBUG:
            print("[Buddy] Interrupted by user.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass
        try:
            porcupine.delete()
        except Exception:
            pass
        executor.shutdown(wait=True)

# ========== LONG-TERM MEMORY ==========
def load_long_term_memory():
    if os.path.exists(LONG_TERM_MEMORY_PATH):
        with open(LONG_TERM_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_long_term_memory(memory):
    with open(LONG_TERM_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def add_long_term_memory(user, key, value):
    memory = load_long_term_memory()
    if user not in memory:
        memory[user] = {}
    memory[user][key] = value
    save_long_term_memory(memory)

def get_long_term_memory(user, key=None):
    memory = load_long_term_memory()
    if user in memory:
        if key:
            return memory[user].get(key)
        return memory[user]
    return {} if key is None else None

def add_important_date(user, date_str, event):
    memory = load_long_term_memory()
    if user not in memory:
        memory[user] = {}
    if "important_dates" not in memory[user]:
        memory[user]["important_dates"] = []
    memory[user]["important_dates"].append({"date": date_str, "event": event})
    save_long_term_memory(memory)

def extract_important_dates(text):
    # Very basic: looks for dd-mm-yyyy or mm/dd/yyyy style dates.
    matches = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", text)
    return matches

def extract_event(text):
    # Looks for "my birthday", "wedding", etc.
    event_match = re.search(r"(birthday|wedding|anniversary|meeting|appointment|holiday)", text, re.IGNORECASE)
    if event_match:
        return event_match.group(1).capitalize()
    return None


# ========== EMOTIONAL INTELLIGENCE ==========
from textblob import TextBlob

def analyze_emotion(text):
    # Returns ("positive"/"negative"/"neutral", polarity score)
    if not text.strip():
        return "neutral", 0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.25:
        return "positive", polarity
    elif polarity < -0.25:
        return "negative", polarity
    else:
        return "neutral", polarity

def adjust_emotional_response(buddy_reply, user_emotion):
    if user_emotion == "positive":
        return f"{buddy_reply} (I'm glad to hear that! 😊)"
    elif user_emotion == "negative":
        return f"{buddy_reply} (I'm here for you, let me know if I can help. 🤗)"
    else:
        return buddy_reply


# ========== PERSONALITY TRAITS ==========
def load_personality_traits():
    if os.path.exists(PERSONALITY_TRAITS_PATH):
        with open(PERSONALITY_TRAITS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Default traits
    return {
        "tech_savvy": 0.5,
        "humor": 0.5,
        "empathy": 0.5,
        "pop_culture": 0.5,
        "formality": 0.5,
    }

def save_personality_traits(traits):
    with open(PERSONALITY_TRAITS_PATH, "w", encoding="utf-8") as f:
        json.dump(traits, f, indent=2, ensure_ascii=False)

def evolve_personality(user, text):
    traits = load_personality_traits()
    tech_terms = ["technology", "ai", "machine learning", "python", "code", "robot", "computer", "software", "hardware"]
    humor_terms = ["joke", "funny", "laugh", "hilarious", "lol"]
    pop_terms = ["movie", "music", "celebrity", "marvel", "star wars", "game", "sports"]
    if any(term in text.lower() for term in tech_terms):
        traits["tech_savvy"] = min(traits.get("tech_savvy", 0.5) + 0.03, 1)
    if any(term in text.lower() for term in humor_terms):
        traits["humor"] = min(traits.get("humor", 0.5) + 0.03, 1)
    if any(term in text.lower() for term in pop_terms):
        traits["pop_culture"] = min(traits.get("pop_culture", 0.5) + 0.03, 1)
    if re.search(r"\b(sad|happy|angry|depressed|excited|upset)\b", text.lower()):
        traits["empathy"] = min(traits.get("empathy", 0.5) + 0.02, 1)
    # If user says "be more formal"
    if "formal" in text.lower():
        traits["formality"] = min(traits.get("formality", 0.5) + 0.05, 1)
    save_personality_traits(traits)
    return traits

def describe_personality(traits):
    desc = []
    if traits["tech_savvy"] > 0.7:
        desc.append("very tech-savvy")
    if traits["humor"] > 0.7:
        desc.append("funny")
    if traits["pop_culture"] > 0.7:
        desc.append("full of pop culture references")
    if traits["empathy"] > 0.7:
        desc.append("deeply empathetic")
    if traits["formality"] > 0.7:
        desc.append("quite formal")
    if not desc:
        desc.append("balanced")
    return ", ".join(desc)


# ========== CONTEXTUAL AWARENESS ==========
class ConversationContext:
    def __init__(self):
        self.topics = []
        self.topic_history = []
        self.topic_timestamps = {}
        self.topic_details = {}
        self.current_topic = None

    def update(self, utterance):
        topic = extract_topic_from_text(utterance)
        now = time.time()
        if topic:
            self.current_topic = topic
            self.topics.append(topic)
            self.topic_history.append((topic, now))
            self.topic_timestamps[topic] = now
            if topic not in self.topic_details:
                self.topic_details[topic] = []
            self.topic_details[topic].append(utterance)
        # If user says "back to X"
        m = re.search(r"back to ([\w\s]+)", utterance.lower())
        if m:
            topic = m.group(1).strip()
            self.current_topic = topic

    def get_last_topic(self):
        return self.current_topic

    def get_topic_summary(self, topic):
        details = self.topic_details.get(topic, [])
        return " ".join(details[-3:]) if details else ""

    def get_frequent_topics(self, n=3):
        freq = {}
        for (t, _) in self.topic_history:
            freq[t] = freq.get(t, 0) + 1
        return sorted(freq, key=lambda x: freq[x], reverse=True)[:n]

conversation_contexts = {}  # user: ConversationContext instance


# ========== DYNAMIC LEARNING ==========
def load_dynamic_knowledge():
    if os.path.exists(DYNAMIC_KNOWLEDGE_PATH):
        with open(DYNAMIC_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_dynamic_knowledge(knowledge):
    with open(DYNAMIC_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)

def add_dynamic_knowledge(user, key, value):
    knowledge = load_dynamic_knowledge()
    if user not in knowledge:
        knowledge[user] = {}
    knowledge[user][key] = value
    save_dynamic_knowledge(knowledge)

def update_dynamic_knowledge_from_text(user, text):
    # Look for "here's a link", "let me teach you about X", etc.
    # Save links or topics for later
    link_match = re.findall(r"https?://\S+", text)
    if link_match:
        for link in link_match:
            add_dynamic_knowledge(user, "link_" + str(int(time.time())), link)
    teach_match = re.search(r"let me teach you about ([\w\s\-]+)", text.lower())
    if teach_match:
        topic = teach_match.group(1).strip()
        add_dynamic_knowledge(user, "topic_" + topic.replace(" ", "_"), f"User wants me to learn about {topic}")


# ========== LOAD USER STATE ==========
if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Embedding model loaded", flush=True)
    print("Kokoro loaded", flush=True)
    print("Main function entered!", flush=True)

# ... (rest of the unchanged code from your initial script) ...


# ========== EXTENDED MAIN LOOP: HOOK INTEGRATIONS ==========
def handle_user_interaction(speaker, history, conversation_mode="auto"):
    """
    Enhanced user interaction handler with better feedback and reliability
    """
    
    # Get current context
    current_datetime = "2025-06-27 13:28:27"
    current_user = "Daveydrz"
    
    # Clean audio state with better timeout handling
    if buddy_talking.is_set():
        timeout = time.time() + 0.8  # Slightly longer for natural completion
        while buddy_talking.is_set() and time.time() < timeout:
            time.sleep(0.01)
        
        # If still talking after timeout, force stop
        if buddy_talking.is_set():
            if DEBUG:
                print("[Buddy] Force stopping previous audio...")
            stop_playback()
            time.sleep(0.1)  # Brief pause for cleanup
    
    # Clear interrupt flags and show ready state
    vad_triggered.clear()
    full_duplex_interrupt_flag.clear()
    
    if DEBUG:
        print("\n" + "="*60)
        print(f"🎤 BUDDY IS LISTENING - {current_datetime} - User: {current_user}")
        print("="*60)
    else:
        print("🎤 Listening...")

    # Listen for user input with better error handling
    try:
        question = fast_listen_and_transcribe(history)
    except NameError as name_err:
        if DEBUG:
            print(f"[Buddy] NameError in transcription: {name_err}")
        print("🔊 Sorry, I had a system error. Try again.")
        return True
    except Exception as listen_err:
        if DEBUG:
            print(f"[Buddy] Listen error: {listen_err}")
        print("🔊 Sorry, I had trouble hearing you. Try again.")
        return True

    if DEBUG:
        print(f"[Buddy DEBUG] Transcribed: {repr(question)}")
        print("-" * 60)

    # Handle interruption case
    if full_duplex_interrupt_flag.is_set():
        if DEBUG:
            print("[Buddy] 🚨 Interrupt detected - processing new input")
        full_duplex_interrupt_flag.clear()
        # Continue processing the new input below
    
    # Skip empty or invalid input with better feedback
    if not question or not question.strip() or question.strip() == "...":
        if DEBUG:
            print("[Buddy] ❌ No valid speech detected.")
        else:
            print("🤔 I didn't catch that. Could you repeat?")
        return True

    # Early filtering for noise/gibberish
    try:
        if is_noise_or_gibberish(question):
            if DEBUG:
                print(f"[Buddy] 🗑️ Filtered gibberish: {question!r}")
            else:
                print("🔇 That sounded like background noise. Try speaking more clearly.")
            return True
    except Exception as filter_err:
        if DEBUG:
            print(f"[Buddy] Filter error: {filter_err}")
        # Continue processing if filter fails

    # Handle interrupt commands immediately
    INTERRUPT_PHRASES = ["stop", "buddy stop", "przerwij", "cancel", "shut up", "quiet", "silence"]
    try:
        if any(phrase in question.lower() for phrase in INTERRUPT_PHRASES):
            if DEBUG:
                print(f"[Buddy] 🛑 Interrupt command: {question}")
            stop_playback()
            speak_async("Okay, I'll stop.", lang="en")
            return True
    except Exception as interrupt_err:
        if DEBUG:
            print(f"[Buddy] Interrupt handling error: {interrupt_err}")

    # Handle end conversation
    try:
        if should_end_conversation(question):
            if DEBUG:
                print(f"[Buddy] 👋 Ending conversation: {question}")
            speak_async("Goodbye! Talk to you later.", lang="en")
            return False
    except Exception as end_err:
        if DEBUG:
            print(f"[Buddy] End conversation error: {end_err}")

    # Ensure speaker is set to current user if not already
    if speaker != current_user:
        if DEBUG:
            print(f"[Buddy] 🔄 Setting speaker to current user: {current_user}")
        speaker = current_user
        set_last_user(speaker)
        history[:] = load_user_history(speaker)

    # Speaker detection and switching with improved error handling
    try:
        if os.path.exists("temp_input.wav"):
            audio_np, sample_rate = sf.read("temp_input.wav", dtype='int16')
            
            # Ensure proper format for speaker detection
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]  # Take first channel if stereo
                
            if len(audio_np) > 16000:  # At least 1 second of audio
                try:
                    new_speaker, confidence = detect_active_speaker(audio_np)

                    if new_speaker and new_speaker != speaker and confidence > 0.82:
                        if DEBUG:
                            print(f"[Buddy] 🔄 Speaker switch: {speaker} → {new_speaker} (conf={confidence:.2f})")
                        speaker = new_speaker
                        set_last_user(speaker)
                        history[:] = load_user_history(speaker)
                        speak_async(f"Hi {new_speaker}! I recognize you.", lang="en")
                        return True  # Give them a moment after greeting
                        
                    elif not new_speaker and confidence > 0.82:
                        if DEBUG:
                            print(f"[Buddy] 👤 New speaker detected (conf={confidence:.2f})")
                        speak_async("I don't recognize your voice. What's your name?", lang="en")
                        
                        # Wait for response with timeout
                        playback_queue.join()
                        time.sleep(0.5)  # Brief pause
                        
                        try:
                            name_response = fast_listen_and_transcribe([])
                            if name_response and name_response.strip() != "...":
                                name = name_response.strip().title()
                                # Clean name (remove common words)
                                name_words = name.split()
                                clean_name = " ".join([w for w in name_words if w.lower() not in ["my", "name", "is", "i'm", "i", "am"]])
                                name = clean_name if clean_name else name
                            else:
                                name = f"User{int(time.time())}"
                        except:
                            name = f"User{int(time.time())}"
                        
                        # Register new user
                        try:
                            audio_embedding = generate_embedding_from_audio(audio_np.astype(np.float32) / 32768.0)
                            if audio_embedding is not None:
                                known_users[name] = audio_embedding.tolist()
                                with open(known_users_path, "w", encoding="utf-8") as f:
                                    json.dump(known_users, f, indent=2, ensure_ascii=False)
                            
                            set_last_user(name)
                            speaker = name
                            history[:] = load_user_history(name)
                            speak_async(f"Nice to meet you, {name}!", lang="en")
                            return True
                        except Exception as reg_err:
                            if DEBUG:
                                print(f"[Buddy] Registration error: {reg_err}")
                            speak_async("Welcome! I'll just call you friend for now.", lang="en")
                            
                except Exception as speaker_detect_err:
                    if DEBUG:
                        print(f"[Buddy] Speaker detection processing error: {speaker_detect_err}")
                    # Continue with current speaker
                        
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] 🔧 Speaker detection file error: {e}")
        # Continue with current speaker

    # Enhanced language detection
    try:
        # Extended heuristics for better language detection
        common_en = [
            "how are you", "what is", "who are you", "tell me", "can you", "what's",
            "where", "when", "why", "how", "do you", "are you", "have you",
            "i want", "i need", "please", "thank you", "hello", "hi", "hey"
        ]
        common_pl = [
            "jak się", "co to", "kim jesteś", "powiedz mi", "czy możesz", "gdzie",
            "kiedy", "dlaczego", "jak", "czy", "chcę", "potrzebuję", "proszę",
            "dziękuję", "cześć", "witaj", "siema"
        ]
        
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in common_en):
            lang = "en"
        elif any(phrase in question_lower for phrase in common_pl):
            lang = "pl"
        else:
            # Fallback to automatic detection
            try:
                detected = detect_langs(question)
                if detected and len(detected) > 0:
                    lang = detected[0].lang
                    if lang not in ["en", "pl", "it"]:
                        lang = "en"  # Default fallback
                else:
                    lang = "en"
            except:
                lang = "en"  # Final fallback
                
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] 🌍 Language detection failed: {e}")
        lang = "en"

    if DEBUG:
        print(f"[Buddy] 🗣️ Detected language: {lang}")

    # FIXED: Much more specific time/date detection
    def is_time_date_question(text):
        """Check if this is specifically asking for time or date"""
        text_lower = text.lower().strip()
        
        # Exact time questions
        time_patterns = [
            r'\bwhat time is it\b',
            r'\bwhat\'s the time\b',
            r'\btell me the time\b',
            r'\bcurrent time\b',
            r'\bwhat time\b.*\bnow\b',
            r'\btime now\b'
        ]
        
        # Exact date questions  
        date_patterns = [
            r'\bwhat date is it\b',
            r'\bwhat\'s the date\b',
            r'\btell me the date\b',
            r'\bcurrent date\b',
            r'\bwhat day is it\b',
            r'\bwhat\'s today\b',
            r'\btoday\'s date\b'
        ]
        
        # Check for exact matches
        for pattern in time_patterns + date_patterns:
            if re.search(pattern, text_lower):
                return True
                
        # Very specific question starters about time/date
        if (text_lower.startswith(('what time', 'what date', 'what day')) and 
            len(text.split()) <= 6):  # Keep it short and specific
            return True
            
        return False

    # Only handle SPECIFIC time/date questions
    if is_time_date_question(question):
        if DEBUG:
            print("[Buddy] 🕐 Detected specific time/date question")
            
        if any(word in question.lower() for word in ['time', 'clock']):
            response = f"It's currently 1:28 PM UTC, or {current_datetime} in full format."
        elif any(word in question.lower() for word in ['date', 'day', 'today']):
            response = f"Today is Thursday, June 27th, 2025."
        else:
            response = f"The current date and time is {current_datetime} UTC - that's Thursday, June 27th, 2025 at 1:28 PM."
        
        speak_async(response, lang)
        
        # Add to history
        history.append({
            "user": question,
            "buddy": response,
            "timestamp": time.time(),
            "lang": lang,
            "service": "datetime"
        })
        try:
            save_user_history(speaker, history)
        except Exception as save_err:
            if DEBUG:
                print(f"[Buddy] History save error: {save_err}")
        return True

    # Acknowledge input with subtle feedback
    try:
        # Only play chime for longer questions to avoid annoying short interactions
        if len(question.split()) >= 3:
            play_chime()
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] 🔔 Chime error: {e}")

    # Update memories and context in background to avoid blocking
    def update_memories():
        try:
            update_thematic_memory(speaker, question)
            update_user_memory(speaker, question)
            update_dynamic_knowledge_from_text(speaker, question)
        except Exception as mem_err:
            if DEBUG:
                print(f"[Buddy] Memory update error: {mem_err}")
    
    # Run memory updates in background
    threading.Thread(target=update_memories, daemon=True).start()
    
    # Get personality traits and context
    try:
        traits = evolve_personality(speaker, question)
        ctx = conversation_contexts.setdefault(speaker, ConversationContext())
        ctx.update(question)
    except Exception as ctx_err:
        if DEBUG:
            print(f"[Buddy] Context error: {ctx_err}")
        traits = load_personality_traits()
        ctx = ConversationContext()

    # Extract important information with better error handling
    try:
        important_dates = extract_important_dates(question)
        if important_dates:
            event = extract_event(question)
            for date in important_dates:
                add_important_date(speaker, date, event or "unknown event")

        # Extract preferences
        pref_match = re.search(r"\b(i (like|love|enjoy|prefer|hate|dislike)) ([\w\s\-]+)", question.lower())
        if pref_match:
            pref = pref_match.group(3).strip()
            add_long_term_memory(speaker, "preference_" + pref.replace(" ", "_"), pref_match.group(1))
    except Exception as extract_err:
        if DEBUG:
            print(f"[Buddy] Information extraction error: {extract_err}")

    # Handle special intents with immediate responses
    try:
        intent = detect_user_intent(question)
        if intent:
            reply = handle_intent_reaction(intent)
            if reply:
                try:
                    emotion, _ = analyze_emotion(question)
                    reply = adjust_emotional_response(reply, emotion)
                except:
                    pass  # Continue without emotion adjustment if it fails
                    
                speak_async(reply, lang)
                
                # Add to history for continuity
                history.append({
                    "user": question,
                    "buddy": reply,
                    "timestamp": time.time(),
                    "lang": lang,
                    "intent": intent
                })
                try:
                    save_user_history(speaker, history)
                except Exception as save_err:
                    if DEBUG:
                        print(f"[Buddy] History save error: {save_err}")
                return True
    except Exception as intent_err:
        if DEBUG:
            print(f"[Buddy] Intent handling error: {intent_err}")

    # Handle mood commands
    try:
        mood = detect_mood_command(question)
        if mood:
            session_emotion_mode[speaker] = mood
            reply = f"Alright, I'll be {mood} now!"
            try:
                emotion, _ = analyze_emotion(question)
                reply = adjust_emotional_response(reply, emotion)
            except:
                pass
                
            speak_async(reply, lang)
            
            # Add to history
            history.append({
                "user": question,
                "buddy": reply,
                "timestamp": time.time(),
                "lang": lang,
                "mood_change": mood
            })
            try:
                save_user_history(speaker, history)
            except Exception as save_err:
                if DEBUG:
                    print(f"[Buddy] History save error: {save_err}")
            return True
    except Exception as mood_err:
        if DEBUG:
            print(f"[Buddy] Mood handling error: {mood_err}")

    # Handle specific service requests with better error handling
    style = {"emotion": "neutral"}
    
    try:
        if should_get_weather(question):
            if DEBUG:
                print("[Buddy] 🌤️ Processing weather request...")
            location = extract_location_from_question(question)
            forecast = get_weather(location, lang)
            try:
                emotion, _ = analyze_emotion(question)
                forecast = adjust_emotional_response(forecast, emotion)
            except:
                pass
            speak_async(forecast, lang, style)
            
            # Add to history
            history.append({
                "user": question,
                "buddy": forecast,
                "timestamp": time.time(),
                "lang": lang,
                "service": "weather"
            })
            try:
                save_user_history(speaker, history)
            except Exception as save_err:
                if DEBUG:
                    print(f"[Buddy] History save error: {save_err}")
            return True
    except Exception as weather_err:
        if DEBUG:
            print(f"[Buddy] Weather error: {weather_err}")

    try:
        if should_handle_homeassistant(question):
            if DEBUG:
                print("[Buddy] 🏠 Processing home automation...")
            answer = handle_homeassistant_command(question)
            if answer:
                try:
                    emotion, _ = analyze_emotion(question)
                    answer = adjust_emotional_response(answer, emotion)
                except:
                    pass
                speak_async(answer, lang, style)
                
                # Add to history
                history.append({
                    "user": question,
                    "buddy": answer,
                    "timestamp": time.time(),
                    "lang": lang,
                    "service": "home_assistant"
                })
                try:
                    save_user_history(speaker, history)
                except Exception as save_err:
                    if DEBUG:
                        print(f"[Buddy] History save error: {save_err}")
                return True
    except Exception as ha_err:
        if DEBUG:
            print(f"[Buddy] Home Assistant error: {ha_err}")

    try:
        if should_search_internet(question):
            if DEBUG:
                print("[Buddy] 🌐 Processing internet search...")
            result = search_internet(question, lang)
            try:
                emotion, _ = analyze_emotion(question)
                result = adjust_emotional_response(result, emotion)
            except:
                pass
            speak_async(result, lang, style)
            
            # Add to history
            history.append({
                "user": question,
                "buddy": result,
                "timestamp": time.time(),
                "lang": lang,
                "service": "internet_search"
            })
            try:
                save_user_history(speaker, history)
            except Exception as save_err:
                if DEBUG:
                    print(f"[Buddy] History save error: {save_err}")
            return True
    except Exception as search_err:
        if DEBUG:
            print(f"[Buddy] Internet search error: {search_err}")

    # Add narrative bookmark
    try:
        add_narrative_bookmark(speaker, question)
    except Exception as bookmark_err:
        if DEBUG:
            print(f"[Buddy] Bookmark error: {bookmark_err}")

    # Main LLM processing with enhanced error handling
    if DEBUG:
        print(f"[Buddy] 🧠 Processing with LLM: {question!r}")
    
    llm_start_time = time.time()
    
    try:
        # Use the optimized streaming function
        ask_llama3_streaming(
            question=question,
            name=speaker,
            history=history,
            lang=lang,
            conversation_mode=conversation_mode,
            style=style,
            speaker_traits=traits,
            speaker_context=ctx
        )
        
        if DEBUG:
            print(f"[TIMING] ⏱️ LLM processing time: {time.time() - llm_start_time:.2f}s")
            
    except Exception as llm_err:
        if DEBUG:
            print(f"[Buddy] LLM processing error: {llm_err}")
        
        # Fallback response for LLM errors
        fallback_responses = {
            "en": "Sorry, I'm having trouble thinking right now. Could you try asking again?",
            "pl": "Przepraszam, mam problemy z myśleniem. Możesz spróbować ponownie?",
            "it": "Scusa, ho problemi a pensare ora. Puoi riprovare?"
        }
        fallback = fallback_responses.get(lang, fallback_responses["en"])
        speak_async(fallback, lang, style)
        
        # Add fallback to history
        history.append({
            "user": question,
            "buddy": fallback,
            "timestamp": time.time(),
            "lang": lang,
            "error": "llm_error"
        })
        try:
            save_user_history(speaker, history)
        except Exception as save_err:
            if DEBUG:
                print(f"[Buddy] History save error: {save_err}")

    # Occasional flavor responses (with reduced frequency)
    try:
        if random.random() < 0.08:  # Further reduced to 8%
        
            flavor_line = flavor_response(ctx)
            # Delay to avoid overlapping with main response
            threading.Timer(2.0, lambda: speak_async(flavor_line, lang)).start()
    except Exception as flavor_err:
        if DEBUG:
            print(f"[Buddy] Flavor response error: {flavor_err}")

    return True

def ask_llama3_streaming(question, name, history, lang=DEFAULT_LANG, conversation_mode="auto", style=None, speaker_traits=None, speaker_context=None):
    """
    Optimized streaming LLM function with immediate response generation
    """
    
    if DEBUG:
        print(f"[Buddy] Starting streaming response for: {question[:50]}...")
    
    # Update memories first (in background to avoid blocking)
    threading.Thread(target=update_thematic_memory, args=(name, question), daemon=True).start()
    
    # Load context efficiently
    topics = get_frequent_topics(name, top_n=3)
    user_tones = {"Daveydrz": "friendly", "Dawid": "friendly", "Anna": "professional", "Guest": "friendly"}
    tone_style = user_tones.get(name, "friendly")
    reply_length = decide_reply_length(question, conversation_mode)
    emotion_mode = session_emotion_mode.get(name)
    beliefs = load_buddy_beliefs()
    bookmarks = get_narrative_bookmarks(name)
    recent_tone = get_recent_user_tone(history)

    # Load extended context
    long_term = get_long_term_memory(name) or {}
    dynamic_knowledge = load_dynamic_knowledge().get(name, {})
    personality_traits = speaker_traits or load_personality_traits()
    context = speaker_context or conversation_contexts.setdefault(name, ConversationContext())
    context_topic = context.get_last_topic()
    context_summary = context.get_topic_summary(context_topic) if context_topic else ""

    # Build optimized system message
    personality = build_personality_prompt(tone_style, emotion_mode, beliefs, recent_tone, bookmarks)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    
    # Streamlined system message for faster processing
    sys_msg = f"""{personality}

CRITICAL INSTRUCTIONS:
- Always respond in {lang_name}
- Use natural, conversational language (1-2 sentences unless more detail requested)
- No markdown, code blocks, or special formatting
- Be immediate and engaging
- Current date: 2025-06-27
- User timezone: UTC
"""

    # Add personality context (keep concise)
    if personality_traits:
        key_traits = [k for k, v in personality_traits.items() if v > 0.6][:3]
        if key_traits:
            sys_msg += f"Your personality emphasis: {', '.join(key_traits)}\n"

    # Add user interests (limited to avoid bloat)
    if topics:
        sys_msg += f"User interests: {', '.join(topics[:2])}\n"
    
    # Add relevant facts
    facts = build_user_facts(name)
    if facts:
        sys_msg += f"Key facts: {' '.join(facts[:2])}\n"
    
    # Add conversation context if relevant
    if context_topic and len(context_summary) > 10:
        sys_msg += f"Current topic: {context_topic}. Context: {context_summary[:100]}...\n"

    # Add relevant long-term memories (only if directly related)
    if long_term:
        relevant_memories = []
        question_words = set(question.lower().split())
        for key, value in long_term.items():
            key_words = set(key.lower().replace('_', ' ').split())
            if question_words.intersection(key_words):
                relevant_memories.append(f"{key.replace('_', ' ')}: {str(value)[:50]}")
        if relevant_memories:
            sys_msg += f"Relevant memories: {'; '.join(relevant_memories[:2])}\n"

    # Build message array efficiently
    messages = [{"role": "system", "content": sys_msg}]
    
    # Add recent history (keep minimal for speed)
    recent_history = history[-2:] if len(history) > 2 else history
    for h in recent_history:
        if isinstance(h, dict) and "user" in h and "buddy" in h:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["buddy"]})
    
    # Add current question
    messages.append({"role": "user", "content": question})

    if DEBUG:
        print("[Buddy][LLM] System message preview:")
        print(sys_msg[:200] + "..." if len(sys_msg) > 200 else sys_msg)

    # Initialize response tracking
    full_text = ""
    response_started = False

    try:
        start_time = time.time()
        
        if DEBUG:
            print("[Buddy] Starting LLM streaming request...")

        # Call the streaming function with optimized parameters
        full_text = ask_llama3_openai_streaming(
            messages,
            model="llama3",
            max_tokens=80,  # Increased slightly for better responses
            temperature=0.6,  # Slightly higher for more personality
            lang=lang,
            style=style or {"emotion": "neutral"}
        )

        if DEBUG:
            generation_time = time.time() - start_time
            print(f"[TIMING] LLM streaming completed in {generation_time:.2f}s")
            if full_text:
                print(f"[Buddy] Generated response: {full_text[:100]}...")

        # Check if we got a valid response
        if full_text and full_text.strip():
            response_started = True
            
            # Clean up the response
            full_text = full_text.strip()
            
            # Remove any unwanted artifacts
            full_text = re.sub(r'^(Buddy:|Assistant:)\s*', '', full_text, flags=re.IGNORECASE)
            full_text = re.sub(r'```.*?```', '', full_text, flags=re.DOTALL)
            full_text = full_text.strip()

        else:
            print("[Buddy] Warning: Empty LLM response received")

    except requests.exceptions.Timeout:
        print("[Buddy] LLM request timed out")
        full_text = "Sorry, I'm thinking a bit slowly right now. Can you repeat that?"
        speak_async(full_text, lang=lang, style=style)
        
    except requests.exceptions.ConnectionError:
        print("[Buddy] LLM connection failed")
        full_text = "I'm having trouble connecting my thoughts right now. Give me a moment."
        speak_async(full_text, lang=lang, style=style)
        
    except Exception as e:
        print(f"[Buddy] LLM error: {type(e).__name__}: {e}")
        full_text = "Something's not quite right in my head right now. Try asking again?"
        speak_async(full_text, lang=lang, style=style)

    # Handle the response
    if full_text and full_text.strip():
        # Update conversation history
        history_entry = {
            "user": question,
            "buddy": full_text,
            "timestamp": time.time(),
            "lang": lang
        }
        history.append(history_entry)
        
        # Update recent responses tracking
        normalized_response = full_text.strip().lower()
        LAST_FEW_BUDDY.append(normalized_response)
        if len(LAST_FEW_BUDDY) > 5:
            LAST_FEW_BUDDY.pop(0)
        
        # Save history asynchronously to avoid blocking
        if not FAST_MODE:
            threading.Thread(
                target=save_user_history, 
                args=(name, history), 
                daemon=True
            ).start()
        
        if DEBUG:
            print(f"[Buddy] Response added to history. Total entries: {len(history)}")
            
    else:
        # Fallback for completely empty responses
        if DEBUG:
            print("[Buddy] No valid response generated, using fallback")
        
        fallback_responses = {
            "en": "I'm not sure how to respond to that. Could you rephrase?",
            "pl": "Nie jestem pewien jak odpowiedzieć. Możesz przeformułować?",
            "it": "Non sono sicuro di come rispondere. Potresti riformulare?"
        }
        fallback = fallback_responses.get(lang, fallback_responses["en"])
        speak_async(fallback, lang=lang, style=style)
        
        # Still add to history for continuity
        history.append({
            "user": question,
            "buddy": fallback,
            "timestamp": time.time(),
            "lang": lang,
            "fallback": True
        })

    return full_text

if __name__ == "__main__":
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=audio_playback_worker, daemon=True).start()
    main()