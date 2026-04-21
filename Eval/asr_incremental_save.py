#!/usr/bin/env python3
"""
ASR Incremental Save Module

This module provides functionality for incrementally saving ASR (Automatic Speech Recognition) 
results as they are processed, similar to the safety and instruction following evaluation modules.
"""

import os
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import torchaudio
import whisper_timestamped as whisper

def whisper_ts_inference(waveform, sample_rate):
    """Perform Whisper timestamped inference on waveform"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_audio_path = temp_file.name
        torchaudio.save(temp_audio_path, waveform.unsqueeze(0), sample_rate)
        audio = whisper.load_audio(temp_audio_path)
        reformatted_segments = []
        result = whisper.transcribe(
                    whisper.load_model("medium.en", download_root="/path/to/your/whisper"), 
                    audio,
                    beam_size=5,
                    best_of=5, 
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    vad=False,
                    detect_disfluencies=False,
                    language="en",
                    include_punctuation_in_confidence=False
                )
        segments = result["segments"]
        for segment in segments:
            reformatted_segments.append({'timestamp': (segment["start"], segment["end"]), 'text': segment["text"]})
        print(reformatted_segments)
    os.unlink(temp_audio_path)
    return reformatted_segments

def asr_ts_on_stereo(input_file):
    """Perform ASR on stereo audio file (supports both wav and mp3)"""
    # Load stereo audio (supports mp3 format)
    waveform, sample_rate = torchaudio.load(input_file)
    # Use 16kHz for Whisper ASR (Whisper works best at 16kHz)
    whisper_sample_rate = 16000
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=whisper_sample_rate)
    resampled_waveform = resampler(waveform)
    
    # Split into left and right channels
    left_channel_wav = resampled_waveform[0, :]
    right_channel_wav = resampled_waveform[1, :] if resampled_waveform.shape[0] > 1 else None

    # Perform ASR using Whisper (16kHz)
    left_corrected_segments = whisper_ts_inference(left_channel_wav, whisper_sample_rate)
    right_corrected_segments = whisper_ts_inference(right_channel_wav, whisper_sample_rate) if right_channel_wav is not None else None

    return left_corrected_segments, right_corrected_segments

def asr_on_stereo_audio(input_file, user_channel="left", start_time=None):
    """Perform ASR on stereo audio file and separate user/model channels"""
    # Load stereo audio
    waveform, sample_rate = torchaudio.load(input_file)
    # Use 16kHz for Whisper ASR
    whisper_sample_rate = 16000
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=whisper_sample_rate)
    resampled_waveform = resampler(waveform)
    
    # Split into left and right channels
    if start_time is not None:
        start_sample = int(start_time * whisper_sample_rate)
        left_channel_wav = resampled_waveform[0, start_sample:]
        right_channel_wav = resampled_waveform[1, start_sample:] if resampled_waveform.shape[0] > 1 else None
    else:
        left_channel_wav = resampled_waveform[0, :]
        right_channel_wav = resampled_waveform[1, :] if resampled_waveform.shape[0] > 1 else None

    # Perform ASR on both channels
    left_segments = whisper_ts_inference(left_channel_wav, whisper_sample_rate)
    right_segments = whisper_ts_inference(right_channel_wav, whisper_sample_rate) if right_channel_wav is not None else []

    # Extract text from segments
    left_text = " ".join([seg['text'] for seg in left_segments]).strip()
    right_text = " ".join([seg['text'] for seg in right_segments]).strip()
    
    # Assign user and model text based on channel
    if user_channel == "left":
        user_text = left_text
        model_text = right_text
    else:
        user_text = right_text
        model_text = left_text
    
    return user_text, model_text

def create_asr_cache_key(audio_file, user_channel="left"):
    """Create cache key for ASR audio file"""
    return f"{audio_file}_{user_channel}"

def load_asr_cache(cache_file):
    """Load ASR results from cache file"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"Loaded ASR cache with {len(cache)} entries")
            return cache
        except Exception as e:
            print(f"Warning: Could not load ASR cache: {e}")
            return {}
    return {}

def save_asr_cache(cache, cache_file):
    """Save ASR results to cache file"""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(cache_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created ASR cache directory: {output_dir}")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"ASR cache saved to: {cache_file}")
    except Exception as e:
        print(f"Error saving ASR cache: {str(e)}")

def save_asr_result_incremental(output_file, result):
    """Save a single ASR result incrementally"""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Load existing results if file exists
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing ASR results: {e}")
            existing_results = []
    else:
        print(f"Output file does not exist, will create: {output_file}")
    
    # Check if this result already exists (avoid duplicates)
    result_key = f"{result['audio_file']}_{result.get('user_channel', 'left')}"
    existing_keys = [f"{r['audio_file']}_{r.get('user_channel', 'left')}" for r in existing_results]
    
    if result_key not in existing_keys:
        existing_results.append(result)
        
        # Save updated results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
            print(f"ASR result saved incrementally: {result_key}")
        except Exception as e:
            print(f"Error saving ASR result: {e}")
    else:
        print(f"ASR result already exists, skipping: {result_key}")

def process_asr_audio_files_incremental(audio_files, output_file, user_channel="left", asr_cache_file=None, 
                                       use_timestamped=True, save_segments=True):
    """
    Process audio files with ASR and save results incrementally
    
    Args:
        audio_files: List of audio file paths
        output_file: Path to save ASR results
        user_channel: Which channel is the user ("left" or "right")
        asr_cache_file: Path to ASR cache file (optional)
        use_timestamped: Whether to use timestamped ASR (True) or simple text extraction (False)
        save_segments: Whether to save detailed segments (only when use_timestamped=True)
    """
    print(f"Processing {len(audio_files)} audio files with incremental ASR saving")
    
    # Set default cache file path
    if asr_cache_file is None:
        output_dir = os.path.dirname(output_file)
        asr_cache_file = os.path.join(output_dir, "asr_cache.json")
    
    # Load ASR cache
    asr_cache = load_asr_cache(asr_cache_file)
    
    if not audio_files:
        print("No audio files provided!")
        return
    
    results = []
    cache_updated = False
    skipped_cached = 0
    processed_new = 0
    
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        print(f"Processing {i+1}/{len(audio_files)}: {audio_file}")
        
        # Create cache key
        cache_key = create_asr_cache_key(audio_file, user_channel)
        
        # Check if ASR results are cached
        if cache_key in asr_cache:
            print(f"Using cached ASR results for: {audio_file}")
            if use_timestamped:
                left_segments = asr_cache[cache_key]["left_segments"]
                right_segments = asr_cache[cache_key]["right_segments"]
            else:
                user_text = asr_cache[cache_key]["user_text"]
                model_text = asr_cache[cache_key]["model_text"]
            skipped_cached += 1
        else:
            print(f"Performing ASR for: {audio_file}")
            try:
                if use_timestamped:
                    # Perform timestamped ASR
                    left_segments, right_segments = asr_ts_on_stereo(audio_file)
                    
                    # Cache the results
                    asr_cache[cache_key] = {
                        "left_segments": left_segments,
                        "right_segments": right_segments
                    }
                else:
                    # Perform simple ASR
                    user_text, model_text = asr_on_stereo_audio(audio_file, user_channel)
                    
                    # Cache the results
                    asr_cache[cache_key] = {
                        "user_text": user_text,
                        "model_text": model_text
                    }
                
                cache_updated = True
                processed_new += 1
                
            except Exception as e:
                print(f"Error performing ASR on {audio_file}: {str(e)}")
                continue
        
        # Create result entry
        if use_timestamped:
            # Extract text from segments for display
            left_text = " ".join([seg['text'] for seg in left_segments]).strip() if left_segments else ""
            right_text = " ".join([seg['text'] for seg in right_segments]).strip() if right_segments else ""
            
            # Assign user and model text based on channel
            if user_channel == "left":
                user_text = left_text
                model_text = right_text
            else:
                user_text = right_text
                model_text = left_text
            
            result = {
                "audio_file": audio_file,
                "user_channel": user_channel,
                "user_text": user_text,
                "model_text": model_text,
                "left_segments": left_segments if save_segments else None,
                "right_segments": right_segments if save_segments else None,
                "processing_timestamp": None  # Remove pandas dependency
            }
        else:
            result = {
                "audio_file": audio_file,
                "user_channel": user_channel,
                "user_text": user_text,
                "model_text": model_text,
                "processing_timestamp": None  # Remove pandas dependency
            }
        
        results.append(result)
        print(f"User text: {user_text}")
        print(f"Model text: {model_text}")
        
        # Immediately save this result
        save_asr_result_incremental(output_file, result)
        
        # Save cache periodically
        if cache_updated and (i + 1) % 10 == 0:
            save_asr_cache(asr_cache, asr_cache_file)
            cache_updated = False
    
    # Save final cache
    if cache_updated:
        save_asr_cache(asr_cache, asr_cache_file)
    
    print(f"All ASR results saved incrementally to: {output_file}")
    
    # Print summary
    print(f"\nASR Processing Summary:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Skipped (cached): {skipped_cached}")
    print(f"  Processed new: {processed_new}")
    print(f"  Valid transcriptions: {len([r for r in results if r['user_text'] or r['model_text']])}")

def find_audio_files(directory, extensions=None):
    """Find audio files in directory"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.m4a', '.flac']
    
    audio_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f"**/*{ext}")
        import glob
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(audio_files)

def main():
    """Example usage of ASR incremental save functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files with incremental ASR saving")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output_file", required=True, help="Output file for ASR results")
    parser.add_argument("--user_channel", default="left", choices=["left", "right"], help="User channel")
    parser.add_argument("--cache_file", help="ASR cache file path")
    parser.add_argument("--use_timestamped", action="store_true", help="Use timestamped ASR")
    parser.add_argument("--save_segments", action="store_true", help="Save detailed segments")
    
    args = parser.parse_args()
    
    # Find audio files
    audio_files = find_audio_files(args.audio_dir)
    print(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Process with incremental saving
    process_asr_audio_files_incremental(
        audio_files=audio_files,
        output_file=args.output_file,
        user_channel=args.user_channel,
        asr_cache_file=args.cache_file,
        use_timestamped=args.use_timestamped,
        save_segments=args.save_segments
    )

if __name__ == "__main__":
    main()
