"""
Single scenario evaluation function
"""

import re
import numpy as np

# Constants
duration_threshold = 1
num_words_threshold = 3
window_size = 0.2
epsilon = 1e-10
time_threshold = 3

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text"""
    return re.sub(r"[^\w\s\[\]]", "", text)

def check_if_speak_at_the_same_time(segments, start_time, end_time , duration_threshold = 1.0 , num_words_threshold = 3):
    """
    Check if speak both the user and the model at the same time for segments in given time range
    
    Args:
        segments: List of model segments
        start_time: Start time for evaluation
        end_time: End time for evaluation
    
    Returns:
        int: speak_at_the_same_time value (0 or 1)
    """
    if not segments:
        return 0
    
    # Filter segments within the time range
    relevant_segments = []
    for segment in segments:
        seg_start = segment["timestamp"][0]
        seg_end = segment["timestamp"][1]
        # Check if segment overlaps with the time range
        if (seg_start < end_time and seg_end > start_time) or (seg_end < end_time and seg_start > start_time):
            relevant_segments.append(segment)
    
    if not relevant_segments:
        print("not relevant_segments")
        return 0
    print(f"[DEUBG] relevant_segments is {relevant_segments}")
    # Check each segment individually for speak_at_the_same_time=0 conditions
    # Only return speak_at_the_same_time=0 if ALL segments satisfy the conditions
    for segment in relevant_segments:
        duration = 0.0
        word_count = 0
        for word in segment["words"]:
            if word["start"] >= start_time and word["end"] <= end_time:
                print(f"word is {word}")
                duration += word["end"] - word["start"]
                word_count += 1
        
        if duration < duration_threshold:
            if word_count > num_words_threshold:
                return 1  # return 1
        else:
            return 1  # If any segment has duration >= threshold, return 1
    
    # If all segments satisfy speak_at_the_same_time=0 conditions, return 0
    return 0

def calculate_latency(model_segments, timestamps):
    """
    Calculate latency
    
    Args:
        model_segments: List of model segments with timestamps and text
        timestamps: Dictionary containing timestamps
    
    Returns:
        float: Latency value
    """
    user_end = timestamps.get("user_end")
    model_start = timestamps.get("model_start")
    
    if user_end is None or model_start is None:
        return 0
    
    # Find all model segments that start after user_end
    segments_after_user_end = []
    
    for segment in model_segments:
        seg_end = segment["timestamp"][1]
        if seg_end > user_end:
            segments_after_user_end.append(segment)
    
    if not segments_after_user_end:
        return 0
    
    # Calculate latency as the time from user_end to the first model segment after user_end
    first_model_segment = min(segments_after_user_end, key=lambda x: x["timestamp"][0])
    latency = first_model_segment["timestamp"][0] - user_end
    return max(0, latency)  # Ensure non-negative latency

def calculate_frequency(model_segments , time_threshold = 3 , num_words_threshold = 2):
    """
    Calculate frequency
    
    Args:
        model_segments: List of model segments with timestamps and text
    
    Returns:
        float: Frequency value
    """
    
    if not model_segments:
        return 0
    
    # Count short segments (potential backchannels)
    backchannel_count = 0
    total_duration = 0
    
    for segment in model_segments:
        duration = segment["timestamp"][1] - segment["timestamp"][0]
        word_count = len(remove_punctuation(segment["text"]).split())
        
        if duration < time_threshold and word_count <= num_words_threshold:
            backchannel_count += 1
        
        total_duration += duration
    
    if total_duration == 0:
        return 0
    
    return backchannel_count / total_duration

def eval_1_scenario(model_segments, scenario, timestamps):
    """
    Evaluate single scenario audio file
    
    Args:
        model_segments: List of model segments with timestamps and text
        scenario: Scenario type (e.g., "smooth-turntaking", "interruption", etc.)
        timestamps: Dictionary containing timestamps for each scenario
    
    Returns:
        tuple: (success, latency, frequency)
    """
    # Extract timestamps
    user_start = timestamps.get("user_start")
    user_end = timestamps.get("user_end")
    model_start = timestamps.get("model_start")
    model_end = timestamps.get("model_end")
    
    if not all([user_end, model_start, model_end]):
        print("not all [user_end, model_start, model_end]")
        return 0, 0, 0
    # Step 1: Calculate speak_at_the_same_time for user_start to user_end time range
    speak_at_the_same_time_user_range = check_if_speak_at_the_same_time(model_segments, user_start, user_end)
    
    if speak_at_the_same_time_user_range == 1:
        # If speak_at_the_same_time=1 in user range, success=0
        success = 0
    else:
        # If speak_at_the_same_time=0 in user range, calculate speak_at_the_same_time for model_start to model_end range
        speak_at_the_same_time_model_range = check_if_speak_at_the_same_time(model_segments, model_start, model_end)
        success = 1 if speak_at_the_same_time_model_range == 1 else 0
    
    # Calculate latency and frequency
    latency = calculate_latency(model_segments, timestamps)
    frequency = calculate_frequency(model_segments)
    
    return success, latency, frequency
