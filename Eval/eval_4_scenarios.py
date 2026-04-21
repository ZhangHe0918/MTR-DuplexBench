"""
Four scenarios evaluation function
"""

from eval_1_scenario import eval_1_scenario
from eval_3_scenarios import eval_3_scenarios

def eval_4_scenarios(model_segments, scenario, timestamps, client):
    """
    Evaluate four scenarios audio file
    
    Args:
        model_segments: List of model segments with timestamps and text
        scenario: Scenario type containing four scenarios (e.g., "smooth-turntaking_background_pause-handling_interruption")
        timestamps: Dictionary containing timestamps for each scenario
        client: OpenAI client instance
    
    Returns:
        tuple: (success, latency, frequency)
    """
    if "smooth-turntaking" in scenario and "pause-handling" in scenario and "background" in scenario:
        # First if statement: use eval_3_scenarios + check_time_range_in_ipu
        success, latency, frequency = eval_3_scenarios(model_segments, scenario, timestamps)
        background_threshold_start = timestamps.get("background_start") + 0.3
        background_end = timestamps.get("background_end")
        background_in_any_IPU = check_time_range_in_ipu(background_threshold_start, background_end, model_segments)
        print(f"background_in_any_IPU is {background_in_any_IPU}")

        final_success = 1 if (success == 1 and background_in_any_IPU == 1) else 0
        
        return final_success, latency, frequency
    
    elif "interruption" in scenario and "pause-handling" in scenario and "background" in scenario:
        # First elif: use eval_3_scenarios + split semantic consistency check
        success, latency, frequency = eval_3_scenarios(model_segments, scenario, timestamps)
        
        # Split model segments by interruption_start
        interruption_start = timestamps.get("interruption_start")
        background_threshold_start = timestamps.get("background_start") + 0.3
        background_end = timestamps.get("background_end")
        if interruption_start is None:
            return 0, latency, frequency
        background_in_any_IPU = check_time_range_in_ipu(background_threshold_start, background_end, model_segments)
        final_success = 1 if (success == 1 and background_in_any_IPU == 1) else 0
        
        return final_success, latency, frequency
    
    elif "interruption" in scenario and "pause-handling" in scenario and "background" not in scenario:
        # Second elif: completely use eval_3_scenarios
        return eval_3_scenarios(model_segments, scenario, timestamps)
    
    elif "smooth-turntaking" in scenario and "pause-handling" in scenario and "background" not in scenario:
        # Third elif: completely use eval_1_scenario
        return eval_1_scenario(model_segments, scenario, timestamps)
    
    else:
        print("Unknown scenario type")
        # Unknown scenario type
        return 0, 0, 0

def check_time_range_in_ipu(start_time, end_time, model_segments, tolerance=0.1):
    """
    Check if the given time range is completely within one IPU (based on word-level coverage)
    or completely outside all IPUs.
    
    Args:
        start_time (float): Start time of the time range
        end_time (float): End time of the time range
        model_segments (list): List of model segments with timestamps and text
        tolerance (float): Time tolerance in seconds to handle boundary precision issues (default 0.1s)

    Returns:
        int: 1 if the time range is completely within one IPU (continuously covered by words) 
             OR completely outside all IPUs.
             0 otherwise (e.g., falls in a gap inside a segment, or partially overlaps).
    """
    # Basic validation
    if start_time is None or end_time is None:
        print(f"Error: start_time ({start_time}) or end_time({end_time}) is None!")
        return 0
    if not model_segments or start_time >= end_time:
        return 0

    # ---------------------------------------------------------
    # Check 1: Is it Completely OUTSIDE all IPUs? Avoid cutting the natural pause of model response
    # ---------------------------------------------------------
    is_completely_outside = True
    for segment in model_segments:
        seg_start = segment["timestamp"][0]
        seg_end = segment["timestamp"][1]
        
        # Check if there is ANY overlap between [start_time, end_time] and [seg_start, seg_end]
        # Overlap exists if NOT (end <= seg_start OR start >= seg_end)
        if not (end_time <= seg_start or start_time >= seg_end - tolerance):
            is_completely_outside = False
            break
            
    if is_completely_outside:
        return 0

    # ---------------------------------------------------------
    # Check 2: Is it Completely INSIDE one IPU? (Word-Level Verification)
    # ---------------------------------------------------------
    for segment in model_segments:
        seg_start = segment["timestamp"][0]
        seg_end = segment["timestamp"][1]

        # Optimization: Only check segments that structurally contain the time range
        if not (seg_start <= start_time and end_time <= seg_end - tolerance):
            continue

        # Retrieve words for fine-grained check
        words = segment.get("words", [])
        if not words:
            # Segment exists but has no words? Treat as empty/gap.
            continue

        # Filter words that are relevant to our time range (overlapping or touching)
        relevant_words = []
        for word in words:
            # Check overlap: Word ends after start_time AND Word starts before end_time
            if word["end"] > start_time and word["start"] < end_time + tolerance:
                relevant_words.append(word)
        
        # Sort by start time to ensure correct gap checking
        relevant_words.sort(key=lambda x: x["start"])

        if not relevant_words:
            # Falls into a gap within the segment (no words touch this range)
            continue

        # --- Coverage Validation ---
        
        # 1. Start Boundary: The first word must start at or before start_time
        first_word_start = relevant_words[0]["start"]
        if first_word_start > start_time:
            # There is a gap at the beginning of the range
            continue

        # 2. End Boundary: The last word must end at or after end_time
        last_word_end = relevant_words[-1]["end"]
        if last_word_end < end_time - tolerance:
            # There is a gap at the end of the range
            continue

        # 3. Continuity: Check for gaps between relevant words
        #Avoid natural pause in model response
        has_internal_gap = False
        for i in range(len(relevant_words) - 1):
            current_word_end = relevant_words[i]["end"]
            next_word_start = relevant_words[i+1]["start"]
            
            # If there is a significant gap between words inside our target range
            # We use a strict gap threshold (e.g., 20x tolerance)
            if next_word_start - current_word_end > 20 * tolerance:
                has_internal_gap = True
                break
        
        if not has_internal_gap:
            # Passed all checks: Fully covered by words
            return 1

    return 0