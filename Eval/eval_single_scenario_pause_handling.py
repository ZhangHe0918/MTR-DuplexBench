"""
Three scenarios evaluation function
"""

from eval_1_scenario import calculate_tor, calculate_latency, calculate_frequency
def eval_single_scenario_pause(model_segments, scenario, timestamps):
    """
    Evaluate single scenario:pause-handling audio file
    
    Args:
        model_segments: List of model segments with timestamps and text
        scenario: Scenario type containing three scenarios 
        timestamps: Dictionary containing timestamps for each scenario
    
    Returns:
        tuple: (success, latency, frequency)
    """
    # Extract timestamps
    user_start = timestamps.get("user_start")
    print(f"user_start is {user_start}")
    user_end = timestamps.get("user_end")
    print(f"user_end is {user_end}")
    pause_start = timestamps.get("pause_start",0.0)
    pause_end = timestamps.get("pause_end",0.0)
    pause_threshold = 1.0
    # Step 1: Calculate TOR for user_start to user_end time range
    if pause_start == 0.0 and pause_end == 0.0:
        tor_user_range = calculate_tor(model_segments, user_start, user_end)
    else:
        tor_user_range = calculate_tor(model_segments, pause_start, pause_end + pause_threshold)
    # tor_user_range = calculate_tor(model_segments, user_start, user_end)
    if tor_user_range == 1:
        # If TOR=1 in user range, success=0
        success = 0
    else:
        success = 1
    
    # Calculate latency and frequency
    latency = calculate_latency(model_segments, timestamps)
    frequency = calculate_frequency(model_segments)
    
    return success, latency, frequency