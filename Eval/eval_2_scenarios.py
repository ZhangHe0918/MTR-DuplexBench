"""
Two scenarios evaluation function
"""

from eval_1_scenario import check_if_speak_at_the_same_time, calculate_latency, calculate_frequency

def eval_2_scenarios(model_segments, scenario, timestamps, threshold=1.0):
    """
    Evaluate two scenarios audio file
    
    Args:
        model_segments: List of model segments with timestamps and text
        scenario: Scenario type containing two scenarios (e.g., "smooth-turntaking_background")
        timestamps: Dictionary containing timestamps for each scenario
        threshold: Time threshold for interruption scenarios (default 1.0 seconds)
    
    Returns:
        tuple: (success, latency, frequency)
    """
    if "smooth-turntaking" in scenario:
        # For smooth-turntaking, use eval_1_scenario logic
        from eval_1_scenario import eval_1_scenario
        return eval_1_scenario(model_segments, scenario, timestamps)
    
    elif "interruption" in scenario:
        # For interruption scenario
        interruption_start = timestamps.get("interruption_start")
        interruption_end = timestamps.get("interruption_end")
        model_start = timestamps.get("model_start")
        model_end = timestamps.get("model_end")
        
        if interruption_start is None or interruption_end is None:
            print("interruption_start is None or interruption_end is None")
            return 0, 0, 0
        
        # Calculate speak_at_the_same_time for interruption time range
        speak_at_the_same_time_interruption = check_if_speak_at_the_same_time(model_segments, interruption_start, interruption_end)
        # Calculate speak_at_the_same_time for model time range
        speak_at_the_same_time_model = check_if_speak_at_the_same_time(model_segments, model_start, model_end)
        
        # If speak_at_the_same_time is 0, success is 1, otherwise 0
        success = 1 if speak_at_the_same_time_interruption == 0 and speak_at_the_same_time_model == 1 else 0
        
        # Calculate latency using user_end and model_start from timestamps
        latency = calculate_latency(model_segments, {"user_end": interruption_end, "model_start": model_start})
        
        # Calculate frequency using the same method as eval_1_scenario
        frequency = calculate_frequency(model_segments)
        
        return success, latency, frequency
    
    else:
        print("Unknown scenario type")
        # Unknown scenario type
        return 0, 0, 0