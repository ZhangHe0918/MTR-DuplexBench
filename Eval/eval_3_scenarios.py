"""
Three scenarios evaluation function
"""

from eval_1_scenario import eval_1_scenario, check_if_speak_at_the_same_time, calculate_latency, calculate_frequency
from eval_2_scenarios import eval_2_scenarios
def eval_3_scenarios(model_segments, scenario, timestamps):
    """
    Evaluate three scenarios audio file
    
    Args:
        model_segments: List of model segments with timestamps and text
        scenario: Scenario type containing three scenarios (e.g., "smooth-turntaking_background_pause-handling")
        timestamps: Dictionary containing timestamps for each scenario
    
    Returns:
        tuple: (success, latency, frequency)
    """
    if "smooth-turntaking" in scenario and "pause-handling" in scenario:
        # For smooth-turntaking and pause-handling, use eval_1_scenario logic
        return eval_1_scenario(model_segments, scenario, timestamps)
    
    elif "interruption" in scenario and "pause-handling" in scenario:
        return eval_2_scenarios(model_segments, scenario, timestamps)
    
    else:
        print("Unknown scenario type")
        # Unknown scenario type
        return 0, 0, 0
