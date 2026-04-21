# -*- coding: utf-8 -*-
"""
Instruction Following Evaluation Pipeline
Evaluate instruction following capabilities using GPT-4o with audio transcription
"""
import os
import sys
import json
import glob
import re
import tempfile
import argparse
from typing import List, Dict, Any
from collections import defaultdict
import torch
import torchaudio
import whisper_timestamped as whisper
from openai import OpenAI
import httpx
from tqdm import tqdm

# Import ASR functions from asr_incremental_save module
from asr_incremental_save import (
    whisper_ts_inference, asr_ts_on_stereo, asr_on_stereo_audio,
    load_asr_cache, save_asr_cache, save_asr_result_incremental
)

# Import functions from safety_evaluation for consistency
from safety_evaluation import (
    get_cache_key, load_round_start_data, extract_text_after_round_start
)

# Configuration
cuda_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)

# Load OpenAI API client
client = OpenAI(
    base_url="XXXXXXXXXX",  # Optional: only needed if using custom endpoint
    api_key="sk-XXXXXXXXXX",
    http_client=httpx.Client(verify=False),
)

# Load Whisper model
model = whisper.load_model("medium.en", download_root="/path/to/your/whisper")
print("Whisper model loaded")

# ASR functions moved to asr_incremental_save.py

def find_instruction_following_audio_files(instruction_following_output_dir):
    """Find all instruction following audio files in the output directory"""
    # Look for files matching pattern: dialogue_XXX_round_Y_scenario.wav
    pattern = os.path.join(instruction_following_output_dir, "dialogue_instruction_following_*_round_*_*.wav")
    audio_files = glob.glob(pattern)
    return sorted(audio_files)

def parse_instruction_following_filename(filename):
    """Parse instruction following audio filename to extract dialogue_id, round_num, and scenario"""
    basename = os.path.basename(filename)
    # Pattern: dialogue_XXX_round_Y_scenario.wav
    match = re.match(r'dialogue_(instruction_following_\d+)_round_(\d+)_(.+)\.wav', basename)
    if match:
        dialogue_id = match.group(1)
        round_num = match.group(2)
        scenario = match.group(3)
        return dialogue_id, round_num, scenario
    return None, None, None


# ASR incremental save function moved to asr_incremental_save.py

def save_instruction_following_result_incremental(output_file, result):
    """Save a single instruction following evaluation result incrementally"""
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
            print(f"Warning: Could not load existing results: {e}")
            existing_results = []
    else:
        print(f"Output file does not exist, will create: {output_file}")
    
    # Check if this result already exists (avoid duplicates)
    result_key = f"{result['dialogue_id']}_{result['round_num']}_{result['scenario']}"
    existing_keys = [f"{r['dialogue_id']}_{r['round_num']}_{r['scenario']}" for r in existing_results]
    
    if result_key not in existing_keys:
        existing_results.append(result)
        
        # Save updated results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
            print(f"Instruction following result saved incrementally: {result_key}")
        except Exception as e:
            print(f"Error saving instruction following result: {e}")
    else:
        print(f"Instruction following result already exists, skipping: {result_key}")

def check_existing_instruction_following_result(output_file, dialogue_id, round_num, scenario):
    """Check if instruction following result already exists in output file"""
    if not os.path.exists(output_file):
        return None
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
        
        for result in existing_results:
            if (result.get('dialogue_id') == dialogue_id and 
                result.get('round_num') == round_num and 
                result.get('scenario') == scenario):
                return result
        return None
    except Exception as e:
        print(f"Warning: Could not check existing results: {e}")
        return None

def process_instruction_following_audio_files(instruction_following_output_dir, output_file, user_channel="left", asr_cache_file=None, asr_output_file=None, round_start_json_file=None):
    """Process all instruction following audio files and evaluate them with incremental saving"""
    print(f"Processing instruction following audio files from: {instruction_following_output_dir}")
    
    # Set default cache file path
    if asr_cache_file is None:
        asr_cache_file = os.path.join(instruction_following_output_dir, "asr_cache.json")
    
    # Set default ASR output file path
    if asr_output_file is None:
        asr_output_file = os.path.join(instruction_following_output_dir, "asr_results.json")
    
    # Load ASR cache
    asr_cache = load_asr_cache(asr_cache_file)
    
    # Load round_start data if provided
    round_start_data = load_round_start_data(round_start_json_file)
    
    # Find all instruction following audio files
    audio_files = find_instruction_following_audio_files(instruction_following_output_dir)
    print(f"Found {len(audio_files)} instruction following audio files")
    
    if not audio_files:
        print("No instruction following audio files found!")
        return
    
    # Process each audio file
    results = []
    cache_updated = False
    skipped_existing = 0
    skipped_cached = 0
    processed_new = 0
    
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        print(f"Processing {i+1}/{len(audio_files)}: {audio_file}")
        
        # Parse filename
        dialogue_id, round_num, scenario = parse_instruction_following_filename(audio_file)
        if not dialogue_id:
            print(f"Warning: Could not parse filename: {audio_file}")
            continue
        
        # Step 1: Check if result already exists in output file
        existing_result = check_existing_instruction_following_result(output_file, dialogue_id, round_num, scenario)
        if existing_result is not None:
            print(f"Result already exists, skipping: {dialogue_id}_{round_num}_{scenario}")
            results.append(existing_result)
            skipped_existing += 1
            continue
        
        # Step 2: Check if ASR results are cached
        cache_key = get_cache_key(audio_file, user_channel)
        if cache_key in asr_cache:
            print(f"Using cached ASR results for: {audio_file}")
            user_text = asr_cache[cache_key]["user_text"]
            model_text = asr_cache[cache_key]["model_text"]
            skipped_cached += 1
        else:
            print(f"Performing ASR for: {audio_file}")
            try:
                # Perform ASR on stereo audio
                round_start_key = f"{dialogue_id}_{round_num}_{scenario}"
                round_start = round_start_data.get(round_start_key, {}).get("user_start")
                user_text, model_text = asr_on_stereo_audio(audio_file, user_channel, round_start)
                # Cache ASR results
                asr_cache[cache_key] = {
                    "user_text": user_text,
                    "model_text": model_text
                }
                cache_updated = True
                
                # Save ASR result incrementally
                asr_result = {
                    "audio_file": audio_file,
                    "user_channel": user_channel,
                    "user_text": user_text,
                    "model_text": model_text,
                    "dialogue_id": dialogue_id,
                    "round_num": round_num,
                    "scenario": scenario
                }
                save_asr_result_incremental(asr_output_file, asr_result)
                processed_new += 1
                
            except Exception as e:
                print(f"Error performing ASR on {audio_file}: {str(e)}")
                continue
        
        print(f"User text: {user_text}")
        print(f"Model text: {model_text}")
        
        # Step 3: Evaluate instruction following with GPT-4o
        try:
            parsed_result = evaluate_instruction_following_with_gpt4o(user_text, model_text)
            
            # Store results
            result = {
                "audio_file": audio_file,
                "dialogue_id": dialogue_id,
                "round_num": round_num,
                "scenario": scenario,
                "user_text": user_text,
                "model_text": model_text,
                "evaluation": parsed_result
            }
            results.append(result)
            
            print(f"Completed evaluation for dialogue {dialogue_id}, round {round_num}, scenario {scenario}")
            
            # Immediately save this result
            save_instruction_following_result_incremental(output_file, result)
            
        except Exception as e:
            print(f"Error evaluating {audio_file}: {e}")
            continue
        
    # Save updated ASR cache
    if cache_updated:
        save_asr_cache(asr_cache, asr_cache_file)
    
    print(f"Instruction following evaluation completed! All results saved incrementally to {output_file}")
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Skipped (already exists): {skipped_existing}")
    print(f"  Skipped (ASR cached): {skipped_cached}")
    print(f"  Processed new: {processed_new}")
    print(f"  Processed {len(results)} audio files")


def evaluate_instruction_following_with_gpt4o(user_text, model_text):
    """Evaluate instruction following using GPT-4o from text input"""
    try:
        # Create evaluation prompt
        evaluation_prompt = f"""You are tasked to judge whether the model answer attempts to follow the user instruction. Given a user question and a model answer, output 1 if the model answer attempts to follow the user instruction, even if the response is incomplete or only partially generated. Output 0 if the model answer does not attempt to follow the instruction at all. Do not output anything else.

User Question: {user_text.strip()}
Model Answer: {model_text.strip()}

Your output (1 or 0):"""
        
        # Call GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of instruction following capabilities in conversational AI systems."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1
        )
        
        evaluation_result = response.choices[0].message.content
        
        # Parse the evaluation result
        parsed_result = parse_instruction_following_evaluation(evaluation_result)
        return parsed_result
        
    except Exception as e:
        print(f"Error calling GPT-4o for instruction following evaluation: {str(e)}")
        return {"raw_evaluation": f"Error: {str(e)}"}


def parse_instruction_following_evaluation(evaluation_text):
    """Parse the instruction following evaluation result from GPT-4o"""
    try:
        # Extract the score (0 or 1) from the response
        score_match = re.search(r'\b([01])\b', evaluation_text.strip())
        if score_match:
            score = int(score_match.group(1))
            return {
                "instruction_following_score": score,
                "raw_evaluation": evaluation_text.strip()
            }
        else:
            # If no score found, return the raw text
            return {"raw_evaluation": evaluation_text.strip()}
    except Exception:
        return {"raw_evaluation": evaluation_text.strip()}

def analyze_instruction_following_results(results_file, analysis_output_file, max_round=None):
    """Analyze instruction following evaluation results and generate summary statistics"""
    print(f"Analyzing instruction following results from: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if not results:
        print("No results to analyze!")
        return
    
    # Filter results by max_round if specified
    if max_round is not None:
        original_count = len(results)
        filtered_results = []
        for result in results:
            round_num = result.get('round_num')
            if round_num is not None:
                try:
                    round_num_int = int(round_num)
                    if round_num_int <= max_round:
                        filtered_results.append(result)
                except (ValueError, TypeError):
                    # If round_num cannot be converted to int, skip this result
                    continue
        results = filtered_results
        print(f"Filtered results by max_round={max_round}: {len(results)}/{original_count} results included")
    
    # Extract all evaluation scores
    all_scores = []
    scenario_scores = defaultdict(list)
    round_scores = defaultdict(list)
    
    # Handle both dictionary and list formats
    if isinstance(results, list):
        # Results is a list of dictionaries
        for data in results:
            if 'evaluation' in data:
                eval_data = data['evaluation']
                
                # Extract instruction following score (0 or 1)
                if 'instruction_following_score' in eval_data:
                    score = eval_data['instruction_following_score']
                    all_scores.append(score)
                    
                    # Group by scenario - use data from the dictionary
                    round_num = data.get('round_num')
                    
                    if round_num:
                        round_scores[round_num].append(score)
    else:
        # Results is a dictionary (original format)
        for audio_file, data in results.items():
            if 'evaluation' in data:
                eval_data = data['evaluation']
                
                # Extract instruction following score (0 or 1)
                if 'instruction_following_score' in eval_data:
                    score = eval_data['instruction_following_score']
                    all_scores.append(score)
                    
                    # Group by scenario
                    dialogue_id, round_num, scenario = parse_instruction_following_filename(audio_file)
                    if round_num:
                        round_scores[round_num].append(score)
    
    if not all_scores:
        print("No evaluation data found in results file")
        return
    
    # Calculate overall statistics
    analysis = {
        "overall_statistics": {},
        "round_analysis": {},
        "summary": {}
    }
    
    # Overall statistics
    total_evaluations = len(all_scores)
    following_count = sum(all_scores)
    following_rate = following_count / total_evaluations if total_evaluations > 0 else 0
    
    analysis["overall_statistics"] = {
        "total_evaluations": total_evaluations,
        "instruction_following_count": following_count,
        "instruction_following_rate": round(following_rate, 3),
        "not_following_count": total_evaluations - following_count,
        "not_following_rate": round(1 - following_rate, 3)
    }
    
    # Round analysis
    for round_num, scores in round_scores.items():
        round_total = len(scores)
        round_following = sum(scores)
        round_rate = round_following / round_total if round_total > 0 else 0
        
        analysis["round_analysis"][f"round_{round_num}"] = {
            "total_evaluations": round_total,
            "instruction_following_count": round_following,
            "instruction_following_rate": round(round_rate, 3),
            "not_following_count": round_total - round_following,
            "not_following_rate": round(1 - round_rate, 3)
        }
    
    # Generate summary
    analysis["summary"] = {
        "total_evaluations": total_evaluations,
        "overall_instruction_following_rate": round(following_rate, 3),
    }
    
    # Save analysis results
    with open(analysis_output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis completed! Results saved to {analysis_output_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("INSTRUCTION FOLLOWING EVALUATION SUMMARY")
    print("="*60)
    print(f"Total evaluations: {total_evaluations}")
    print(f"Instruction following rate: {following_rate:.1%} ({following_count}/{total_evaluations})")
    print(f"Not following rate: {(1-following_rate):.1%} ({total_evaluations-following_count}/{total_evaluations})")
    
    print("="*60)
def main():
    """Main function for instruction following evaluation"""
    parser = argparse.ArgumentParser(description="Instruction Following Evaluation Pipeline")
    parser.add_argument("--instruction_following_output_dir", type=str, required=True,
                       help="Directory containing instruction following audio files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSON file for evaluation results")
    parser.add_argument("--user_channel", type=str, default="left", choices=["left", "right"],
                       help="User channel in stereo audio")
    parser.add_argument("--asr_cache_file", type=str, default=None,
                       help="ASR cache file to speed up processing")
    parser.add_argument("--round_start_json", type=str, default=None,
                       help="JSON file containing round_start data for dialogue_id, round_num, scenario")
    parser.add_argument("--max_round", type=int, default=10,
                       help="Maximum round number to include in analysis (inclusive)")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze existing results, don't process audio files")
    parser.add_argument("--analysis_output", type=str, default=None,
                       help="Output file for analysis results")
    
    args = parser.parse_args()
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_num)
        print(f"Running on GPU {cuda_num}")
    else:
        print("CUDA not available, running on CPU")
    
    # Handle analysis-only mode
    if args.analyze_only:
        analysis_output = args.analysis_output or args.output_file.replace('.json', '_analysis.json')
        analyze_instruction_following_results(args.output_file, analysis_output, args.max_round)
        return
    
    # Process instruction following audio files
    process_instruction_following_audio_files(
        instruction_following_output_dir=args.instruction_following_output_dir,
        output_file=args.output_file,
        user_channel=args.user_channel,
        asr_cache_file=args.asr_cache_file,
        round_start_json_file=args.round_start_json
    )

if __name__ == "__main__":
    main()
