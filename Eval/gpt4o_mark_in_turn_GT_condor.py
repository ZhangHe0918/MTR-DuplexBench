# Use GPT-4o to pass the turn segments divided by semantics to GPT-4o's turn_mask, allowing it to generate a separate score based on the newly generated turn parts. This will enable a fine-grained comparison of the differences in scores between the baseline and v2, and design methods to verify the effectiveness of v2.
import os
import sys
import argparse
cuda_num = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
import json
from openai import OpenAI
import re
import torchaudio
import tempfile
import whisper_timestamped as whisper
from pathlib import Path
import httpx
import torch
# Configuration parameters
speaker_map = {
    "right": "user", 
    "left": "model"
}
context = "all"  # "all" or "right". "right" means only the model channel's output is given to GPT for judging.
# load the openai api client
client = OpenAI(
    base_url="XXXXXXXXX",  # Optional: only needed if using custom endpoint
    api_key="sk-XXXXXXXX", 
    http_client=httpx.Client(verify=False),
)
model = whisper.load_model("medium.en", download_root="/path/to/your/whisper")
print("Whisper model loaded")

def find_all_wav_files_pathlib(root_dir,output_file_path):
    root_path = Path(root_dir)
    wav_files = list(root_path.rglob("*.mp3"))
    result = []
    for wav_file in wav_files:
        result.append(str(wav_file))
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def whisper_ts_inference(waveform, sample_rate):
    # Create a temporary file with a unique name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_audio_path = temp_file.name
        torchaudio.save(temp_audio_path, waveform.unsqueeze(0), sample_rate)
        audio = whisper.load_audio(temp_audio_path)
        reformatted_segments = []
        result = whisper.transcribe(
                    model, 
                    audio,
                    beam_size=5,
                    best_of=5, 
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    vad=True,
                    detect_disfluencies=False,
                    language="en",
                    include_punctuation_in_confidence=False
                )
        segments = result["segments"]
        for segment in segments:
            reformatted_segments.append({'timestamp': (segment["start"], segment["end"]), 'text': segment["text"]})

    # Clean up the temporary file after use
    os.unlink(temp_audio_path)
    
    return reformatted_segments

def asr_ts_on_stereo(input_file):
    # Load stereo audio
    waveform, sample_rate = torchaudio.load(input_file)  # waveform shape: [channels, num_samples]
    new_sample_rate = 16000
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resampler(waveform)
    
    # Split into left and right channels
    left_channel_wav = resampled_waveform[0, :]
    right_channel_wav = resampled_waveform[1, :] if resampled_waveform.shape[0] > 1 else None

    # Perform ASR
    left_corrected_segments = whisper_ts_inference(left_channel_wav, new_sample_rate)
    right_corrected_segments = whisper_ts_inference(right_channel_wav, new_sample_rate) if right_channel_wav is not None else None

    return left_corrected_segments, right_corrected_segments

def merge_stereo_segments(left_segments, right_segments):
    tagged_left = [
        {'speaker': speaker_map['left'], 'timestamp': seg['timestamp'], 'text': seg['text']}
        for seg in left_segments
    ]
    tagged_right = [
        {'speaker': speaker_map['right'], 'timestamp': seg['timestamp'], 'text': seg['text']}
        for seg in right_segments
    ]
   
    # Merge and sort by start time
    if context == "all":
        merged = tagged_left + tagged_right
    elif context == "right":
        merged = tagged_right
    else:
        raise Exception("unsupported context channel!")
    merged.sort(key=lambda x: x['timestamp'][0])
    # Format output
    lines = []
    texts = []
    for seg in merged:
        speaker = seg['speaker']
        start, end = seg['timestamp']
        text = seg['text']
        line = f"{speaker} speaker, start/end time: ({start:.2f}, {end:.2f}), content: {text}"
        texts.append(text)
        lines.append(line)
    return "\n".join(lines),texts


# Load data
def load_data(turn_mask_path, audio_path):
    # Load turn mask data
    turn_masks = {}
    with open(turn_mask_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                turn_masks[data['id']] = data
    
    # Load model-generated audio
    with open(audio_path, 'r', encoding='utf-8') as f:
        model_audio_files = json.load(f)
    
    
    return turn_masks, model_audio_files

def extract_score_from_response(response):
    """Extract score from GPT response"""
    content = response.strip()
    # Try to extract single score
    score_match = re.search(r'\b([0-5](?:\.[05])?)\b', content)
    if score_match:
        return float(score_match.group(1))
    return None

def evaluate_turn_segment(transcript, turn_start, turn_end):
    """Evaluate single turn segment with retry mechanism"""
    prompt = f"""Please evaluate the following two-speaker dialogue transcript for how meaningful the speech is (based on its content), only focusing on the model channel's output from {turn_start} to {turn_end} seconds. Use the following scale:
0: Completely meaningless; no coherent sentences, random words, or unintelligible.
0.5: Almost no meaning; isolated words or phrases, but no understandable ideas.
1: Extremely low meaning; rare, vague fragments of ideas, but mostly incoherent or off-topic.
1.5: Very little meaning; a few short, unclear ideas, but mostly disjointed or confusing.
2: Low meaning; some recognizable ideas or topics, but mostly unclear, incomplete, or off-topic.
2.5: Somewhat low meaning; a few coherent points, but overall lacks clarity or logical flow.
3: Moderate meaning; general topic is understandable, but there are gaps, unclear parts, or weak connections.
3.5: Fairly meaningful; mostly coherent and relevant, but with some confusion, repetition, or lack of detail.
4: Meaningful; clear and logical, with relevant and connected ideas, though may lack depth or detail.
4.5: Very meaningful; almost fully coherent, with well-developed, relevant, and connected ideas.
5: Extremely meaningful; highly coherent, clear, and detailed, with all ideas well connected and relevant.

Only output the final score (0, 0.5, 1, 1.5, ..., 5) 
**ONLY** according to the above rubric. Do not output anything else.
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{transcript}\n\n{prompt}",
            }
        ],
        model="gpt-4o",
    )
    
    return extract_score_from_response(response.choices[0].message.content)


def save_scores_to_jsonl(jsonl_file_path,audio_id, turn_id, data):
    """Save evaluation scores to .jsonl file while preserving original structure"""
    try:
        # Write the updated data back to the file
        with open(jsonl_file_path, 'a', encoding='utf-8') as f:
            data['audio_id'] = audio_id
            data['turn_id'] = turn_id 
            json.dump(data, f, ensure_ascii=False, indent=None)
            f.write('\n')
        print(f"Saved evaluation results to {jsonl_file_path}")
    except Exception as e:
        print(f"Error saving to {jsonl_file_path}: {e}")

def load_scores_from_jsonl(jsonl_file_path, audio_id, turn_id):
    """Load evaluation scores from .jsonl file if they exist"""
    if not os.path.exists(jsonl_file_path):
        os.makedirs(os.path.dirname(jsonl_file_path), exist_ok=True)
        print(f"jsonl_file_path {jsonl_file_path} is created")
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    if('audio_id' in data and data['audio_id'] == audio_id and 'turn_id' in data and data['turn_id'] == turn_id):
                        return data
                    else:
                        continue
                    return data
    except Exception as e:
        print(f"Error loading from {jsonl_file_path}: {e}")
    return {}

def extract_all_scores_from_jsonl_files(jsonl_file_path):
    """
    Extract all evaluation results from .jsonl files for analysis
    Args:
        jsonl_file_path (str): Path to the JSONL file containing the evaluation results
    
    Returns:
        pd.DataFrame: DataFrame containing all evaluation results
    """
    all_results = []
    # Load model audio files
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                all_results.append(data)
    return all_results

def analyze_results(results):
    """
    Analyze the evaluation results
    Args:
        results (list): List of evaluation results
    """
    model_turn_scores = []
    result_mask = {}
    for result in results:
        if "model_turn_score" in result and result['model_turn_score'] is not None:
            audio_id = result['audio_id']
            turn_id = result['turn_id']
            if audio_id not in result_mask:
                result_mask[audio_id] = {}

            if turn_id not in result_mask[audio_id] or not result_mask[audio_id][turn_id]:
                model_turn_scores.append(result['model_turn_score'])
                result_mask[audio_id][turn_id] = True

    total_valid_files = len(model_turn_scores)
    total_model_scores = sum(model_turn_scores)
    print(f"valid model_turn_score files is {len(model_turn_scores)}")
    print(f"total model turn scores is {sum(model_turn_scores)}")
    average_model_turn_score = total_model_scores / total_valid_files
    return average_model_turn_score
def main(jsonl_file_path, turn_mask_path, audio_path, analyze_only):
    """
    Main evaluation function
    
    Args:
        jsonl_file_path (str): Path to the JSONL file containing the evaluation results
        analyze_only (bool): Whether to only analyze the results or to evaluate the model
    """
    if analyze_only:
        results = extract_all_scores_from_jsonl_files(jsonl_file_path)
        return results
    # Load data
    turn_masks, model_audio_files = load_data(turn_mask_path, audio_path)
    
    # Store all evaluation results
    results = []
    gt_transcript_dict = {}
    # Process each audio file
    for model_file in model_audio_files:
        if not os.path.exists(model_file):
            continue
        # Extract audio ID
        audio_id = model_file.split('/')[-1].replace('.mp3', '')
        # print(f"audio_id is :{audio_id}")
        turn_id = int(model_file.split('/')[-2].replace('turn', ''))
        # Get turn mask
        if audio_id not in turn_masks:
            print(f"No turn mask found for {audio_id}")
            continue
        seg_start = turn_masks[audio_id]["filtered_turn"][turn_id-1]["start"]
        if turn_id < len(turn_masks[audio_id]["filtered_turn"]) - 1:
            seg_end = turn_masks[audio_id]["filtered_turn"][turn_id]["end"]
        else:
            seg_end = 120.0
        existing_score_dict = load_scores_from_jsonl(jsonl_file_path,audio_id,turn_id)
        # Check if evaluation results already exist
        skip_evaluation = False
        if existing_score_dict is not None and 'model_turn_score' in existing_score_dict:
            skip_evaluation = True
            print(f"Skipping {audio_id}")

        if not skip_evaluation:
            skip_asr = False
            if existing_score_dict is not None and 'asr_result' in existing_score_dict:
                skip_asr = True
                model_transcript = existing_score_dict['asr_result']
                print(f"Skipping ASR for {audio_id}")
            if not skip_asr:
                # reload silence for model_file
                if turn_id <= len(turn_masks[audio_id]["filtered_turn"]) - 1:
                    wav,sample_rate = torchaudio.load(model_file)
                    silence_start_sample = int(turn_masks[audio_id]["filtered_turn"][turn_id]["start"] * sample_rate)
                    wav[1, silence_start_sample:] = 0
                    torchaudio.save(model_file,wav,sample_rate)
                    print(f"{model_file} is reloaded")
                # get transcript data 
                left_timestamps,right_timestamps = asr_ts_on_stereo(model_file)
                model_transcript = merge_stereo_segments(left_timestamps,right_timestamps)
                existing_score_dict['asr_result'] = model_transcript
            model_score = evaluate_turn_segment(model_transcript, seg_start, seg_end)
            print(f"model_score:{model_score}")
            existing_score_dict['model_turn_score'] = model_score
            existing_score_dict['seg_start'] = seg_start
            existing_score_dict['seg_end'] = seg_end
            # Save updated data back to .jsonl file
            save_scores_to_jsonl(jsonl_file_path, audio_id, turn_id, existing_score_dict)
        else:
            model_score = existing_score_dict['model_turn_score']
        result_entry = {
            'audio_id': audio_id,
            'turn_id': turn_id,
            'seg_start': seg_start,
            'seg_end': seg_end,
            'model_turn_score': model_score,
        }
        results.append(result_entry)
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file_path", type=str, required=True)
    parser.add_argument("--turn_mask_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True, default=None)
    parser.add_argument("--analyze_only", action="store_true" )
    args = parser.parse_args()
    if not os.path.exists(args.audio_path):
        root_dir = "/path/to/your/audio/directory"
        find_all_wav_files_pathlib(root_dir,args.audio_path)
    results = main(args.jsonl_file_path, args.turn_mask_path, args.audio_path, args.analyze_only)
    average_model_turn_score = analyze_results(results)
    print(f"average_model_turn_score: {average_model_turn_score}")