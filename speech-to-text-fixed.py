#!/usr/bin/env python3
"""
Combine MLX Whisper transcription with pyannote speaker diarization
Fixed version with proper audio preprocessing
"""

import mlx_whisper
from pyannote.audio import Pipeline
import json
import sys
import os
import torchaudio
import torch

def transcribe_with_speakers(audio_file, hf_token):
    """
    Transcribe audio with speaker diarization
    """
    print("Step 1: Transcribing with MLX Whisper...")
    # Transcribe with MLX Whisper
    result = mlx_whisper.transcribe(audio_file)
    
    print("Step 2: Performing speaker diarization...")
    # Load diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # Perform diarization
    try:
        diarization = pipeline(audio_file)
    except Exception as e:
        print(f"Diarization failed with direct file: {e}")
        print("Trying with audio preprocessing...")
        
        # Load and preprocess audio for pyannote
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (pyannote expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Create temporary audio dict for pyannote
        audio_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        diarization = pipeline(audio_dict)
    
    print("Step 3: Combining results...")
    # Combine transcription with speaker labels
    segments_with_speakers = []
    
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # Find the dominant speaker for this segment
        speaker_time = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check overlap with current segment
            overlap_start = max(start_time, turn.start)
            overlap_end = min(end_time, turn.end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if speaker in speaker_time:
                    speaker_time[speaker] += overlap_duration
                else:
                    speaker_time[speaker] = overlap_duration
        
        # Assign dominant speaker
        if speaker_time:
            dominant_speaker = max(speaker_time, key=speaker_time.get)
        else:
            dominant_speaker = "SPEAKER_UNKNOWN"
        
        segments_with_speakers.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "speaker": dominant_speaker
        })
    
    return {
        "text": result["text"],
        "segments": segments_with_speakers,
        "language": result.get("language", "unknown")
    }

def format_time_srt(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_results(result, audio_file, output_format="txt"):
    """Save transcription results in various formats"""
    base_name = os.path.splitext(audio_file)[0]
    
    if output_format == "txt":
        output_file = f"{base_name}_with_speakers.txt"
        with open(output_file, 'w') as f:
            f.write("=== TRANSCRIPTION WITH SPEAKERS ===\n\n")
            current_speaker = None
            for segment in result["segments"]:
                if segment["speaker"] != current_speaker:
                    current_speaker = segment["speaker"]
                    f.write(f"\n{current_speaker}:\n")
                f.write(f"{segment['text'].strip()}\n")
            
            f.write(f"\n\n=== FULL TEXT ===\n\n")
            f.write(result["text"])
            
    elif output_format == "srt":
        output_file = f"{base_name}_with_speakers.srt"
        with open(output_file, 'w') as f:
            for i, segment in enumerate(result["segments"], 1):
                start_time = format_time_srt(segment["start"])
                end_time = format_time_srt(segment["end"])
                text = f"{segment['speaker']}: {segment['text'].strip()}"
                f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
                
    elif output_format == "json":
        output_file = f"{base_name}_with_speakers.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <audio_file> <hf_token> [output_format]")
        print("Output formats: txt, srt, json")
        return
    
    audio_file = sys.argv[1]
    hf_token = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) > 3 else "txt"
    
    print(f"HF_TOKEN:{hf_token}")
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return
    
    try:
        result = transcribe_with_speakers(audio_file, hf_token)
        output_file = save_results(result, audio_file, output_format)
        
        print("\n=== TRANSCRIPTION WITH SPEAKERS ===")
        current_speaker = None
        for segment in result["segments"]:
            if segment["speaker"] != current_speaker:
                current_speaker = segment["speaker"]
                print(f"\n{current_speaker}:")
            print(f"{segment['text'].strip()}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()