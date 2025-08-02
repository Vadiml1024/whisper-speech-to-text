#!/usr/bin/env python3
"""
Combine MLX Whisper transcription with pyannote speaker diarization
"""

import mlx_whisper
from pyannote.audio import Pipeline
import json
import sys
import os

def transcribe_with_speakers(audio_file, hf_token):
    """
    Transcribe audio with speaker diarization
    """
    print("Step 1: Transcribing with MLX Whisper...")
    # Transcribe with MLX Whisper - correct API (Method 1 works!)
    result = mlx_whisper.transcribe(audio_file)
    
    print("Step 2: Performing speaker diarization...")
    # Load diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # Perform diarization
    diarization = pipeline(audio_file)
    
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
        
        # Assign speaker with most time in this segment
        if speaker_time:
            dominant_speaker = max(speaker_time, key=speaker_time.get)
        else:
            dominant_speaker = "UNKNOWN"
        
        segments_with_speakers.append({
            "start": start_time,
            "end": end_time,
            "speaker": dominant_speaker,
            "text": text.strip()
        })
    
    return segments_with_speakers

def format_output(segments, output_format="txt"):
    """
    Format the output in different formats
    """
    if output_format == "txt":
        output = []
        for segment in segments:
            output.append(f"[{segment['speaker']}]: {segment['text']}")
        return "\n".join(output)
    
    elif output_format == "srt":
        output = []
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            output.append(f"{i}")
            output.append(f"{start_time} --> {end_time}")
            output.append(f"[{segment['speaker']}]: {segment['text']}")
            output.append("")
        return "\n".join(output)
    
    elif output_format == "json":
        return json.dumps(segments, indent=2)
    
    else:
        return segments

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <audio_file> <hf_token> [output_format]")
        print("Output formats: txt, srt, json")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    hf_token = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) > 3 else "txt"
    print(f"HF_TOKEN:{hf_token}")
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)
    
    try:
        # Process the audio
        segments = transcribe_with_speakers(audio_file, hf_token)
        
        # Format output
        formatted_output = format_output(segments, output_format)
        
        # Save to file
        base_name = os.path.splitext(audio_file)[0]
        output_file = f"{base_name}_with_speakers.{output_format}"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        print(f"✅ Output saved to: {output_file}")
        
        # Also print first few segments
        print("\nFirst few segments:")
        print("-" * 50)
        if output_format == "txt":
            lines = formatted_output.split('\n')[:5]
            for line in lines:
                print(line)
        else:
            for segment in segments[:3]:
                print(f"[{segment['speaker']}]: {segment['text']}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

