#!/usr/bin/env python3
"""
Simple MLX Whisper transcription without speaker diarization
"""

import mlx_whisper
import sys
import json
import os

def transcribe_audio(audio_file, output_format="txt"):
    """
    Transcribe audio with MLX Whisper only
    """
    print(f"Transcribing {audio_file} with MLX Whisper...")
    
    # Transcribe with MLX Whisper
    result = mlx_whisper.transcribe(audio_file)
    
    # Generate output filename
    base_name = os.path.splitext(audio_file)[0]
    
    if output_format == "txt":
        output_file = f"{base_name}_transcription.txt"
        with open(output_file, 'w') as f:
            f.write(result['text'])
        print(f"\nTranscription saved to: {output_file}")
        
    elif output_format == "json":
        output_file = f"{base_name}_transcription.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull result saved to: {output_file}")
        
    elif output_format == "srt":
        output_file = f"{base_name}_transcription.srt"
        with open(output_file, 'w') as f:
            for i, segment in enumerate(result['segments'], 1):
                start_time = format_time(segment['start'])
                end_time = format_time(segment['end'])
                f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n")
        print(f"\nSRT file saved to: {output_file}")
    
    # Always print the transcription
    print('\n=== TRANSCRIPTION ===')
    print(result['text'])
    
    return result

def format_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_only.py <audio_file> [output_format]")
        print("Output formats: txt (default), srt, json")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "txt"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)
    
    transcribe_audio(audio_file, output_format)