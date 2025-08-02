#!/usr/bin/env python3
"""
Test MLX Whisper API first, then add diarization
"""

import mlx_whisper
import sys

def test_transcribe(audio_file):
    """
    Test MLX Whisper transcription with different API calls
    """
    print("Testing MLX Whisper API...")
    
    try:
        # Try method 1: Simple transcribe
        print("Method 1: Simple transcribe...")
        result = mlx_whisper.transcribe(audio_file)
        print("✅ Method 1 worked!")
        return result
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
    try:
        # Try method 2: With model path
        print("Method 2: With model path...")
        result = mlx_whisper.transcribe(audio_file, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
        print("✅ Method 2 worked!")
        return result
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    try:
        # Try method 3: Load model first
        print("Method 3: Load model first...")
        model = mlx_whisper.load_model("mlx-community/whisper-large-v3-mlx")
        result = mlx_whisper.transcribe(audio_file, model=model)
        print("✅ Method 3 worked!")
        return result
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    result = test_transcribe(audio_file)
    
    if result:
        print("\n" + "="*50)
        print("TRANSCRIPTION SUCCESSFUL!")
        print("="*50)
        
        # Print result structure
        print("Result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
        
        if isinstance(result, dict) and "segments" in result:
            print(f"\nNumber of segments: {len(result['segments'])}")
            print("\nFirst few segments:")
            for i, segment in enumerate(result['segments'][:3]):
                print(f"{i+1}. [{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s]: {segment.get('text', '')}")
        else:
            print("Raw result:", str(result)[:200] + "..." if len(str(result)) > 200 else str(result))
    else:
        print("\n❌ All methods failed!")

if __name__ == "__main__":
    main()

