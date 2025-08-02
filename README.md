# Audio Transcription with Speaker Diarization

A Python project that combines MLX Whisper for fast local transcription with pyannote.audio for speaker diarization, optimized for Apple Silicon.

## Features

- **Fast local transcription** using MLX Whisper (Apple Silicon optimized)
- **Speaker diarization** with pyannote.audio to identify different speakers
- **Multiple output formats**: TXT, SRT subtitles, JSON
- **Privacy-focused**: All processing done locally (only model downloads require internet)
- **Robust error handling** with fallback methods

## Requirements

- Python 3.8+
- Apple Silicon Mac (for MLX optimization)
- HuggingFace account and token with gated repository access

## Setup

1. Install dependencies:
```bash
pip install mlx-whisper pyannote.audio torch torchaudio
```

2. Get HuggingFace token:
   - Visit https://huggingface.co/settings/tokens
   - Create token with "Read access to gated repositories" permission
   - Accept user conditions for:
     - https://hf.co/pyannote/speaker-diarization-3.1
     - https://hf.co/pyannote/segmentation-3.0

3. Create `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

## Usage

### Basic transcription with speaker diarization:
```bash
python speech-to-text-fixed.py audio_file.mp3 your_hf_token
```

### Generate SRT subtitles:
```bash
python speech-to-text-fixed.py audio_file.mp3 your_hf_token srt
```

### Generate JSON output:
```bash
python speech-to-text-fixed.py audio_file.mp3 your_hf_token json
```

### Transcription only (no speaker diarization):
```bash
python transcribe_only.py audio_file.mp3
```

## Output Formats

- **TXT**: Clean text with speaker labels
- **SRT**: Subtitle file with timestamps and speaker identification
- **JSON**: Full structured data with segments, timestamps, and metadata

## Files

- `speech-to-text-fixed.py` - Main script with speaker diarization
- `transcribe_only.py` - Simple transcription without speaker identification  
- `debug_pyannote.py` - Debugging tool for pyannote issues
- `speech-to-text.py` - Original script (may have tensor size issues)
- `test-mlx.py` - MLX Whisper testing script
- `CLAUDE.md` - Development guidance for Claude Code

## Troubleshooting

Common issues and solutions are documented in `CLAUDE.md`. For debugging pyannote issues, use:
```bash
python debug_pyannote.py your_hf_token audio_file.mp3
```