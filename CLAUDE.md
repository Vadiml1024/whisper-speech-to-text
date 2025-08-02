# Whisper Speech-to-Text with Speaker Diarization

## Project Overview

This is a Python-based audio transcription project that combines MLX Whisper for speech-to-text with pyannote for speaker diarization. The project creates transcripts that identify different speakers in audio files.

## Architecture

### Core Components

1. **speech-to-text.py** - Main production script that combines MLX Whisper transcription with pyannote speaker diarization
2. **test-mlx.py** - Testing script for validating MLX Whisper API functionality
3. **.env** - Environment configuration file containing HuggingFace token

### Technology Stack

- **Python 3** - Core runtime
- **mlx_whisper** - Apple MLX-optimized Whisper for fast speech transcription on Apple Silicon
- **pyannote.audio** - Speaker diarization pipeline from HuggingFace
- **HuggingFace Hub** - Model hosting and authentication

### Data Flow

1. Audio file input
2. MLX Whisper transcription (segments with timestamps)
3. Pyannote speaker diarization (speaker segments with timestamps)
4. Overlap analysis to assign speakers to transcript segments
5. Output formatting (txt, srt, or json)

## Key Features

- **Multi-format output**: Support for plain text, SRT subtitles, and JSON
- **Speaker identification**: Assigns speaker labels to transcript segments
- **Timestamp alignment**: Matches speakers to text based on time overlap
- **Error handling**: Graceful handling of API failures and file issues

## Dependencies

The project requires these key Python packages:
- `mlx_whisper` - MLX-optimized Whisper implementation
- `pyannote.audio` - Speaker diarization toolkit
- Standard library: `json`, `sys`, `os`

## Configuration

### Environment Variables
- `HF_TOKEN` - HuggingFace authentication token (stored in .env)
  - Required for accessing pyannote speaker diarization models
  - Format: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Models Used
- MLX Whisper: Default model or `mlx-community/whisper-large-v3-mlx`
- Pyannote: `pyannote/speaker-diarization-3.1`

## Usage Patterns

### Main Script Usage
```bash
python speech-to-text.py <audio_file> <hf_token> [output_format]
```

### Test Script Usage
```bash
python test-mlx.py <audio_file>
```

### Output Files
- Generated files follow pattern: `{original_name}_with_speakers.{format}`
- Supported formats: txt, srt, json

## Development Guidelines

### File Structure
- Keep the simple, flat structure - no complex directory hierarchy needed
- Scripts are executable and can be run directly
- All dependencies should be installable via pip

### Code Patterns
- Each script is self-contained with its own main() function
- Error handling with try/catch blocks and informative error messages
- Progress indicators for long-running operations
- Modular functions for different processing steps

### Testing Strategy
- Use test-mlx.py to validate Whisper API functionality before full processing
- Test different API methods to ensure compatibility
- Validate audio file existence before processing

## Common Operations

### Adding New Output Formats
- Extend the `format_output()` function in speech-to-text.py
- Add new format case in the conditional logic
- Update command-line help text

### Debugging Transcription Issues
- Run test-mlx.py first to isolate Whisper problems
- Check HuggingFace token validity for diarization issues
- Verify audio file format compatibility

### Performance Optimization
- MLX Whisper is optimized for Apple Silicon
- Consider audio file length for memory usage
- Speaker diarization is typically the slower component

## Security Notes

- HuggingFace token is stored in .env file (not version controlled)
- Audio files are processed locally, no data sent to external services except model downloads
- pyannote models require HuggingFace authentication

## Future Enhancement Areas

- Batch processing multiple files
- Real-time audio processing
- Additional output formats (VTT, etc.)
- Speaker identification (names vs. labels)
- Audio preprocessing and enhancement
- Configuration file for model selection

## Troubleshooting

### Common Issues
1. **MLX Whisper not working**: Try different API methods in test-mlx.py
2. **Diarization failing**: Check HF_TOKEN validity and model access permissions
3. **Poor speaker separation**: Audio quality or number of speakers may affect results
4. **Memory issues**: Large audio files may require chunking

### Dependencies Installation
```bash
pip install mlx_whisper pyannote.audio
```

Note: This project appears to be in active development with recent modifications to the speech-to-text functionality.