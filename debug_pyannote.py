#!/usr/bin/env python3
"""
Debug script to identify pyannote pipeline loading issues
"""

import sys
import os
import logging
import traceback

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pyannote_loading(hf_token):
    """Test pyannote pipeline loading with detailed debugging"""
    
    logger.info("Starting pyannote debugging...")
    
    try:
        logger.info("Importing pyannote.audio...")
        from pyannote.audio import Pipeline
        logger.info("✓ pyannote.audio imported successfully")
        
        logger.info(f"Using HF token: {hf_token[:10]}...")
        
        # Test 1: Check if we can access HuggingFace Hub
        logger.info("Testing HuggingFace Hub connection...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            user_info = api.whoami()
            logger.info(f"✓ HF Hub connection OK, user: {user_info['name']}")
        except Exception as e:
            logger.error(f"✗ HF Hub connection failed: {e}")
            return False
            
        # Test 2: Check model access
        logger.info("Testing model access...")
        try:
            model_info = api.model_info("pyannote/speaker-diarization-3.1")
            logger.info(f"✓ Model accessible, ID: {model_info.id}")
        except Exception as e:
            logger.error(f"✗ Model access failed: {e}")
            return False
            
        # Test 3: Try to load pipeline with different methods
        logger.info("Attempting to load pipeline...")
        
        # Method 1: Standard loading
        try:
            logger.info("Method 1: Standard Pipeline.from_pretrained...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            logger.info("✓ Pipeline loaded successfully with Method 1!")
            return pipeline
        except Exception as e:
            logger.error(f"✗ Method 1 failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
        # Method 2: Try with token parameter
        try:
            logger.info("Method 2: Using token parameter...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            logger.info("✓ Pipeline loaded successfully with Method 2!")
            return pipeline
        except Exception as e:
            logger.error(f"✗ Method 2 failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
        # Method 3: Try with explicit cache dir
        try:
            logger.info("Method 3: Using explicit cache directory...")
            cache_dir = os.path.expanduser("~/whisper_cache")
            os.makedirs(cache_dir, exist_ok=True)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
                cache_dir=cache_dir
            )
            logger.info("✓ Pipeline loaded successfully with Method 3!")
            return pipeline
        except Exception as e:
            logger.error(f"✗ Method 3 failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
        logger.error("All methods failed to load pipeline")
        return None
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def test_with_audio(pipeline, audio_file):
    """Test diarization with actual audio file"""
    if pipeline is None:
        logger.error("No pipeline available for testing")
        return
        
    try:
        logger.info(f"Testing diarization on {audio_file}...")
        diarization = pipeline(audio_file)
        logger.info("✓ Diarization completed successfully!")
        
        # Show results
        logger.info("Diarization results:")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            logger.info(f"  {turn.start:.2f}s - {turn.end:.2f}s: {speaker}")
            
        return diarization
        
    except Exception as e:
        logger.error(f"✗ Diarization failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pyannote.py <hf_token> [audio_file]")
        sys.exit(1)
        
    hf_token = sys.argv[1]
    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    logger.info("=== PYANNOTE DEBUG SESSION ===")
    
    # Test pipeline loading
    pipeline = test_pyannote_loading(hf_token)
    
    # Test with audio if provided and pipeline loaded
    if audio_file and pipeline:
        if os.path.exists(audio_file):
            test_with_audio(pipeline, audio_file)
        else:
            logger.error(f"Audio file not found: {audio_file}")
    
    logger.info("=== DEBUG SESSION COMPLETE ===")