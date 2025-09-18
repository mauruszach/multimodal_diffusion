#!/usr/bin/env python3
"""
Configuration validation script for the Audio-Visual Diffusion model.

This script validates the structure and values of configuration files.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

class ConfigValidator:
    """Validates configuration files against a schema."""
    
    def __init__(self):
        self.schema = {
            'mvp.yaml': {
                'required': [
                    'experiment', 'seed', 'device', 'mixed_precision',
                    'paths.video_root', 'paths.audio_root', 'paths.out_root',
                    'data.clip_seconds', 'data.hop_seconds',
                    'video.fps', 'video.size',
                    'audio.sr', 'audio.representation'
                ],
                'rules': [
                    ('video.size', lambda x: len(x) == 2 and all(isinstance(i, int) for i in x), 
                     'must be a list of two integers [height, width]'),
                    ('mixed_precision', lambda x: x in ['fp32', 'fp16', 'bf16'],
                     'must be one of: fp32, fp16, bf16'),
                    ('device', lambda x: x in ['cuda', 'cpu'],
                     'must be either "cuda" or "cpu"')
                ]
            },
            'a2v.yaml': {
                'required': [
                    'experiment', 'seed', 'device', 'mixed_precision',
                    'paths.samples_dir', 'paths.ckpt_path',
                    'sampling.prompt_modality', 'sampling.guidance_scale.video'
                ],
                'rules': [
                    ('sampling.prompt_modality', lambda x: x == 'audio',
                     'must be "audio" for audio-to-video')
                ]
            },
            'v2a.yaml': {
                'required': [
                    'experiment', 'seed', 'device', 'mixed_precision',
                    'paths.samples_dir', 'paths.ckpt_path',
                    'sampling.prompt_modality', 'sampling.guidance_scale.audio'
                ],
                'rules': [
                    ('sampling.prompt_modality', lambda x: x == 'video',
                     'must be "video" for video-to-audio')
                ]
            }
        }
    
    def validate(self, config_path: str) -> List[str]:
        """Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            List of error messages, empty if valid
        """
        errors = []
        config_name = os.path.basename(config_path)
        
        if config_name not in self.schema:
            return [f"No validation schema for {config_name}"]
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return [f"Failed to load {config_name}: {str(e)}"]
        
        schema = self.schema[config_name]
        
        # Check required fields
        for field in schema.get('required', []):
            if not self._nested_get(config, field.split('.')):
                errors.append(f"Missing required field: {field}")
        
        # Check validation rules
        for field, validator, message in schema.get('rules', []):
            value = self._nested_get(config, field.split('.'))
            if value is not None and not validator(value):
                errors.append(f"Invalid value for {field}: {message}")
        
        return errors
    
    def _nested_get(self, d: Dict, keys: List[str]) -> Any:
        """Safely get a nested value from a dictionary."""
        for key in keys:
            if not isinstance(d, dict) or key not in d:
                return None
            d = d[key]
        return d

def expand_env_vars(config_path: str) -> Dict:
    """Load a YAML config file with environment variable expansion."""
    def expand_env(loader, node):
        value = loader.construct_scalar(node)
        return os.path.expandvars(value)
    
    yaml.add_constructor('!env', expand_env, yaml.SafeLoader)
    
    with open(config_path, 'r') as f:
        content = os.path.expandvars(f.read())
        return yaml.safe_load(content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config_file> [config_file ...]")
        sys.exit(1)
    
    validator = ConfigValidator()
    has_errors = False
    
    for config_path in sys.argv[1:]:
        print(f"\nValidating {config_path}...")
        errors = validator.validate(config_path)
        
        if not errors:
            print("  ✓ Valid configuration")
        else:
            has_errors = True
            for error in errors:
                print(f"  ✗ {error}")
    
    if has_errors:
        sys.exit(1)

if __name__ == "__main__":
    main()
