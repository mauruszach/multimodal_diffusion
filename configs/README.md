# Configuration Files

This directory contains configuration files for the Audio-Visual Diffusion model.

## File Structure

- `mvp.yaml`: Main training configuration
- `a2v.yaml`: Audio-to-video inference configuration
- `v2a.yaml`: Video-to-audio inference configuration

## Versioning

Configuration files follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: Backward-compatible additions
- **PATCH**: Backward-compatible bug fixes

## Environment Variables

Paths in configurations support environment variables using `${VAR_NAME}` syntax:

```yaml
paths:
  video_root: "${DATA_DIR}/video"
  audio_root: "${DATA_DIR}/audio"
```

## Validation Rules

### Video Configuration
- `video.fps` must match between training and preprocessing
- `video.size` must be divisible by `video.latent.s_down`

### Audio Configuration
- `audio.frames_per_clip * audio.frame_hop_ms / 1000` should be approximately equal to `data.clip_seconds`

## Best Practices

1. Use absolute paths or environment variables for all file paths
2. Keep configurations in version control
3. Document any non-obvious parameters
4. Test configurations in a development environment before production

## Example: Using Environment Variables

```bash
# Set environment variables
export DATA_DIR="/path/to/data"
export OUTPUT_DIR="/path/to/output"

# Run with environment variables
python train.py --config configs/mvp.yaml
```

## Configuration Updates

When updating configurations:
1. Increment the version number
2. Document changes in the file header
3. Test compatibility with existing checkpoints
4. Provide migration instructions if breaking changes are introduced
