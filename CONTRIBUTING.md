# Contributing

Thanks for helping improve METIS.

## Development Setup

1. Create a Python environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo to verify your setup:

   ```bash
   python demo_metis.py
   ```

## Project Structure

```
metis/
  core/          # Signal processing: entropy, statistics, types
  cognitive/     # Higher-order: CoT, boundary guard, curiosity
  integrations/  # Hook API for external model integration
  inference.py   # Main inference pipeline
docs/            # Philosophy, metacognition, roadmap
demo_metis.py    # Interactive cognitive demo
```

## Pull Requests

- Keep PRs focused and small when possible.
- Include a clear description of what changed and why.
- Provide reproduction steps or benchmark outputs when relevant.
- Avoid committing model weights or large datasets.

## Reporting Bugs

Please use the bug report template and include:

- GPU model and VRAM
- Python / PyTorch / Transformers version
- Model identifier
- Full logs (attach or link)

