"""Fine-tune трансформера по фолдам: OOF и test-предсказания для стекинга."""

import argparse
from pathlib import Path

from youarebot.transformer import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/transformer/rubert_tiny2.yaml")
    )
    args = parser.parse_args()
    run(args.config)
