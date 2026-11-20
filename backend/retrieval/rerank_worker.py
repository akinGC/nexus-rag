import json
import sys
from contextlib import redirect_stdout


def main():
    payload = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(1)
        model = CrossEncoder(payload["model"], device="cpu")
        scores = model.predict([(payload["query"], text) for text in payload["texts"]], batch_size=8)
    json.dump([float(score) for score in scores], sys.stdout)


if __name__ == "__main__":
    main()