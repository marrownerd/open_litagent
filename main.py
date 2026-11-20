"""CLI: search academic papers, review abstract evidence, save Markdown + JSON."""
import argparse
import json
from pathlib import Path

from agent import LiteratureAgent
from llm import DEFAULT_MODEL, GGUFModel, TransformersModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Local literature discovery and extraction agent")
    parser.add_argument("question", type=str, help="Research question in quotes")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["transformers", "gguf"], default="transformers")
    parser.add_argument("--gguf-path", help="Local Qwen2.5 GGUF file (requires --backend gguf)")
    parser.add_argument("--adapter", help="LoRA adapter output directory (Transformers only)")
    parser.add_argument("--provider", choices=["both", "openalex", "arxiv"], default="both")
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--per-search", type=int, default=8)
    parser.add_argument("--max-papers", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if args.backend == "gguf":
        if not args.gguf_path:
            parser.error("--backend gguf requires --gguf-path")
        if args.adapter:
            parser.error("PEFT --adapter requires --backend transformers")
        model = GGUFModel(args.gguf_path)
    else:
        if args.gguf_path:
            parser.error("--gguf-path requires --backend gguf")
        model = TransformersModel(args.model, adapter=args.adapter)

    agent = LiteratureAgent(model, max_searches=args.max_searches,
                            per_search=args.per_search,
                            max_papers=args.max_papers, provider=args.provider)
    review = agent.run(args.question)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    markdown = review.to_markdown()
    (out / "review.md").write_text(markdown, encoding="utf-8")
    (out / "review.json").write_text(json.dumps(review.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(markdown)
    if review.errors:
        print("Provider/model errors:", *review.errors, sep="\n- ")
    print(f"Saved: {out / 'review.md'}, {out / 'review.json'}")


if __name__ == "__main__":
    main()
