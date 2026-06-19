"""Train a same-data MarioGPT baseline without released HF model weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
MARIOGPT_ROOT = ROOT / "external" / "mario-gpt"
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf-cache" / "transformers"))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MARIOGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(MARIOGPT_ROOT))

from mario_gpt import MarioDataset, MarioLM
from tileflow.benchmarks.mariogpt_prompt import CountFeaturePrompter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-text", type=Path, default=Path("results/baselines/same_data/data/mariogpt_train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/same_data/mariogpt"))
    parser.add_argument("--base-tokenizer", default="shyamsn97/Mario-GPT2-700-context-length")
    parser.add_argument("--context-len", type=int, default=700)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--epoch-steps",
        type=int,
        default=None,
        help="Override steps per epoch for controlled budgets based on prepared train windows.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    level_string = args.train_text.read_text(encoding="utf-8")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, local_files_only=True)
    dataset = MarioDataset(
        tokenizer=base_tokenizer,
        level_string=level_string,
        context_len=args.context_len,
        sample_all_indices=False,
    )
    tokenizer = dataset.tokenizer
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    dataset_steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    steps_per_epoch = args.epoch_steps or dataset_steps_per_epoch
    if args.epochs is not None:
        args.steps = steps_per_epoch * args.epochs

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.context_len,
        n_ctx=args.context_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        add_cross_attention=True,
    )
    lm = GPT2LMHeadModel(config)
    prompter = CountFeaturePrompter(tokenizer, hidden_dim=config.n_embd)
    mario_lm = MarioLM(lm=lm, tokenizer=tokenizer, context_len=args.context_len, prompter=prompter)
    mario_lm.to(torch.device(args.device))
    mario_lm.train()

    optimizer = torch.optim.AdamW(mario_lm.lm.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    log_path = args.output_dir / "training_log.jsonl"
    for step in range(args.steps):
        indices = torch.randint(low=0, high=len(dataset), size=(args.batch_size,)).tolist()
        batch = dataset[indices]
        input_ids = batch[0].view(args.batch_size, -1).to(mario_lm.device)
        attention_masks = batch[1].to(mario_lm.device)
        encoder_hidden_states = []
        for level in input_ids:
            encoder_hidden_states.append(mario_lm.prompter(level)[1])
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0).view(args.batch_size, 1, -1)

        optimizer.zero_grad(set_to_none=True)
        outputs = mario_lm.lm(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_masks,
            encoder_hidden_states=encoder_hidden_states,
            token_type_ids=None,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mario_lm.lm.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % 10 == 0:
            entry = {
                "step": step + 1,
                "loss": float(loss.item()),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(entry)

        if (step + 1) % args.save_every == 0:
            checkpoint_dir = args.output_dir / f"iteration_{step + 1}"
            mario_lm.lm.save_pretrained(checkpoint_dir)

    final_dir = args.output_dir / "iteration_final"
    mario_lm.lm.save_pretrained(final_dir)
    manifest = {
        "scope": "same-data MarioGPT baseline",
        "released_hf_model_weights": False,
        "base_tokenizer": args.base_tokenizer,
        "model_path": str(final_dir),
        "tokenizer_path": str(args.output_dir / "tokenizer"),
        "steps": args.steps,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "dataset_steps_per_epoch": dataset_steps_per_epoch,
        "epoch_steps_override": args.epoch_steps,
        "context_len": args.context_len,
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "train_text": str(args.train_text),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
