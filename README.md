# Customer Support Bot — LoRA Finetuned Qwen2.5-3B

A LoRA-finetuned Qwen2.5-3B customer-support assistant, trained on real support
conversations, with a live demo.

## Why this exists

Most production support chatbots are built as a wrapper around a large hosted
LLM (GPT-4o, Claude) plus RAG over a company's own docs — RAG handles the
*facts*, since a knowledge base can be updated instantly without retraining
anything. But RAG alone doesn't reliably enforce *tone, structure, or
behavior* (always opening with empathy, always ending with a clear next step,
refusing to promise refunds it isn't authorized to promise). That's what
finetuning is actually good at.

This project focuses specifically on that second half: teaching a small,
self-hostable open model to consistently answer in a support-agent voice and
format — the kind of thing a cost-sensitive or privacy-sensitive startup
would want instead of paying per-token to call a hosted LLM for every ticket.
In a full system, this finetuned model would sit behind a RAG layer that
supplies the company-specific facts.

## Before vs. after

| Question | Base Qwen2.5-3B-Instruct | After LoRA finetuning |
|---|---|---|
| "My order hasn't arrived yet, what should I do?" | *[paste your step 8 output here]* | *[paste your step 13 output here]* |

*(This table is the single most important part of this README — it's the
proof the finetuning did something. Fill it in with your actual saved
outputs before publishing.)*

## How it was built

- **Base model:** `unsloth/Qwen2.5-3B-Instruct`, loaded in 4-bit
- **Method:** LoRA via [Unsloth](https://github.com/unslothai/unsloth)
  (rank 16, alpha 16, targeting the attention + MLP projection layers)
- **Dataset:** [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  — 500-row random subset, reformatted into chat-template format.
  Licensed CDLA-Sharing-1.0.
- **Training:** 2 epochs, batch size 2 (gradient accumulation 4), learning
  rate 2e-4, on a free Google Colab T4 GPU. Training loss went from
  *[your start loss]* to *[your end loss]* — total training time ~*[X]* minutes.
- **Hardware used to build this:** Google Colab free tier (no local GPU required)

## Try it yourself

- Hugging Face Hub (model weights): *[your repo link, e.g. huggingface.co/your-username/qwen2.5-3b-support-lora]*
- Live demo: *[paste a fresh Gradio `share=True` link here before sending to anyone — these expire after ~72 hours]*

## What I'd do next

- Pair this with a RAG layer over real company docs/FAQ, so the model has
  up-to-date facts instead of only tone/format
- Evaluate on a larger held-out test set rather than spot-checking a few
  questions by hand
- Try a higher LoRA rank or more training examples and compare whether the
  stylistic shift gets stronger
- Stress-test on edge cases (angry customers, ambiguous requests, questions
  outside the support domain) to see where it should hand off to a human

## Project structure

```
support-bot-finetune/
├── README.md
├── notebook.ipynb       # the Colab notebook, exported via File → Download
├── requirements.txt
└── examples/             # before/after screenshots, demo screenshot
```
