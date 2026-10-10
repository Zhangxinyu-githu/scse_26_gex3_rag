"""Ground Qwen3-0.6B answers in the retrieved university IT policies."""

import os
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_tool import search_policy, verified_service_desk_phone


UNKNOWN_PHONE = (
    "I don't know the exact University IT Service Desk phone number. "
    "It is not specified in the available policy documents."
)


def default_model_path():
    configured = os.environ.get("QWEN_MODEL_PATH")
    if configured:
        return configured
    sibling = Path(__file__).resolve().parent.parent / "SCSE_26_GEX3" / "models" / "qwen3-0.6b"
    return str(sibling) if sibling.is_dir() else "Qwen/Qwen3-0.6B"


class RAGAssistant:
    def __init__(self, model_path=None):
        self.model_path = model_path or default_model_path()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype="auto")
        self.model.to(self.device)
        self.model.eval()

    def answer(self, question):
        evidence = search_policy.invoke({"question": question})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a university IT support assistant. Answer using only the "
                    "policy evidence below. If the requested fact is not stated, say "
                    "that you do not know it. Never invent a phone number. Keep the "
                    "answer concise.\n\nPolicy evidence:\n" + evidence
                ),
            },
            {"role": "user", "content": question},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # A number is answerable only when the supplied policies explicitly give it.
        asks_for_service_desk_phone = re.search(r"service desk", question, re.I) and re.search(
            r"phone|telephone|number|call", question, re.I
        )
        if asks_for_service_desk_phone and verified_service_desk_phone() is None:
            return UNKNOWN_PHONE
        return response
