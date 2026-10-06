"""Retrieval-augmented generation and optional model-based answer judging."""

import json
import re

from .retrieval import SPLADERetriever, clean_query, get_doc_text, get_text_from_index


def strip_think(text):
    """Remove a completed model reasoning block from generated text."""
    return text.rsplit("</think>", 1)[-1].strip()


def generate_completion(
    model, tokenizer, messages, max_new_tokens=200, thinking=False, do_sample=True, device=None
):
    """Decode generated tokens only, keeping the prompt out of the answer."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    device = model.device if device is None else device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation["temperature"] = 0.7
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation)
    finally:
        model.train(was_training)
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True))


class _GenerationMixin:
    def generate(self, query, context, max_new_tokens=200):
        messages = [
            {
                "role": "system",
                "content": "Answer the question using the provided context. Say when the context does not contain the answer.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return generate_completion(
            self.generator,
            self.gen_tokenizer,
            messages,
            max_new_tokens,
            self.thinking,
            device=self.device,
        )

    def __call__(self, query):
        retrieved_docs = self.retrieve(query)
        texts = self._get_texts([doc_id for doc_id, _ in retrieved_docs])
        context = "\n\n".join(texts.get(doc_id, "") for doc_id, _ in retrieved_docs)
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_answer": self.generate(query, context),
        }


class RAGPipelineSplade(_GenerationMixin, SPLADERetriever):
    def __init__(
        self,
        retriever,
        generator,
        generator_tokenizer,
        pt_index,
        doc_ids,
        top_k=3,
        thinking=False,
        device=None,
    ):
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.thinking = thinking
        self.device = device
        super().__init__(retriever, pt_index, doc_ids, top_k)


class RAGPipelineBM25(_GenerationMixin):
    def __init__(
        self,
        retriever,
        generator,
        generator_tokenizer,
        pt_index,
        doc_ids=None,
        top_k=3,
        thinking=False,
        device=None,
    ):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.pt_index = pt_index
        self.doc_ids = doc_ids
        self.top_k = top_k
        self.thinking = thinking
        self.device = device

    def _get_doc_text(self, doc_id):
        return get_doc_text(self.pt_index, doc_id)

    def _get_texts(self, doc_ids):
        return get_text_from_index(self.pt_index, doc_ids)

    def retrieve(self, query, top_k=None):
        top_k = self.top_k if top_k is None else top_k
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if top_k == 0:
            return []
        results = self.retriever.search(clean_query(query)).head(top_k)
        return [(str(row.docno), float(row.score)) for row in results.itertuples(index=False)]


RAGPipeline = RAGPipelineSplade


def parse_judgment(text):
    """Validate judge scores; malformed output cannot silently become a rating."""
    text = strip_think(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        scores = [data.get(key) for key in ("relevance", "helpfulness")]
        if all(type(score) is int and 1 <= score <= 5 for score in scores) and isinstance(
            data.get("notes"), str
        ):
            return {key: data[key] for key in ("relevance", "helpfulness", "notes")}
    return {
        "relevance": None,
        "helpfulness": None,
        "notes": "Missing or invalid judge JSON",
        "raw": text,
    }


def judge_answer_quality(judge_model, judge_tokenizer, query, answer, max_new_tokens=256):
    messages = [
        {
            "role": "system",
            "content": "Evaluate answer quality. Output ONLY valid JSON with integer relevance and helpfulness scores from 1 to 5 and a short notes string.",
        },
        {"role": "user", "content": f"Question:\n{query}\n\nAnswer:\n{answer}"},
    ]
    completion = generate_completion(
        judge_model, judge_tokenizer, messages, max_new_tokens, thinking=False, do_sample=False
    )
    return parse_judgment(completion)
