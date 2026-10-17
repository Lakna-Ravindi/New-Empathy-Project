import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency in local tests
    genai = None


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / "backend" / ".env")
CACHE_DIR = BASE_DIR / "backend" / ".cache"
CACHE_PATH = CACHE_DIR / "semantic_search_cache.json"

_DEFAULT_MODEL = "gemini-embedding-001"
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "she", "that", "the", "their", "them", "they", "this",
    "to", "us", "was", "we", "what", "when", "where", "which", "who", "why",
    "with", "you", "your"
}


def _cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _stable_content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_tokens(text: str) -> list[str]:
    if not text:
        return []
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in _STOP_WORDS]


def _content_text(node: dict) -> str:
    if not isinstance(node, dict):
        return ""
    parts = [node.get("title", ""), node.get("content", "")]
    return " ".join(part for part in parts if isinstance(part, str) and part.strip())


def _flatten_content(knowledge_base: Iterable[dict]) -> list[dict]:
    content_nodes = []
    for node in knowledge_base or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "content":
            continue
        text = _content_text(node)
        if not text.strip():
            continue
        item = dict(node)
        item["_search_text"] = text.strip()
        content_nodes.append(item)
    return content_nodes


def _load_cache() -> dict:
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return payload
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def _save_cache(cache: dict) -> None:
    _cache_dir()
    with CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def _local_vector(text: str) -> Counter:
    tokens = Counter(_normalize_tokens(text))
    if not tokens:
        return Counter()
    return tokens


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    if not vec_a or not vec_b:
        return 0.0
    vocab = set(vec_a) | set(vec_b)
    if not vocab:
        return 0.0
    dot = sum(vec_a.get(token, 0) * vec_b.get(token, 0) for token in vocab)
    norm_a = sum(value * value for value in vec_a.values()) ** 0.5
    norm_b = sum(value * value for value in vec_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding(model_name: str, text: str, cache: dict) -> list[float]:
    cache_key = f"{model_name}:{_stable_content_hash(text)}"
    if cache_key in cache:
        return cache[cache_key]

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and genai is not None:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.embed_content(
                model=model_name,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            values = response.embeddings[0].values if getattr(response, "embeddings", None) else None
            if values:
                cache[cache_key] = values
                return values
        except Exception:
            pass

    vector = [float(value) for value in _local_vector(text).values()]
    if not vector:
        return []
    cache[cache_key] = vector
    return vector


def _belongs_to_skill(node: dict, skill_id: str, nodes_by_id: dict) -> bool:
    if not node or not skill_id:
        return False
    current = node
    visited = set()
    while current and current.get("id") not in visited:
        visited.add(current.get("id"))
        if current.get("id") == skill_id:
            return True
        current = nodes_by_id.get(current.get("parent_id"))
    return False


def semantic_search(question: str, knowledge_base: list[dict], *, limit: int = 5, skill_id: str = None):
    """Return relevant approved content nodes using Gemini embeddings when available,
    with a conservative local lexical fallback when no API key or model is configured.
    """
    if not question or not knowledge_base:
        return []

    model_name = os.getenv("GEMINI_EMBEDDING_MODEL") or _DEFAULT_MODEL
    cache = _load_cache()
    query_text = (question or "").strip()
    if not query_text:
        return []

    query_vector = _get_embedding(model_name, query_text, cache)
    if not query_vector:
        query_vector = _local_vector(query_text)
    if not query_vector:
        return []

    nodes_by_id = {node.get("id"): node for node in knowledge_base if isinstance(node, dict) and node.get("id")}

    candidates = []
    for node in _flatten_content(knowledge_base):
        if skill_id and not (
            node.get("id") == skill_id
            or node.get("skill_id") == skill_id
            or _belongs_to_skill(node, skill_id, nodes_by_id)
        ):
            continue

        candidate_text = node.get("_search_text", "")
        candidate_vector = _get_embedding(model_name, candidate_text, cache)
        if not candidate_vector:
            candidate_vector = _local_vector(candidate_text)
        if not candidate_vector:
            continue
        similarity = _cosine_similarity(
            _local_vector(query_text) if not isinstance(query_vector, Counter) else query_vector,
            _local_vector(candidate_text) if not isinstance(candidate_vector, Counter) else candidate_vector,
        ) if not isinstance(query_vector, Counter) or not isinstance(candidate_vector, Counter) else _cosine_similarity(query_vector, candidate_vector)
        if similarity <= 0:
            continue
        candidates.append({
            "id": node.get("id"),
            "type": node.get("type"),
            "title": node.get("title"),
            "content": node.get("content"),
            "page": node.get("page"),
            "skill_id": node.get("skill_id"),
            "objective_id": node.get("objective_id"),
            "source": node.get("source"),
            "score": similarity,
        })

    if not candidates:
        _save_cache(cache)
        return []

    candidates.sort(key=lambda item: (-float(item["score"]), item.get("id", "")))
    selected = candidates[:limit]
    _save_cache(cache)
    return selected
