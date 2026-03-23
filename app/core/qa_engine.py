"""
RAG-based Q&A Engine
Retrieves context chunks from Qdrant and generates answers via OpenAI GPT.
"""

import logging
from openai import OpenAI

from app.core.config import get_settings
from app.ingestion.pipeline import semantic_search
from app.models.schemas import QueryResponse, SourceReference

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a knowledgeable and empathetic assistant specializing in U.S. immigration \
and visa information for international students. You help students understand F-1 visa rules, \
OPT/CPT procedures, STEM OPT extensions, H-1B transitions, cap-gap provisions, and related topics.

Guidelines:
- Answer based ONLY on the provided context from official USCIS sources.
- If the context does not contain enough information, say so clearly and suggest the student \
  consult an immigration attorney or their DSO (Designated School Official).
- Be precise, clear, and actionable. Use bullet points when listing steps.
- Always remind the student that immigration rules change — they should verify with official sources.
- Never provide legal advice; provide informational guidance only.
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['title']}]\n{chunk['text']}\n(URL: {chunk['url']})"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, top_k: int = None) -> QueryResponse:
    """Full RAG pipeline: retrieve → augment → generate."""
    top_k = top_k or settings.top_k_results

    # 1. Semantic retrieval
    logger.info("Retrieving context for: %s", question[:80])
    chunks = semantic_search(question, top_k=top_k)

    if not chunks:
        return QueryResponse(
            question=question,
            answer=(
                "I couldn't find relevant information in my knowledge base. "
                "Please consult your DSO or visit uscis.gov directly."
            ),
            sources=[],
            model_used=settings.openai_model,
        )

    # 2. Build RAG prompt
    context = build_context(chunks)
    user_message = (
        f"Context from official USCIS sources:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Please provide a clear, accurate answer based on the context above."
    )

    # 3. Generate answer via OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # 4. Deduplicate sources
    seen_urls: set[str] = set()
    sources: list[SourceReference] = []
    for chunk in chunks:
        if chunk["url"] not in seen_urls:
            seen_urls.add(chunk["url"])
            sources.append(
                SourceReference(
                    title=chunk["title"],
                    url=chunk["url"],
                    category=chunk["category"],
                    score=chunk["score"],
                )
            )

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        model_used=settings.openai_model,
    )
