import asyncio
import json
import os
import logging
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError, RateLimitError  # Import specific error

load_dotenv()

logger = logging.getLogger("core")

# Configuration
BATCH_SIZE = 150
MODEL_NAME = "text-embedding-3-large"
MAX_CONCURRENT_REQUESTS = 3  # Control parallel API calls
MAX_RETRIES = 3  # Max retries for backoff


def create_batches(texts: list, batch_size: int = BATCH_SIZE) -> list:
    """Groups texts into batches without exceeding the max token limit."""
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batches.append(batch)
    return batches


async def embed_batch(client, batch, semaphore):
    """Embeds a single batch asynchronously while respecting API limits and handling rate limits."""
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Embedding batch of {len(batch)} texts (Attempt {attempt})"
                )
                response = await client.embeddings.create(input=batch, model=MODEL_NAME)
                logger.info(f"Received embeddings for {len(response.data)} texts")
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                if attempt < MAX_RETRIES:
                    backoff = 2 ** (attempt - 1)
                    logger.warning(
                        f"Rate limit hit. Retrying in {backoff} seconds... (Attempt {attempt})"
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Exceeded retries due to rate limiting: {e}")
                    return [None] * len(batch)
            except Exception as e:
                logger.error(f"Unexpected error while embedding batch: {e}")
                return [None] * len(batch)


async def _embed_texts_async(texts: list, api_key: str):
    """Embeds a list of texts asynchronously, ensuring max token limits per batch."""
    if not api_key:
        raise ValueError("API key is required")
    if not texts:
        raise ValueError("Texts are required")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with AsyncOpenAI(api_key=api_key) as client:
        batches = create_batches(texts)
        embeddings = [None] * len(texts)
        batch_indices = []
        start_idx = 0
        for batch in batches:
            batch_indices.append((start_idx, start_idx + len(batch)))
            start_idx += len(batch)

        tasks = [embed_batch(client, batch, semaphore) for batch in batches]
        results = await asyncio.gather(*tasks)

        for (start, end), batch_result in zip(batch_indices, results):
            embeddings[start:end] = batch_result

        return embeddings


def create_embeddings(texts: list, api_key: str):
    """Safe sync wrapper to run embedding inside a Celery or sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_embed_texts_async(texts, api_key))
    finally:
        loop.close()


if __name__ == "__main__":
    from src.utils.logger import logger
    api_key = os.getenv("OPENAI_API_KEY")
    with open(r"core/data/temp/Jain Yug_chunk.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    embeddings = create_embeddings(texts["chunks"], api_key=api_key)

    # Save embeddings to structured JSON with text-embedding pairs
    with open("embeddings.json", "w", encoding="utf-8") as f:
        result = [
            {"text": text, "embedding": emb}
            for text, emb in zip(texts["chunks"], embeddings)
        ]
        json.dump(result, f, ensure_ascii=False, indent=2)
