# Interview Talking Points

## Why Qdrant?

It provides cosine vector search, metadata payloads, stable point IDs, persistence,
and a straightforward Python client. That fits a retrieval-focused system without
adding a relational database before application state requires one.

## Dense vs. lexical retrieval?

Dense retrieval matches semantic similarity and paraphrases; lexical retrieval rewards
exact term overlap. BM25 can recover identifiers or fact-heavy wording that a compact
embedding model misses, while dense search handles vocabulary mismatch better.

## Why hybrid search?

The two retrievers fail differently. Combining their rankings improved Hit Rate@5
from 0.85 to 1.00 on the unchanged benchmark.

## What is RRF?

Reciprocal Rank Fusion gives each result a score based on its position in each ranked
list, commonly `1 / (k + rank)`, then sums contributions. It avoids trying to normalize
incompatible BM25 and cosine scores and rewards agreement between retrievers.

## What does the reranker add?

The cross-encoder scores each query/chunk pair jointly, giving a more precise ordering
than independently produced embeddings. It moved difficult relevant chunks upward and
raised benchmark MRR from 0.8475 for hybrid to 0.9750.

## Why not rerank everything?

Cross-encoder inference is more expensive than vector lookup. Reranking only a bounded
candidate pool controls latency; the recorded average reranking stage was about
102.8 ms and still depends on hardware and model warm-up.

## What do Hit Rate@K and MRR measure?

Hit Rate@K is the share of questions with at least one labeled relevant chunk in the
top K. MRR averages the reciprocal rank of the first relevant chunk, so it is sensitive
to whether useful evidence appears near the top.

## What is the main latency bottleneck?

For retrieval experiments, local cross-encoder inference is the added bottleneck. In
the complete answer flow, local LLM generation is typically much slower than retrieval,
which is why token streaming matters for perceived responsiveness.

## Why is Ollama outside Compose?

Models are large and hardware/runtime choices vary by host. Keeping Ollama host-managed
avoids baking weights into an image, preserves the local model cache, and lets the host
control acceleration and model lifecycle.

## Why is production retrieval still dense?

Dense is the stable, simpler endpoint baseline. Hybrid and reranking were added behind
the evaluation boundary first, making quality and latency changes measurable without
silently changing answer behavior or adding production index-refresh concerns.

## What would you add next?

I would expand the labeled corpus, add end-to-end answer faithfulness evaluation, and
test hybrid index refresh and latency under realistic load before promoting it to the
production path. Authentication and persistent application state would follow only
when multi-user requirements justify them.
