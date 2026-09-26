import re
from collections import defaultdict
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
from src.utils.constants import EMBEDDING_MODEL, CROSS_ENCODER_NAME


class ReRanker:
    """Collects several reranking strategies behind one interface.
    Models are loaded lazily and cached, so creating a ReRanker is cheap
    even if you only ever call one of the methods."""

    def __init__(
        self,
        cross_encoder_name: str = CROSS_ENCODER_NAME,   # model used by cross_encoder_rerank
        embed_model_name: str = EMBEDDING_MODEL,             # model used by mmr_rerank
    ):
        self.cross_encoder_name = cross_encoder_name   # store name, don't load yet
        self.embed_model_name = embed_model_name        # store name, don't load yet
        self._cross_encoder = None                        # lazy-loaded CrossEncoder
        self._embedder = None                               # lazy-loaded SentenceTransformer

    # ---------- lazy model accessors ----------

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:                    # only load once, on first use
            self._cross_encoder = CrossEncoder(self.cross_encoder_name)
        return self._cross_encoder                          # cached on every later call

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:                          # only load once, on first use
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder                                # cached on every later call

    # ---------- shared helper ----------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())   # crude whitespace/punctuation tokenizer, lowercased

    # ---------- 1. Cross-encoder reranking ----------

    def cross_encoder_rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Score each (query, passage) pair jointly with a cross-encoder —
        slower than embedding similarity but much more accurate, since the
        model attends over query and passage together instead of comparing
        two separately-computed vectors."""
        if not candidates:                                    # nothing to rerank
            return []

        pairs = [(query, c["text"]) for c in candidates]        # build (query, passage) pairs, one per candidate
        scores = self.cross_encoder.predict(pairs)                # batched forward pass -> one relevance score per pair

        for c, s in zip(candidates, scores):                     # attach the new score onto each candidate dict
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)  # highest relevance first
        return ranked[:top_k]                                    # keep only the best top_k

    # ---------- 2. Maximal Marginal Relevance (diversity-aware) ----------

    def mmr_rerank(
        self, query: str, candidates: list[dict], top_k: int = 5, lambda_param: float = 0.5
    ) -> list[dict]:
        """Greedily pick candidates that are relevant to the query but not
        redundant with what's already been picked — trades a bit of pure
        relevance for coverage/diversity, useful when top results tend to
        be near-duplicates of each other."""
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]                       # pull out raw text for embedding
        doc_vecs = self.embedder.encode(texts, normalize_embeddings=True)  # embed all candidates, L2-normalized
        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]  # embed the query the same way

        relevance = doc_vecs @ query_vec                               # cosine similarity of each doc to the query

        selected_idx: list[int] = []                                    # indices already chosen, in order
        remaining_idx = list(range(len(candidates)))                     # indices not yet chosen

        while remaining_idx and len(selected_idx) < top_k:
            if not selected_idx:
                # first pick: no redundancy penalty possible yet, just take the most relevant
                best = max(remaining_idx, key=lambda i: relevance[i])
            else:
                def mmr_score(i):
                    # similarity of candidate i to every already-selected candidate
                    redundancy = max(doc_vecs[i] @ doc_vecs[j] for j in selected_idx)
                    # MMR formula: balance relevance to query vs. similarity to what's already picked
                    return lambda_param * relevance[i] - (1 - lambda_param) * redundancy

                best = max(remaining_idx, key=mmr_score)

            selected_idx.append(best)                                     # commit this candidate
            remaining_idx.remove(best)                                     # remove it from the pool

        results = [candidates[i] for i in selected_idx]                   # rebuild the candidate dicts in MMR order
        for rank, i in enumerate(selected_idx):
            results[rank]["rerank_score"] = float(relevance[i])             # keep the plain relevance score for reference
        return results

    # ---------- 3. BM25 lexical reranking ----------

    def bm25_rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank by keyword overlap (BM25) rather than semantics — cheap,
        no model download required, and strong for queries with exact terms
        (IDs, product names, acronyms) that embedding models can blur."""
        if not candidates:
            return []

        corpus_tokens = [self._tokenize(c["text"]) for c in candidates]   # tokenize every candidate passage
        bm25 = BM25Okapi(corpus_tokens)                                     # build BM25 index over this candidate set only
        query_tokens = self._tokenize(query)                                 # tokenize the query the same way
        scores = bm25.get_scores(query_tokens)                                # one BM25 score per candidate

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)                                       # attach the score

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)  # highest score first
        return ranked[:top_k]

    # ---------- 4. Reciprocal Rank Fusion (combine multiple rankers) ----------

    def reciprocal_rank_fusion(
        self, ranked_lists: list[list[dict]], top_k: int = 5, k_constant: int = 60
    ) -> list[dict]:
        """Merge several independently-ranked lists (e.g. vector search +
        BM25 + cross-encoder) into one ranking using only rank position, not
        raw scores — sidesteps the problem that different rankers' scores
        aren't on comparable scales."""
        rrf_scores: dict[str, float] = defaultdict(float)                    # accumulate fused score per unique text
        text_to_candidate: dict[str, dict] = {}                                # remember one representative dict per text

        for ranked_list in ranked_lists:                                       # go through each input ranking
            for rank, candidate in enumerate(ranked_list):                       # rank is 0-based position in that list
                key = candidate["text"]                                            # use text as the identity key
                rrf_scores[key] += 1.0 / (k_constant + rank + 1)                     # RRF formula: 1 / (k + rank)
                text_to_candidate.setdefault(key, candidate)                         # keep first-seen dict for this text

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)   # sort texts by fused score, descending
        results = []
        for text, score in fused[:top_k]:                                        # take the top_k fused results
            c = dict(text_to_candidate[text])                                      # copy so we don't mutate the original dict
            c["rerank_score"] = score                                               # attach the fused RRF score
            results.append(c)
        return results

    # ---------- 5. LLM-based pointwise reranking ----------

    def llm_rerank(
        self, query: str, candidates: list[dict], llm_generate, top_k: int = 5
    ) -> list[dict]:
        """Ask an LLM to directly score each passage's relevance to the
        query (1-10). Most expensive method (one generation call per
        candidate) but can capture nuanced relevance embedding/BM25 miss —
        typically used as a final precision pass over a small candidate set.

        `llm_generate` is any callable: llm_generate(prompt: str) -> str
        """
        if not candidates:
            return []

        for c in candidates:
            prompt = (                                                              # ask for a single numeric score
                f"On a scale of 1-10, how relevant is this passage to the query?\n"
                f"Only respond with a number.\n\n"
                f"Query: {query}\n"
                f"Passage: {c['text']}\n"
                f"Score:"
            )
            raw = llm_generate(prompt)                                              # call the provided LLM function
            match = re.search(r"\d+(\.\d+)?", raw)                                    # pull the first number out of the reply
            c["rerank_score"] = float(match.group()) if match else 0.0                 # default to 0 if parsing fails

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)     # highest LLM score first
        return ranked[:top_k]

    # ---------- dispatcher ----------

    def rerank(self, query: str, candidates: list[dict], method: str = "cross_encoder", **kwargs) -> list[dict]:
        """Single entry point: rerank(query, candidates, method="cross_encoder"|
        "mmr"|"bm25"|"llm", **kwargs_for_that_method). Note: "rrf" is not
        included here since it takes multiple ranked lists, not one
        candidate list + query — call reciprocal_rank_fusion directly."""
        dispatch = {
            "cross_encoder": self.cross_encoder_rerank,
            "mmr": self.mmr_rerank,
            "bm25": self.bm25_rerank,
            "llm": self.llm_rerank,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)} (or call reciprocal_rank_fusion directly)")
        return dispatch[method](query, candidates, **kwargs)

