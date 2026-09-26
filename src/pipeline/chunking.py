import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class TextChunker:
    """Collects several document-chunking strategies behind one interface.

    Models (embedder / tokenizer) are loaded lazily and cached, so creating
    a TextChunker is cheap even if you only ever call one of the methods.
    """

    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.embed_model_name = embed_model_name
        self.tokenizer_name = tokenizer_name
        self._embedder = None      # lazy-loaded SentenceTransformer
        self._tokenizer = None     # lazy-loaded AutoTokenizer

    # ---------- lazy model accessors ----------

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    # ---------- shared helpers ----------

    @staticmethod
    def _validate_window(size: int, overlap: int) -> int:
        """Shared guard for the sliding-window methods (word/char). Raises
        instead of silently producing an infinite loop (overlap == size) or
        nonsense (overlap > size)."""
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({size}), "
                "otherwise the window never advances"
            )
        return size - overlap  # step

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        # split after ./!/? followed by whitespace, keeping punctuation attached
        return re.split(r"(?<=[.!?])\s+", text)

    # ---------- 1. word-based sliding window ----------

    def word_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if not words:
            return []
        step = self._validate_window(chunk_size, overlap)

        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    # ---------- 2. character-based sliding window ----------

    def char_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks by raw character count.
        Fastest option; can cut words mid-way."""
        if not text.strip():
            return []
        step = self._validate_window(chunk_size, overlap)

        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(text):
                break
        return chunks

    # ---------- 3. sentence-based sliding window ----------

    def sentence_chunks(
        self, text: str, max_sentences: int = 5, overlap_sentences: int = 1
    ) -> list[str]:
        """Group whole sentences into chunks, never splitting mid-sentence."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        step = self._validate_window(max_sentences, overlap_sentences)

        chunks = []
        for i in range(0, len(sentences), step):
            chunk = " ".join(sentences[i : i + max_sentences])
            if chunk:
                chunks.append(chunk)
            if i + max_sentences >= len(sentences):
                break
        return chunks

    # ---------- 4. token-based sliding window ----------

    def token_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split by actual model tokens, matching the context-window budget
        an embedding/generation model will really consume."""
        if not text.strip():
            return []
        step = self._validate_window(chunk_size, overlap)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []

        chunks = []
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i + chunk_size >= len(tokens):
                break
        return chunks

    # ---------- 5. recursive / hierarchical ----------

    def recursive_chunks(
        self,
        text: str,
        chunk_size: int = 500,
        separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
    ) -> list[str]:
        """Split on the coarsest separator that keeps pieces under
        chunk_size, falling back to finer separators only where needed."""
        if len(text) <= chunk_size or not separators:
            return [text] if text.strip() else []

        sep, rest = separators[0], separators[1:]
        parts = text.split(sep)
        chunks, buf = [], ""
        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                if len(part) > chunk_size:
                    chunks.extend(self.recursive_chunks(part, chunk_size, rest))
                    buf = ""
                else:
                    buf = part
        if buf:
            chunks.append(buf)
        return chunks

    # ---------- 6. semantic (embedding-similarity based) ----------

    def semantic_chunks(
        self, text: str, threshold: float = 0.5, max_chunk_chars: int = 2000
    ) -> list[str]:
        """Group consecutive sentences until embedding similarity drops
        below `threshold` (a likely topic shift), with a hard size cap so a
        long on-topic run can't grow unbounded."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        embeddings = self.embedder.encode(sentences, normalize_embeddings=True)

        chunks, current = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(embeddings[i] @ embeddings[i - 1])
            current_len = sum(len(s) for s in current)
            topic_shift = sim < threshold
            too_long = current_len >= max_chunk_chars
            if topic_shift or too_long:
                chunks.append(" ".join(current))
                current = []
            current.append(sentences[i])
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ---------- dispatcher ----------

    def chunk(self, text: str, method: str = "recursive", **kwargs) -> list[str]:
        """Single entry point: chunk(text, method="word"|"char"|"sentence"|
        "token"|"recursive"|"semantic", **kwargs_for_that_method)."""
        dispatch = {
            "word": self.word_chunks,
            "char": self.char_chunks,
            "sentence": self.sentence_chunks,
            "token": self.token_chunks,
            "recursive": self.recursive_chunks,
            "semantic": self.semantic_chunks,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(dispatch)}")
        return dispatch[method](text, **kwargs)
