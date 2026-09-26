from flask import Flask, jsonify, request
from src.pipeline.chunking import TextChunker
from src.utils.constants import CHUNKING_VALID_METHODS, RE_RANKING_VALID_METHODS

app = Flask(__name__)

# One shared instance so models loaded by "token"/"semantic" methods are
# cached across requests instead of reloading every call.
chunker = TextChunker()


@app.route("/chunk", methods=["POST"])
def chunk_endpoint():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="Request body must be valid JSON"), 400

    text = data.get("text")
    method = data.get("method", "recursive")
    params = data.get("params", {})

    if not text or not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' is required and must be a non-empty string"), 400

    if method not in CHUNKING_VALID_METHODS:
        return jsonify(error=f"'method' must be one of {CHUNKING_VALID_METHODS}"), 400

    if not isinstance(params, dict):
        return jsonify(error="'params' must be a JSON object of keyword args"), 400

    try:
        chunks = chunker.chunk(text, method=method, **params)
    except TypeError as e:
        # e.g. an unexpected/misspelled param name for the chosen method
        return jsonify(error=f"Invalid params for method '{method}': {e}"), 400
    except ValueError as e:
        # e.g. overlap >= chunk_size
        return jsonify(error=str(e)), 400
    except Exception as e:
        app.logger.exception("Chunking failed")
        return jsonify(error=f"Internal error: {e}"), 500

    return jsonify(
        method=method,
        params=params,
        chunk_count=len(chunks),
        chunks=chunks,
    )

@app.route("/rerank", methods=["POST"])
def rerank_endpoint():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="Request body must be valid json"), 400

    method = data.get("method", "cross_encoder")
    params = data.get("params", {})

    if method not in RE_RANKING_VALID_METHODS:
        return jsonify(error="Invalid Re-ranking methods"), 400

    if not isinstance(params, dict):
        return jsonify(error="'params' must be a JSON object of keyword args"), 400

    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", available_methods=list(CHUNKING_VALID_METHODS))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)