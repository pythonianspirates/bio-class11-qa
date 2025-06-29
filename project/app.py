from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pdf_loader import extract_text_from_pdf, chunk_text
import os
import uuid

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Qdrant client (in-memory)
qdrant = QdrantClient(":memory:")

# ✅ Create collection only if not exists (avoids warning)
if not qdrant.collection_exists("bio_class11"):
    qdrant.create_collection(
        collection_name="bio_class11",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# 📥 Load all PDFs from the 'pdfs' folder and upload their chunks
pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        full_path = os.path.join(pdf_dir, filename)
        chapter_text = extract_text_from_pdf(full_path)
        chunks = chunk_text(chapter_text)
        vectors = model.encode(chunks).tolist()

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vectors[i], payload={"text": chunks[i], "chapter": filename})
            for i in range(len(chunks))
        ]

        qdrant.upload_points(collection_name="bio_class11", points=points)

# 🌐 Main page route
@app.route("/")
def index():
    return render_template("index.html")

# 🔍 Search API
@app.route("/search", methods=["POST"])
def search():
    query = request.json["query"]
    vector = model.encode([query])[0].tolist()

    results = qdrant.search(
        collection_name="bio_class11",
        query_vector=vector,
        limit=5,
        with_payload=True,
        with_vectors=False
    )

    # Filter results by minimum similarity score
    MIN_SCORE = 0.5  # Tune this value (0.3–0.6 recommended)
    filtered = [
        {
            "chapter": r.payload['chapter'],
            "text": r.payload['text']
        }
        for r in results if r.score >= MIN_SCORE
    ]

    return jsonify({
    "query": query,
    "results": filtered if filtered else [{"chapter": "Not Found", "text": "No relevant content found."}]
})

# 🚀 Run app
if __name__ == "__main__":
    app.run(debug=True)