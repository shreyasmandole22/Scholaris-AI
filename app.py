import os

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# 2. Model configuration
# --------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


# --------------------------------------------------
# 3. Load RAG data
# --------------------------------------------------

DATA_PATH = os.path.join(BASE_DIR, "embedding_gemini.joblib")

df = joblib.load(DATA_PATH)

# Prepare all stored embeddings once when Flask starts
ALL_EMBEDDINGS = np.asarray(
    df["embedding"].tolist(),
    dtype=np.float32
)


# --------------------------------------------------
# 4. Convert seconds into video timestamps
# --------------------------------------------------

def seconds_to_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


# --------------------------------------------------
# 5. Generate question embedding using Gemini
# --------------------------------------------------

def get_embedding(text):

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    return response.embeddings[0].values


# --------------------------------------------------
# 6. Generate answer using Gemini
# --------------------------------------------------

def ask_llm(prompt):

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "max_output_tokens": 220
        }
    )

    return response.text


# --------------------------------------------------
# 7. Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index1.html")


# --------------------------------------------------
# 8. Ask question
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please type a question.",
            "sources": []
        })

    try:

        # Step 1: Generate question embedding
        question_embedding = get_embedding(question)

        question_vector = np.asarray(
            question_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        # Step 2: Find similarity with stored embeddings
        similarities = cosine_similarity(
            ALL_EMBEDDINGS,
            question_vector
        ).flatten()

        # Step 3: Retrieve top 5 relevant chunks
        top_5_indexes = similarities.argsort()[::-1][:5]

        top_chunks = df.iloc[top_5_indexes]

        # Step 4: Prepare context and sources
        context_text = ""
        sources = []

        for _, chunk in top_chunks.iterrows():

            start = seconds_to_time(chunk["start"])
            end = seconds_to_time(chunk["end"])

            context_text += (
                f"[Video {chunk['number']} | "
                f"{start}-{end}] {chunk['text']}\n"
            )

            sources.append({
                "video": chunk["number"],
                "title": chunk["titile"],
                "time": f"{start}-{end}"
            })

        # Step 5: Build prompt
        prompt = f"""
You are a teaching assistant for a web development course.

Only use the course content below.
Do not make up information.

Structure your answer like this:

1. Start with one short friendly line introducing what you found.
2. Give 4 to 5 short lines, one point each.
3. Every factual line must end with its source
   in this exact format:
   (Video <number>, <start>-<end>)
4. End with one short closing line inviting a follow-up question.

If the question is unrelated to this content,
reply with one line saying you can only help
with course questions.

Course content:
{context_text}

Question: {question}

Answer:
"""

        # Step 6: Generate answer
        answer = ask_llm(prompt)

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception:

        app.logger.exception("Error processing question")

        return jsonify({
            "answer": "Something went wrong while processing your question. Please try again.",
            "sources": []
        }), 500


# --------------------------------------------------
# 9. Run Flask locally
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
