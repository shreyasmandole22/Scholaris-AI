
import os
import time
import joblib

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 20

# File paths
input_path = os.path.join(BASE_DIR, "embedding.joblib")
output_path = os.path.join(BASE_DIR, "embedding_gemini.joblib")
checkpoint_path = os.path.join(BASE_DIR, "embedding_checkpoint.joblib")

# Load original data
df = joblib.load(input_path)

texts = df["text"].fillna("").astype(str).tolist()

# Resume previously saved embeddings
if os.path.exists(checkpoint_path):
    all_embeddings = joblib.load(checkpoint_path)
    print("Resuming from saved checkpoint...")
else:
    all_embeddings = []

if len(all_embeddings) > len(texts):
    raise ValueError("Checkpoint contains more embeddings than source chunks.")

print("Total chunks:", len(texts))
print("Already completed:", len(all_embeddings))

# Generate remaining embeddings
for start in range(len(all_embeddings), len(texts), BATCH_SIZE):

    batch = texts[start:start + BATCH_SIZE]

    print(f"\nProcessing chunks {start + 1} to {start + len(batch)}")

    retries = 0

    while True:
        try:

            response = client.models.embed_content(
                model=MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DIMENSIONS
                )
            )

            batch_embeddings = [
                embedding.values
                for embedding in response.embeddings
            ]

            if len(batch_embeddings) != len(batch):
                raise ValueError("Embedding count mismatch.")

            all_embeddings.extend(batch_embeddings)

            # Save progress after every successful batch
            joblib.dump(all_embeddings, checkpoint_path)

            print("Completed:", len(all_embeddings), "/", len(texts))

            break

        except Exception as e:

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):

                retries += 1

                if retries > 3:
                    print("Rate limit persists. Progress has been saved.")
                    raise

                print("Gemini rate limit reached.")
                print("Waiting 65 seconds before retrying...")

                time.sleep(65)

            else:
                raise

    # Pause after every 4 batches to stay below the
    # apparent 100-input-per-minute free-tier limit.
    completed_batches = (start // BATCH_SIZE) + 1

    if completed_batches % 4 == 0 and start + len(batch) < len(texts):
        print("Taking a 65-second quota safety break...")
        time.sleep(65)

# Save final dataframe
df["embedding"] = all_embeddings

joblib.dump(df, output_path)

print("\nAll embeddings generated successfully!")
print("Total chunks:", len(df))
print("Embedding dimensions:", len(all_embeddings[0]))
print("Saved file:", output_path)

