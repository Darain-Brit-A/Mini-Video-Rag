# Mini Video RAG

Mini Video RAG is a beginner-friendly Retrieval-Augmented Generation (RAG) application for **one local 5-10 minute video**. It turns speech into timestamped evidence, embeds that evidence, retrieves the three most semantically similar chunks for a question, and produces an answer using only those retrieved chunks.

The project intentionally keeps generation simple. The main academic focus is retrieval, embeddings, semantic similarity, timestamps, and grounding.

## Problem Statement

Finding a precise answer inside a video normally requires watching the entire recording. Mini Video RAG makes the video searchable by converting its speech into timestamped transcript chunks and using semantic search to find relevant passages.

## Objective

Demonstrate a complete local RAG pipeline:

```text
Video -> audio -> Whisper transcript -> timestamped chunks
      -> sentence embeddings -> FAISS index
Question -> question embedding -> top-k similarity retrieval
         -> retrieved evidence -> grounded answer
```

## Project Structure

```text
project/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── __init__.py
│   ├── transcription.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── generation.py
│   └── utils.py
├── data/
│   ├── videos/
│   ├── transcripts/
│   └── chunks/
└── vectorstore/
```

Generated transcript, chunk, index, and video files are ignored by git because this demo is designed for one local video at a time.

## Technologies Used

- **Streamlit**: simple interactive Python UI.
- **FFmpeg**: extracts mono 16 kHz WAV audio from the uploaded video. The app prefers system FFmpeg and falls back to the bundled `imageio-ffmpeg` binary.
- **faster-whisper**: local Whisper speech-to-text with segment timestamps.
- **sentence-transformers/all-MiniLM-L6-v2**: lightweight local text embedding model.
- **FAISS**: fast local vector index. Normalized vectors make inner product equivalent to cosine similarity.
- **OpenAI API (optional)**: improves answer wording while receiving only retrieved evidence.

## Installation

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
cd "d:\Christ\5th Semister\amagi\project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
cd mini-video-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install FFmpeg

The app can use its bundled FFmpeg binary, but a system FFmpeg installation is recommended. Verify a system installation with:

```powershell
ffmpeg -version
```

On Windows, install a current FFmpeg build, add its `bin` directory to PATH, then open a new terminal. On macOS use `brew install ffmpeg`; on Ubuntu/Debian use `sudo apt update && sudo apt install ffmpeg`. If you skip this step, `imageio-ffmpeg` downloads and supplies a compatible binary after installing the Python requirements.

### 3. Optional API configuration

Copy `.env.example` to `.env` and add `OPENAI_API_KEY` if you want a concise LLM-written answer:

```powershell
Copy-Item .env.example .env
```

The API is optional. Without a key, the app uses an extractive fallback that quotes the best retrieved transcript chunk. The fallback never uses general knowledge.

## Run the Application

From the activated virtual environment:

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## How to Use It

1. Upload one MP4, MOV, MKV, AVI, or WEBM video, ideally 5-10 minutes long.
2. Click **Transcribe and build retrieval index**.
3. Wait while audio is extracted, Whisper transcribes the speech, chunks are created, and embeddings are indexed.
4. Review the readable timestamped transcript.
5. Ask a question and choose the top-k value. The default is 3.
6. Inspect every retrieved chunk, timestamp, and similarity score before reading the grounded answer.

The first run downloads local Whisper and embedding model files, so it can take several minutes and use additional disk space.

## How Each RAG Component Works

### Transcription

`src/transcription.py` extracts audio with FFmpeg and passes it to faster-whisper. Each non-empty speech segment keeps numeric `start` and `end` seconds plus display timestamps such as `00:02:14`.

### Timestamped Chunking

`src/chunking.py` combines nearby Whisper segments until a chunk is approximately 150 words. It never splits an individual Whisper segment and carries the last segment into the next chunk as a small overlap. Each chunk stores `chunk_id`, `start_time`, `end_time`, and `text`.

### Embeddings

`src/embeddings.py` converts each chunk into a numeric vector. Similar meanings produce vectors that are close together even when the exact words differ. The same model embeds the user question, which makes a semantic comparison possible.

### Retrieval

`src/retrieval.py` stores chunk vectors in a FAISS `IndexFlatIP` index. Since vectors are normalized, the inner product is cosine similarity. A question embedding is compared with every chunk and the highest-scoring `top_k` chunks are returned. The UI displays the score and timestamp so retrieval quality is visible.

### Grounded Generation

`src/generation.py` receives only the retrieved chunks. With `OPENAI_API_KEY`, it instructs the selected model to answer only from that evidence and to refuse when the evidence is insufficient. Without a key, it quotes the best-scoring chunk. A low-similarity query returns:

> I could not find enough information about this in the video.

This design prevents the answer step from silently searching the internet or relying on unrelated model knowledge.

## Example Questions

Questions should be answerable from the uploaded video, for example:

- What is the main topic of the video?
- What example did the speaker give about cloud computing?
- Which steps were described in the process?
- What problem does the speaker say this method solves?
- What conclusion was given at the end?

## Example Output

```text
Question: What did the speaker say about cloud computing?

Retrieved Evidence:
1. 00:02:14 - 00:02:48 | Similarity: 0.734
2. 00:04:03 - 00:04:35 | Similarity: 0.612
3. 00:06:12 - 00:06:39 | Similarity: 0.541

Grounded Answer:
According to the video (00:02:14 - 00:02:48), the relevant passage says: "..."

Sources: 00:02:14 - 00:02:48, 00:04:03 - 00:04:35, 00:06:12 - 00:06:39
```

## Retrieval Evaluation

For a simple academic evaluation, prepare 5-10 questions whose answers are known from different parts of the video. For each question, record:

| Question | Expected timestamp | Retrieved top-1 timestamp | Top-3 contains answer? | Best score |
|---|---|---|---|---|
| Question 1 | 00:01:20 | 00:01:18 | Yes/No | 0.00 |

A useful measure is **Recall@3**: the percentage of questions where at least one of the top three retrieved chunks contains the expected answer. Also inspect false positives, low scores, and whether a small wording change alters retrieval. This demonstrates semantic similarity rather than just keyword matching.

## Limitations

- The app intentionally supports one video and one local process at a time.
- CPU transcription can be slow, especially with the `small` Whisper model.
- Automatic speech recognition can make errors with accents, noise, or overlapping speakers.
- Similarity scores are ranking signals, not probabilities of correctness.
- The fallback answer is intentionally simple and extractive.
- There is no speaker diarization, video-frame understanding, authentication, or production database.

## Future Improvements

- Add a model-size selector and GPU support.
- Add speaker labels and better sentence-aware chunk boundaries.
- Add a small evaluation form that calculates Recall@k automatically.
- Add a transcript search view and clickable timestamps.
- Support multiple videos with separate indexes.
- Add citation-aware answer formatting and answer confidence checks.

## College Presentation Notes

- **Why embeddings?** They represent the meaning of text as vectors, allowing a question and a relevant passage to match even when they do not share exact keywords.
- **How does retrieval work?** The question is embedded, compared with all chunk embeddings, and the highest cosine-similarity chunks are selected.
- **Why store timestamps?** Timestamps let a user verify the answer against the exact part of the source video and make the result useful for navigation.
- **What does top-k mean?** `k` is the number of highest-scoring chunks returned for the question. The default top-k is 3.
- **How does grounding prevent unsupported answers?** The answer step receives only retrieved transcript evidence and has an explicit refusal when that evidence does not contain the answer.
- **Why is this RAG?** It retrieves relevant external context from the video first, then augments answer generation with that context. The retrieval and generation stages together are Retrieval-Augmented Generation.
