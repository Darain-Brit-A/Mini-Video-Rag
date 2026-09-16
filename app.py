"""Streamlit interface for the one-video Mini Video RAG demonstration."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chunking import build_chunks, save_chunks
from src.embeddings import EmbeddingModel
from src.generation import generate_answer
from src.retrieval import VectorStore
from src.transcription import extract_audio, save_transcript, transcribe_audio
from src.utils import CHUNK_DIR, TRANSCRIPT_DIR, VECTORSTORE_DIR, VIDEO_DIR, ensure_project_directories

load_dotenv()
ensure_project_directories()

st.set_page_config(page_title="Mini Video RAG", page_icon="🎥", layout="wide")
st.title("Mini Video RAG")
st.caption("A small, transparent retrieval-augmented question-answering system for one video")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.header("1. Upload Video")
uploaded_file = st.file_uploader("Choose one MP4 or common video file", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file:
    st.info(f"Selected: {uploaded_file.name}")
    st.header("2. Process Video")
    if st.button("Transcribe and build retrieval index", type="primary"):
        video_path = VIDEO_DIR / uploaded_file.name
        audio_path = VIDEO_DIR / f"{video_path.stem}.wav"
        transcript_path = TRANSCRIPT_DIR / f"{video_path.stem}.json"
        chunks_path = CHUNK_DIR / f"{video_path.stem}.json"
        index_path = VECTORSTORE_DIR / f"{video_path.stem}.faiss"
        saved_video = video_path
        try:
            with st.status("Processing video...", expanded=True) as status:
                saved_video.write_bytes(uploaded_file.getbuffer())
                st.write("Extracting audio with FFmpeg...")
                extract_audio(saved_video, audio_path)
                st.write("Transcribing with Whisper...")
                transcript = transcribe_audio(audio_path, os.getenv("WHISPER_MODEL", "small"))
                if not transcript:
                    raise RuntimeError("Whisper returned no speech segments. Try a clearer video.")
                save_transcript(transcript, transcript_path)
                st.write("Creating timestamped chunks...")
                chunks = build_chunks(transcript)
                save_chunks(chunks, chunks_path)
                st.write("Generating embeddings and building FAISS index...")
                store = VectorStore(EmbeddingModel())
                store.build(chunks)
                store.save(index_path, VECTORSTORE_DIR / f"{video_path.stem}.chunks.json")
                st.session_state.transcript = transcript
                st.session_state.chunks = chunks
                st.session_state.vector_store = store
                status.update(label=f"Ready: {len(chunks)} timestamped chunks indexed", state="complete")
            st.success("The video is ready for questions.")
        except Exception as error:
            st.error(f"Processing failed: {error}")

if st.session_state.transcript:
    with st.expander("View timestamped transcript"):
        for segment in st.session_state.transcript:
            st.markdown(f"**{segment['start_time']} - {segment['end_time']}**  {segment['text']}")

st.header("3. Ask a Question")
question = st.text_input("Ask something about the video", placeholder="What did the speaker explain about...?", disabled=st.session_state.vector_store is None)
top_k = st.slider("Top-k evidence chunks", min_value=1, max_value=5, value=3, disabled=st.session_state.vector_store is None)

if st.button("Retrieve evidence and answer", disabled=not question or st.session_state.vector_store is None):
    evidence = st.session_state.vector_store.search(question, top_k)
    st.header("4. Retrieved Evidence")
    for rank, item in enumerate(evidence, start=1):
        st.markdown(
            f"**{rank}. {item['start_time']} - {item['end_time']}**  "
            f"Similarity: `{item['similarity']:.3f}`\n\n{item['text']}"
        )
        st.divider()
    st.header("5. Grounded Answer")
    st.markdown(generate_answer(question, evidence))
    st.caption("Sources: " + ", ".join(f"{item['start_time']} - {item['end_time']}" for item in evidence))
