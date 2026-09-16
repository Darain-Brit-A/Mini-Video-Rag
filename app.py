"""Streamlit dashboard for the one-video Mini Video RAG demonstration."""

import html
import os

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

st.set_page_config(page_title="Mini Video RAG", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #17221f; --muted: #6c7771; --paper: #f4f6f1; --line: #dce3db; --lime: #c9f17c; --teal: #1c6b5e; --coral: #e8755f; }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'Manrope', sans-serif; }
    [data-testid='stHeader'] { background: transparent; }
    [data-testid='stSidebar'] { background: #17221f; border-right: 0; }
    [data-testid='stSidebar'] * { color: #eaf1e8; }
    .block-container { max-width: 1380px; padding: 2.4rem 3.2rem 4rem; }
    h1, h2, h3, p, label { font-family: 'Manrope', sans-serif; }
    h1 { letter-spacing: -1.8px; font-size: clamp(2.2rem, 4vw, 4.2rem); line-height: 1.02; margin: 0; }
    h2 { letter-spacing: -0.8px; margin-top: 1.8rem; }
    .eyebrow { color: var(--teal); font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.8rem; }
    .hero { display: flex; justify-content: space-between; gap: 2rem; align-items: end; padding-bottom: 2.3rem; border-bottom: 1px solid var(--line); }
    .hero-copy { max-width: 720px; }
    .hero-copy p { color: var(--muted); font-size: 1rem; max-width: 580px; margin: 1rem 0 0; }
    .hero-mark { width: 116px; height: 116px; border-radius: 50%; background: var(--lime); display: grid; place-items: center; color: var(--ink); font-size: 3.6rem; transform: rotate(-8deg); box-shadow: 10px 10px 0 #dbe8c8; }
    .metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1.5rem 0 2rem; }
    .metric { border: 1px solid var(--line); background: rgba(255,255,255,.55); padding: 1.05rem 1.15rem; min-height: 92px; }
    .metric-label { color: var(--muted); font: 500 .68rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .8px; }
    .metric-value { color: var(--ink); font-size: 1.6rem; font-weight: 800; margin-top: .55rem; }
    .section-kicker { color: var(--teal); font: 500 .7rem 'DM Mono', monospace; letter-spacing: 1px; text-transform: uppercase; margin-bottom: .45rem; }
    .panel { background: white; border: 1px solid var(--line); padding: 1.35rem; }
    .panel-title { font-size: 1.1rem; font-weight: 800; margin-bottom: .25rem; }
    .panel-note { color: var(--muted); font-size: .84rem; margin-bottom: 1rem; }
    [data-testid='stSlider'] label, [data-testid='stSlider'] p { color: var(--ink) !important; }
    [data-testid='stTextInput'] input { color: var(--ink) !important; background: #ffffff !important; }
    [data-testid='stButton'] button { white-space: normal !important; line-height: 1.2 !important; }
    .evidence { border: 1px solid var(--line); border-left: 4px solid var(--teal); background: white; padding: 1.1rem 1.25rem; margin: .75rem 0; }
    .evidence-top { display: flex; justify-content: space-between; gap: 1rem; font: 500 .72rem 'DM Mono', monospace; color: var(--teal); }
    .evidence-text { line-height: 1.65; margin-top: .75rem; }
    .score-track { height: 5px; background: #e6ece4; margin-top: .7rem; }
    .score-fill { height: 5px; background: var(--coral); }
    .answer { background: var(--ink); color: #f4f6f1; padding: 1.5rem; font-size: 1.05rem; line-height: 1.7; border-left: 6px solid var(--lime); }
    .source-line { color: var(--muted); font: .68rem 'DM Mono', monospace; margin-top: 1rem; }
    .sidebar-title { color: var(--lime); font-size: 1.35rem; font-weight: 800; letter-spacing: -.6px; }
    .sidebar-copy { color: #aab9ae; font-size: .82rem; line-height: 1.6; }
    .step { display: flex; align-items: center; gap: .65rem; margin: .8rem 0; color: #aab9ae; font-size: .8rem; }
    .step-dot { width: 22px; height: 22px; border: 1px solid #60766a; border-radius: 50%; display: grid; place-items: center; font: .65rem 'DM Mono', monospace; }
    .step-active { color: var(--lime); } .step-active .step-dot { background: var(--lime); color: var(--ink); border-color: var(--lime); }
    @media (max-width: 800px) { .block-container { padding: 1.5rem 1rem 3rem; } .hero-mark { display: none; } .metric-row { grid-template-columns: repeat(2, 1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in {"vector_store": None, "transcript": [], "chunks": [], "evidence": [], "answer": "", "last_question": "", "video_name": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = default

ready = st.session_state.vector_store is not None

with st.sidebar:
    st.markdown('<div class="sidebar-title">◈ Mini Video RAG</div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-copy">A focused workspace for asking precise questions inside one local video.</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-kicker" style="color:#c9f17c">Workflow</div>', unsafe_allow_html=True)
    steps = [("01", "Upload source", bool(st.session_state.video_name)), ("02", "Build index", ready), ("03", "Ask question", bool(st.session_state.last_question))]
    for number, label, complete in steps:
        active_class = "step-active" if complete else ""
        st.markdown(f'<div class="step {active_class}"><span class="step-dot">{number}</span>{label}</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="section-kicker" style="color:#c9f17c">System</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sidebar-copy">Embedding model<br><strong>all-MiniLM-L6-v2</strong><br><br>Retriever<br><strong>FAISS · cosine similarity</strong></p>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="hero-copy"><div class="eyebrow">Research console / single video</div><h1>Ask the video.<br><em>See the evidence.</em></h1><p>Turn a 5–10 minute recording into a searchable, timestamped knowledge base. Every answer stays close to the retrieved transcript.</p></div><div class="hero-mark">◈</div></div>',
    unsafe_allow_html=True,
)

file_label = st.session_state.video_name or "No video loaded"
video_status = "Indexed and ready" if ready else ("Video selected" if st.session_state.video_name else "Waiting for upload")

st.markdown(
    f'<div class="metric-row"><div class="metric"><div class="metric-label">Source</div><div class="metric-value" style="font-size:1.05rem;overflow-wrap:anywhere">{html.escape(file_label)}</div></div><div class="metric"><div class="metric-label">Status</div><div class="metric-value" style="font-size:1.05rem">{video_status}</div></div><div class="metric"><div class="metric-label">Chunks indexed</div><div class="metric-value">{len(st.session_state.chunks)}</div></div><div class="metric"><div class="metric-label">Top-k default</div><div class="metric-value">3</div></div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1.5], gap="large")
with left:
    st.markdown('<div class="section-kicker">01 / Source</div><div class="panel-title">Load a video</div><div class="panel-note">One local file · MP4, MOV, MKV, AVI, WEBM</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "mov", "mkv", "avi", "webm"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.video_name = uploaded_file.name
        st.markdown(f'<div class="panel-note">Selected <strong>{html.escape(uploaded_file.name)}</strong></div>', unsafe_allow_html=True)
        process_clicked = st.button("Build index  →", type="primary", use_container_width=True)
        if process_clicked:
            video_path = VIDEO_DIR / uploaded_file.name
            audio_path = VIDEO_DIR / f"{video_path.stem}.wav"
            transcript_path = TRANSCRIPT_DIR / f"{video_path.stem}.json"
            chunks_path = CHUNK_DIR / f"{video_path.stem}.json"
            index_path = VECTORSTORE_DIR / f"{video_path.stem}.faiss"
            try:
                with st.status("Building your video index...", expanded=True) as status:
                    video_path.write_bytes(uploaded_file.getbuffer())
                    st.write("Extracting audio with FFmpeg")
                    extract_audio(video_path, audio_path)
                    st.write("Transcribing speech with Whisper")
                    transcript = transcribe_audio(audio_path, os.getenv("WHISPER_MODEL", "small"))
                    if not transcript:
                        raise RuntimeError("Whisper returned no speech segments. Try a clearer video.")
                    save_transcript(transcript, transcript_path)
                    st.write("Creating overlapping timestamped chunks")
                    chunks = build_chunks(transcript)
                    save_chunks(chunks, chunks_path)
                    st.write("Embedding chunks and building FAISS index")
                    store = VectorStore(EmbeddingModel())
                    store.build(chunks)
                    store.save(index_path, VECTORSTORE_DIR / f"{video_path.stem}.chunks.json")
                    st.session_state.transcript = transcript
                    st.session_state.chunks = chunks
                    st.session_state.vector_store = store
                    st.session_state.evidence = []
                    st.session_state.answer = ""
                    status.update(label=f"Ready · {len(chunks)} chunks indexed", state="complete")
                st.rerun()
            except Exception as error:
                st.error(f"Processing failed: {error}")
    else:
        st.markdown('<div class="panel-note" style="margin-top:1rem">Start by selecting a short video. The first run downloads the local AI models.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-kicker">02 / Query</div><div class="panel-title">Ask a grounded question</div><div class="panel-note">The answer model sees only the evidence retrieved below.</div>', unsafe_allow_html=True)
    with st.form("question_form"):
        question = st.text_input("Question", value=st.session_state.last_question, placeholder="What did the speaker explain about...?", disabled=not ready, label_visibility="collapsed")
        query_col, action_col = st.columns([1, 1])
        with query_col:
            top_k = st.slider("Evidence chunks (top-k)", min_value=1, max_value=5, value=3, disabled=not ready)
        with action_col:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            ask_clicked = st.form_submit_button("Retrieve  →", type="primary", use_container_width=True, disabled=not ready)
    if ask_clicked and question.strip():
        st.session_state.evidence = st.session_state.vector_store.search(question.strip(), top_k)
        st.session_state.answer = generate_answer(question.strip(), st.session_state.evidence)
        st.session_state.last_question = question.strip()
        st.rerun()

if st.session_state.evidence:
    st.markdown('<div class="section-kicker" style="margin-top:2.5rem">03 / Retrieved evidence</div><div class="panel-title">What the retriever found</div><div class="panel-note">Ranked by cosine similarity · higher scores indicate closer semantic meaning</div>', unsafe_allow_html=True)
    evidence_left, evidence_right = st.columns([1.5, 1], gap="large")
    with evidence_left:
        for rank, item in enumerate(st.session_state.evidence, start=1):
            score = max(0, min(1, float(item["similarity"])))
            st.markdown(
                f'<div class="evidence"><div class="evidence-top"><span>#{rank} · {item["start_time"]} — {item["end_time"]}</span><span>{item["similarity"]:.3f}</span></div><div class="score-track"><div class="score-fill" style="width:{score * 100:.1f}%"></div></div><div class="evidence-text">{html.escape(str(item["text"]))}</div></div>',
                unsafe_allow_html=True,
            )
    with evidence_right:
        st.markdown('<div class="section-kicker">04 / Grounded answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer">{html.escape(st.session_state.answer)}</div>', unsafe_allow_html=True)
        sources = ", ".join(f'{item["start_time"]} — {item["end_time"]}' for item in st.session_state.evidence)
        st.markdown(f'<div class="source-line">SOURCES · {sources}</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note" style="margin-top:1.6rem">This response is constrained to the retrieved transcript chunks. No outside knowledge is added.</div>', unsafe_allow_html=True)

if st.session_state.transcript and sum(len(str(segment["text"]).split()) for segment in st.session_state.transcript) < 5:
    st.warning("Only a few spoken words were detected in this video. Try a video with clearer speech before asking content questions.")

if st.session_state.transcript:
    st.markdown('<div class="section-kicker" style="margin-top:2.5rem">05 / Source transcript</div>', unsafe_allow_html=True)
    with st.expander(f"Browse {len(st.session_state.transcript)} timestamped speech segments"):
        for segment in st.session_state.transcript:
            st.markdown(f'**{segment["start_time"]} — {segment["end_time"]}** &nbsp; {segment["text"]}')
