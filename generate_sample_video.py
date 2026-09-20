import os
import subprocess
import imageio_ffmpeg

wav_path = os.path.abspath("data/videos/sample_lecture.wav")
mp4_path = os.path.abspath("data/videos/sample_lecture.mp4")

# PowerShell command to synthesize clean speech
ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.SetOutputToWaveFile("{wav_path.replace(chr(92), '/')}")
$text = "Welcome to this lecture on Artificial Intelligence and Video Retrieval Augmented Generation. " +
        "First, let us examine how semantic embeddings work. Embeddings convert words and sentences into high dimensional vectors where semantically similar concepts are located close together in vector space. " +
        "Second, we use FAISS, which stands for Facebook AI Similarity Search, to index these dense vectors and perform ultra fast cosine similarity search. " +
        "Third, speech recognition models like Whisper extract timestamped transcripts directly from the video audio track. " +
        "Finally, when a user asks a question, the system retrieves only the top three most relevant timestamped chunks and passes them to the language model. " +
        "In conclusion, Retrieval Augmented Generation prevents hallucination by grounding all responses directly in verified transcript evidence. Thank you."
$synth.Speak($text)
$synth.Dispose()
'''

print("Synthesizing speech audio...")
res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
print("PS output:", res.stdout, res.stderr)

if os.path.exists(wav_path):
    print(f"Generated WAV file ({os.path.getsize(wav_path)} bytes). Creating MP4 video...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Generate MP4 video with a test card visual and the synthesized audio
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=#1e1e2e:s=1280x720:r=25",
        "-i", wav_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        mp4_path
    ]
    res_ffmpeg = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(mp4_path):
        print(f"Successfully generated test video at: {mp4_path} ({os.path.getsize(mp4_path)} bytes)")
    else:
        print("FFmpeg error:", res_ffmpeg.stderr)
