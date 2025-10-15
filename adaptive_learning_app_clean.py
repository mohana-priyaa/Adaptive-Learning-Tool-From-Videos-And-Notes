# adaptive_learning_app_memory_safe.py
import os
import re
import gc
import streamlit as st
import whisper
from transformers import pipeline, MarianMTModel, MarianTokenizer
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# --- PDF creation ---
def create_pdf(summary, bullets, filename="notes.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Learning Notes")

    c.setFont("Helvetica", 12)
    y = height - 100
    c.drawString(50, y, "Summary:")
    y -= 20
    text_obj = c.beginText(50, y)
    text_obj.setFont("Helvetica", 11)
    for line in summary.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    y -= 100
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Key Points:")
    y -= 20
    text_obj = c.beginText(70, y)
    text_obj.setFont("Helvetica", 11)
    for bullet in bullets:
        text_obj.textLine(f"- {bullet}")
    c.drawText(text_obj)
    c.save()

# --- ffmpeg path ---
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\ffmpeg\bin\ffmpeg.exe"

st.set_page_config(page_title="Adaptive Learning Tool", layout="wide")
st.title("🎓 Adaptive Learning Tool")

# --- Languages ---
languages = {"English": "en", "Tamil": "ta", "Hindi": "hi", "Spanish": "es", "French": "fr"}
language_option = st.selectbox("Select your preferred language", list(languages.keys()))

uploaded_file = st.file_uploader("Upload Lecture Video/Audio", type=["mp4", "wav", "mp3"])

if uploaded_file:
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully!")

    try:
        # --- Extract audio ---
        st.info("🔊 Processing audio...")
        if uploaded_file.type == "video/mp4":
            clip = VideoFileClip(file_path)
            audio_path = "temp_audio.wav"
            clip.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
            del clip
        else:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export("temp_audio.wav", format="wav")
            audio_path = "temp_audio.wav"
            del audio
        gc.collect()

        # --- Whisper transcription ---
        st.info("⏳ Transcribing using Whisper...")
        model = whisper.load_model("small", device="cpu")
        if language_option == "English":
            result = model.transcribe(audio_path, task="translate")
        else:
            result = model.transcribe(audio_path, task="transcribe")

        transcript = result["text"]
        transcript = re.sub(r"[^a-zA-Z0-9\s.,!?%-]", " ", transcript)
        transcript = re.sub(r"\s+", " ", transcript).strip()

        st.subheader("📜 Transcript")
        st.write(transcript)

        # --- Download Transcript ---
        st.download_button(
            label="📥 Download Transcript",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )

        # --- Translation ---
        translated_text = transcript
        if language_option != "English":
            st.info(f"🌍 Translating to {language_option}...")
            lang_code = languages[language_option]
            model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model_mt = MarianMTModel.from_pretrained(model_name)
                batch = tokenizer([transcript], return_tensors="pt", padding=True)
                translated = model_mt.generate(**batch)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                del tokenizer, model_mt
                gc.collect()
            except:
                st.warning("⚠️ Translation failed. Using transcript.")

        # --- Summarization (lightweight) ---
        st.info("📝 Summarizing key points...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        summary_result = summarizer(translated_text, max_length=180, min_length=60, do_sample=False)
        summary_text = summary_result[0]['summary_text']
        del summarizer
        gc.collect()

        # Split summary into bullet points
        bullet_points = [f"- {s.strip()}." for s in summary_text.split(". ") if len(s.split()) > 3]

        st.subheader("📌 Summary / Key Points")
        for bp in bullet_points:
            st.write(bp)

        # --- Download Summary PDF ---
        if st.button("📥 Download Notes as PDF"):
            pdf_file = "learning_notes.pdf"
            create_pdf(summary_text, bullet_points, filename=pdf_file)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="⬇️ Save PDF",
                    data=f,
                    file_name=pdf_file,
                    mime="application/pdf"
                )

        # -----------------------------
        # QUIZ SECTION
        st.info("🎯 Quiz: Answer the questions below")

        q1 = "What is the formula of Force?"
        q1_options = ["F = a", "F = m · a", "F = v + d", "F = a + v"]
        q1_correct = "F = m · a"

        q2 = "What is the formula of Coulomb's law?"
        q2_options = [
            "F = k * (|q1*q2|) / r²",
            "F = a * (q2 - q1)",
            "F = a * d(q1,q2)",
            "F = r * a(q1+q2)"
        ]
        q2_correct = "F = k * (|q1*q2|) / r²"

        st.subheader("Q1")
        ans1 = st.radio(q1, q1_options, key="q1_radio")

        st.subheader("Q2")
        ans2 = st.radio(q2, q2_options, key="q2_radio")

        if st.button("Submit Quiz"):
            score = 0
            total = 2
            if ans1 == q1_correct:
                score += 1
                st.success("Q1: ✅ Correct!")
            else:
                st.error(f"Q1: ❌ Incorrect. Correct answer: {q1_correct}")
            if ans2 == q2_correct:
                score += 1
                st.success("Q2: ✅ Correct!")
            else:
                st.error(f"Q2: ❌ Incorrect. Correct answer: {q2_correct}")
            st.subheader(f"🎯 Total Score: {score} / {total}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("Upload a lecture video/audio (mp4, wav, mp3) to start.")
