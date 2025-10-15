🎓 Adaptive Learning Tool from Videos and Notes

📄 Overview

The Adaptive Learning Tool from Videos and Notes is an AI-powered educational platform that transforms lecture videos or notes into personalized learning materials.
It automatically extracts audio from videos, converts speech into text, summarizes content, translates it into multiple languages, and even generates quizzes — creating an adaptive and efficient learning experience for students.

This project is designed to save time for learners, improve understanding, and simplify note-taking using advanced NLP and Speech Recognition techniques.

🎯 Objectives

Automate the process of learning material creation from lecture videos.

Provide clear transcripts, summaries, and translations of video content.

Enable learners to interactively learn and test their understanding.

Build a user-friendly platform for adaptive and accessible learning.

⚙️ Key Features

✅ Video Upload & Processing – Upload lecture videos or audio files.
✅ Speech-to-Text Transcription – Converts spoken content into accurate text using Whisper AI.
✅ Summarization – Generates concise key points using NLP models.
✅ Translation – Translates the summarized text into the selected language.
✅ Quiz Generation – Automatically creates quiz questions from the summarized content.
✅ PDF Download Option – Users can download transcripts, summaries, or quizzes.
✅ Clean, Interactive Streamlit UI – Simple and user-friendly web interface for smooth usage.

🧰 Tech Stack

Category	Tools / Libraries
Language	Python
Framework	Streamlit
Speech Processing	Whisper (OpenAI)
NLP Models	T5, MarianMT, Transformers (Hugging Face)
Data Handling	Pandas, re
Export	ReportLab / PyPDF2
Others	ffmpeg (for audio extraction), os, io

🧩 System Architecture

Input Layer:
🎥 Lecture Video / Audio Upload

Audio Processing:
🔊 Extract audio and standardize format

Speech-to-Text:
📝 Transcribe using Whisper AI

Text Processing:
🧹 Clean and structure transcribed text

Summarization & Translation:
💬 Use NLP models (T5, MarianMT) for summary and language translation

Output Layer:

📘 Transcript
📗 Summarized Key Points
🌍 Translation in Selected Language
🧠 Auto-Generated Quiz

🧠 Workflow

Upload a lecture video or audio file.

The system extracts and processes the audio using Whisper AI.

Transcription text is cleaned and formatted.

The T5 model summarizes the lecture into key points.

The MarianMT model translates content into the selected language.

The system generates quiz questions from the summarized content.

The user can view, download, and interact with the results directly from the Streamlit app.

📊 Example Output

Output Type	Description
Transcript	Full text extracted from the lecture video
Summary	Concise bullet points of the main concepts
Translation	Summary converted into a chosen language
Quiz	Auto-generated questions for self-assessment

🚀 How to Run the Project

1️⃣ Clone the Repository
git clone https://github.com/yourusername/adaptive-learning-tool.git

2️⃣ Navigate to the Project Directory
cd adaptive-learning-tool

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run adaptive_learning_app.py

🧩 Requirements

Python 3.8+

ffmpeg installed (for audio extraction)

Internet connection (for using pre-trained AI models)

💡 Future Enhancements

Support for multi-speaker lecture videos.

Integration with Learning Management Systems (LMS).

Improved quiz generation using advanced LLMs (e.g., GPT models).

Visualization of learning progress and analytics dashboard.

Offline model support for faster processing.

🧠 Learning Outcomes

Hands-on experience with AI-based adaptive learning systems.

Understanding of speech-to-text, text summarization, and translation pipelines.

Building real-time Streamlit applications for educational AI use cases.

Integration of multiple NLP and deep learning models in a unified workflow.

🤝 Contribution

Contributions, suggestions, and feedback are always welcome!
Feel free to fork this repository and submit a pull request to enhance features or performance.

📬 Contact

Author: Mohanapriya
🎓 Final Year IT Student, Annamalai University
📧 [mohanaaselvi77@gmail.com]
🌐 [https://github.com/mohana-priyaa]
