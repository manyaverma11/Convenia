from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from constants import hf_key, cohere_key
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere
from fpdf import FPDF
import numpy as np
import librosa
import soundfile as sf
import uvicorn

app = FastAPI()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_key
os.environ['COHERE_API_KEY'] = cohere_key
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Models
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, torch_dtype=torch_dtype, device=device)

prompt_template = "Summarize the following conversation:\n{transcription_text}"
prompt = PromptTemplate(input_variables=["transcription_text"], template=prompt_template)
cohere_llm = Cohere(model="command-xlarge-nightly", temperature=0.5, max_tokens=500)
llm_chain = LLMChain(llm=cohere_llm, prompt=prompt)

# In-memory storage for session data
meeting_transcriptions: dict[str, dict] = {}

class TranscriptionResponse(BaseModel):
    text: str
    timestamps: list[dict]

# class SummaryResponse(BaseModel):
#     summary: str

def convert_video_to_audio(video_data):
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_data)
        video_clip = VideoFileClip("temp_video.mp4")
        video_clip.audio.write_audiofile("temp_audio.wav")
        video_clip.close()
        audio_data, _ = librosa.load("temp_audio.wav", sr=16000, mono=True)
        return audio_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in converting video to audio: {e}")
    finally:
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

def generate_pdf(transcription_text, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add summary section
    pdf.cell(200, 10, txt="Meeting Summary", ln=True, align="C")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=summary_text)
    pdf.ln(10)

    if isinstance(transcription_text, list):
        transcription_text = "\n".join(str(item) for item in transcription_text)

    # Add transcription section
    pdf.cell(200, 10, txt="Full Transcription", ln=True, align="C")
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=transcription_text)

    pdf_path = "meeting_summary.pdf"
    pdf.output(pdf_path)
    return pdf_path

@app.post("/transcribe/{session_id}", response_model=TranscriptionResponse)
async def transcribe_video(session_id: str, video: UploadFile = File(...)):
    if session_id not in meeting_transcriptions:
        meeting_transcriptions[session_id] = {"text": "", "timestamps": []}

    try:
        video_data = await video.read()
        audio_data = convert_video_to_audio(video_data)

        result = pipe(audio_data, generate_kwargs={'language': 'en'}, return_timestamps=True)
        transcription_text = result["text"]
        timestamps = result['chunks']

        # Append to session transcription
        meeting_transcriptions[session_id]["text"] += " " + transcription_text
        meeting_transcriptions[session_id]["timestamps"].extend(timestamps)

        return TranscriptionResponse(text=transcription_text, timestamps=timestamps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in transcribing video: {e}")
    finally:
        clean_up_temp_files()

@app.post("/end_session/{session_id}")
async def end_meeting(session_id: str, background_task: BackgroundTasks):
    if session_id not in meeting_transcriptions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        transcription_data = meeting_transcriptions[session_id]
        transcription_text = transcription_data["text"]

        summary_text = llm_chain.run(transcription_text)

        # Generate PDF
        pdf_path = generate_pdf(transcription_data['timestamps'], summary_text)
        print("PDF generated:",pdf_path)
        # Cleanup and reset session data
        background_task.add_task(clean_up_temp_files)
        del meeting_transcriptions[session_id]

        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{session_id}_meeting_summary.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ending session: {e}")

def clean_up_temp_files():
    files_to_remove = ["meeting_summary.pdf"]
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
