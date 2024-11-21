from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
import openai
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM, AutoTokenizer
from constants import hf_key, anthropic_key, cohere_key
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline, Cohere
from fpdf import FPDF
import numpy as np
import librosa
import soundfile as sf
import uvicorn

app = FastAPI()

os.environ['HUGGINGFACEHUB_API_TOKEN']=hf_key
os.environ['COHERE_API_KEY']=cohere_key
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

prompt_template = "Summarize the following conversation:\n{transcription_text}"
prompt = PromptTemplate(input_variables=["transcription_text"], template=prompt_template)

cohere_llm = Cohere(
    model="command-xlarge-nightly",  
    temperature=0.5,
    max_tokens=500,
)

llm_chain = LLMChain(llm=cohere_llm, prompt=prompt)

class TranscriptionResponse(BaseModel):
    text: str
    timestamps: list[dict]

class SummaryResponse(BaseModel):
    summary: str

def convert_video_to_audio(video_data):
    try:
        # Save video data to temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(video_data)
        
        # Extract audio using moviepy
        video_clip = VideoFileClip("temp_video.mp4")
        video_clip.audio.write_audiofile("temp_audio.wav")
        video_clip.close()
        
        # Load and process audio using librosa
        audio_data, sr = librosa.load("temp_audio.wav", sr=16000, mono=True)
        return audio_data
        
    except Exception as e:
        print(f"Error in convert_video_to_audio: {str(e)}")
        raise e
    finally:
        # Clean up video clip if it exists
        if 'video_clip' in locals():
            try:
                video_clip.close()
            except:
                pass

def clean_up_temp_files():
    files_to_remove = ["temp_video.mp4", "temp_audio.wav", "meeting_summary.pdf"]
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Error removing {file}: {str(e)}")

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

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(video: UploadFile = File(...)):
    try:
        video_data = await video.read()
        audio_data = convert_video_to_audio(video_data)
        
        result = pipe(audio_data, generate_kwargs={'language': 'en'}, return_timestamps=True)
        transcription_text = result["text"]
        timestamps = result['chunks']
        
        return TranscriptionResponse(text=transcription_text, timestamps=timestamps)
    except Exception as e:
        print(f"Error in transcribe_video: {str(e)}")
        raise e
    finally:
        clean_up_temp_files()

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_transcription(transcription_text: str):
    try:
        summary = llm_chain.run(transcription_text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        print(f"Error in summarize_transcription: {str(e)}")
        raise e

@app.post("/generate_pdf")
async def generate_pdf_file(background_task: BackgroundTasks, video: UploadFile = File(...)):
    try:
        print("Starting PDF generation process...")
        
        transcription_response = await transcribe_video(video)
        # print("Transcription completed:", transcription_response)
        
        transcription_text = transcription_response.text
        summary_response = await summarize_transcription(transcription_text)
        # print("Summary completed:", summary_response)
        
        summary_text = summary_response.summary
        pdf_path = generate_pdf(transcription_response.timestamps, summary_text)
        print("PDF generated:", pdf_path)
        background_task.add_task(clean_up_temp_files)
        
        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename="meeting_summary.pdf",
        )
    except Exception as e:
        print(f"Error in generate_pdf_file: {str(e)}")
        clean_up_temp_files()
        raise e

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
