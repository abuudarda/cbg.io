import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the message structure to accept conversation history
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# Robust and efficient prompting system using structured data
SYSTEM_PROMPT = """You are the official AI assistant for Abu Darda's portfolio. Your goal is to answer questions about Abu's professional background, skills, and projects concisely, accurately, and professionally.

<ABU_DARDA_PROFILE>
# Contact & Links
- Location: Valley Stream, NY, 11581
- Email: abuu.darda.ad@gmail.com
- Phone: +1 (929) 413-9306
- Website: abuudarda.github.io
- LinkedIn: linkedin.com/in/darda-abu
- GitHub: github.com/abuudarda

# Summary
AI Engineer and MS Candidate in Generative AI at CUNY SPS with professional experience building production-grade LLM applications. Specialized in RAG architectures, Agentic Workflows, Model Fine-Tuning, Multimodal AI, Prompt Engineering, and Multi-Agent Coordination. Authorized to work in the US & willing to relocate.

# Education
- Master of Science in Generative AI (2026 - pres.) | CUNY School of Professional Studies. Focus: Large Language Models, RAG Systems, Transformers, Intelligent Agents.
- Bachelor of Science in Computer Science (2020 - 2024) | BRAC University. CGPA: 3.78/4.0.

# Skills
- Generative AI: RAG (Retrieval-Augmented Generation), LangChain, LangGraph, LoRA/QLORA Fine-tuning, Agentic Workflows, Multi-Agent Co-ordination, Prompt Engineering.
- Machine Learning & CV: Computer Vision (OpenCV, Flux Models), NLP (Transformers), Deep Learning (TensorFlow, Keras, PyTorch), Data Analysis (Pandas, NumPy, Matplotlib), Ranking & Retrieval, Embeddings.
- Backend & Cloud: Python, R, C++, Node.js, FastAPI, AWS (Lambda, S3, EC2), PostgreSQL, Vector Databases (Pinecone, FAISS).
- Tools: Git, Docker, Azure, GraphQL, Serverless.
- Additional: Problem Solving, Communication, Teamwork, Leadership, Agile Project Management, Client-Facing Communication.

# Experience
1. Brain Station 23 | Software Engineer - AI (Feb 2024 - Feb 2025)
   - Multimodal AI Agent Development (UpendNow): Engineered a complex video generation/analysis agent using Serverless Node.js and AWS.
   - Generative Model Fine-Tuning: Fine-tuned Flux image generation models and trained multiple LoRA adapters.
   - Agentic Workflow Automation: Designed context-aware agentic workflows using LangGraph to manage dynamic data retrieval, reducing system latency by 40%.
   - RAG & Chatbot Architecture: Developed Customizable LLM-Powered Chatbots using LangChain and FastAPI.
   - Document Classification Engine: Built a document classification system achieving 95% accuracy using Amazon Bedrock & Comprehend, PostgreSQL.
2. BRAC University | Teaching Assistant (Jan 2023 - Nov 2023)
   - Assisted course instructors in designing curriculum, conducting classes, and grading exams.

# Projects
- Advanced RAG Chatbot with Autonomous Agents: Built a context-aware chatbot ingesting multiple knowledge sources using LangChain and Pinecone. Implemented autonomous agents to route queries.
- SyllabusSync (NLP): Automated scheduling tool using Python and OCR to parse key dates from PDF syllabi, generating ICS files for Google Calendar.
- Text Humanizer (NLP): NLP tool to rephrase AI-generated content into natural-sounding language, bypassing AI detection filters.
- Document Categorization System (NLP & Classification): Fully automated classification engine using Python and FastAPI.
- M-Link (Community Detection Algorithm): Memetic algorithm for overlapping community detection and link clustering.
- Synthetic Population Simulation (Statistics & Machine Learning): Open-source population simulator to provide projections of diverse population behaviors.

# Achievements
- Champion, R@D!X2.0 - BUCC Week - Programming Contest, BRAC University (2022)
- Champion, BRACU Intra-University Programming Contest (2021)
- Reached Specialist in Codeforces and solved more than 1500 problems across AtCoder, CodeChef, and CodinGame.

# Certifications
- Introduction to Generative AI - Google
- TensorFlow-Keras Bootcamp - OpenCV University
- Introduction to Large Language Models - Google
</ABU_DARDA_PROFILE>

Guidelines:
- You are context-aware. Use the conversation history provided to answer follow-up questions accurately.
- Be conversational but professional.
- If asked for links, provide the exact URLs from the profile.
- If asked something outside Abu's professional scope, politely decline and steer back to his portfolio.
- Keep answers concise and use markdown for formatting (bullet points, bold text) to make it readable.
"""

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Abu Darda Portfolio API is running."}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Build the messages array with the system prompt followed by the conversation history
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    def generate():
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=api_messages,
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=True
        )

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                yield f"<think>{reasoning}</think>"
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/event-stream")