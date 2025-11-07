# Vehicle Voice RAG Agent  
**Hindi Voice-Enabled AI Agent for Truck Dealers | LangGraph + OpenAI + LangSmith**

---

## Objective  
Built for **91Trucks** — a startup serving **500+ commercial vehicle dealers**.  
Empower dealers to answer customer queries in **Hindi via voice** in **<10 seconds** — reducing manual lookup by **90%** and **query time by 40%**.

---

## Key Features  
- **Voice Input** → OpenAI Whisper (STT)  
- **Voice Output** → gTTS (TTS with auto-play)  
- **Hybrid RAG** → TF-IDF (local) + DuckDuckGo (web) + CLIP (image search)  
- **LangGraph Agent** → Multi-step: `local → decide → web → answer`  
- **LangSmith (Paid)** → Full trace visibility + analytics  
- **Streamlit UI** → Budget slider, real images, auto-play audio

---

## Tech Stack  
```text
Python | LangGraph | ChatOpenAI (GPT-4o) | Whisper | gTTS  
LangSmith (Paid) | TF-IDF | DuckDuckGo | CLIP | Streamlit  
Pandas | NumPy | Requests | PIL | Git
