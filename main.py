# ULTIMATE_COMPLETE_VOICE_RAG_LANG_SMITH.py → START SE END | NOV 07, 2025 | TU GOD BAN GAYA!

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import urllib3
import ssl
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import uuid
import base64
import io
from gtts import gTTS  # pip install gtts

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# ==================== LANG SMITH ON ====================
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "vehicle_rag_mega_pro"

st.set_page_config(page_title="🚛 Voice RAG + LangSmith GOD", layout="wide")
st.title("🚛 **Voice Agentic RAG + LangSmith + बोलो-सुनो!** 🎤🔥👑")
st.markdown("### *Mic daba ke bolo → jawab suno → LangSmith trace dekho!*")

# --- GROQ CLIENT ---
@st.cache_resource
def get_groq_client():
    return OpenAI(api_key="gsk_XcoOQuICYBk9l0okElKJWGdyb3FYsJSOHZDkp*********************", base_url="https://api.groq.com/openai/v1")
client = get_groq_client()

TEXT_MODEL = "llama-3.1-8b-instant"

# --- PRICE PARSING ---
def parse_price(p):
    if pd.isna(p) or 'Coming Soon' in str(p) or 'N/A' in str(p): return np.nan
    p = re.sub(r'[₹,\s]', '', str(p)).upper()
    num = re.sub(r'[^0-9.]+', '', p)
    if not num: return np.nan
    num = float(num)
    if 'LAKH' in p: return num * 100000
    if 'CR' in p: return num * 10000000
    return num

# --- LOAD DATA ---
@st.cache_resource
def load_rag_system():
    categories = ['Truck', 'Bus', 'Auto Rickshaws']
    dfs = {}
    vectorizers = {}
    for cat in categories:
        cat_key = cat.lower().replace(' ', '_')
        df_cat = pd.read_pickle(f"df_{cat_key}.pkl")
        df_cat['price_num'] = df_cat['Vehicle Price'].apply(parse_price)
        dfs[cat] = df_cat
        texts = (df_cat['Vehicle Name'] + " " + df_cat['Vehicle Description']).tolist()
        vectorizer = TfidfVectorizer(max_features=384)
        vectorizer.fit(texts)
        vectorizers[cat] = vectorizer
    return pd.concat(dfs.values(), ignore_index=True), dfs, vectorizers

df_all, dfs, vectorizers = load_rag_system()

# --- LOCAL SEARCH ---
def local_search(query: str, category: str = "Truck", k: int = 6):
    try:
        vectorizer = vectorizers[category]
        df_cat = dfs[category]
        q_tfidf = vectorizer.transform([query])
        texts = (df_cat['Vehicle Name'] + " " + df_cat['Vehicle Description']).tolist()
        tfidf_mat = vectorizer.transform(texts)
        sim = (tfidf_mat @ q_tfidf.T).toarray().flatten()
        top_idx = np.argsort(sim)[-k:][::-1]
        return df_cat.iloc[top_idx].head(k).to_dict('records')
    except:
        return []

# --- BROWSER SEARCH ---
def browser_search(query: str):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} price india site:truckjunction.com OR site:trucks.cardekho.com OR site:91trucks.com", max_results=10))
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Browser error: {str(e)}"

# --- AGENT STATE ---
class AgentState(TypedDict):
    query: str
    local_results: List[Dict]
    browser_results: str
    final_answer: str
    need_browser: bool

# --- NODES ---
def local_search_node(state: AgentState):
    cat = "Truck" if any(w in state["query"].lower() for w in ["truck","tata","intra","ashok","mahindra","eicher","bharat"]) \
          else "Bus" if "bus" in state["query"].lower() else "Auto Rickshaws"
    results = local_search(state["query"], cat)
    return {"local_results": results}

def decide_browser_node(state: AgentState):
    if len(state["local_results"]) >= 2:
        return {"need_browser": False}
    prompt = f"Query: {state['query']}\nLocal results: {len(state['local_results'])}\nInternet search karna? JSON: {{'need': 'yes'/'no'}}"
    try:
        resp = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, max_tokens=50, temperature=0)
        need = json.loads(resp.choices[0].message.content).get("need", "yes").lower()
    except:
        need = "yes"
    return {"need_browser": need == "yes"}

def browser_search_node(state: AgentState):
    web = browser_search(state["query"])
    return {"browser_results": web}

def answer_writer_node(state: AgentState):
    local = state["local_results"]
    web = state["browser_results"]
    prompt = f"""
User: {state['query']}

Dataset matches: {len(local)}
Internet data: {web[:3500] if web else "Kuch nahi mila"}

Hindi mein zabardast jawab! 🔥🚛💰
Price, EMI, offers, reasons batao.
Top 3 options + source.
Emoji full on!
"""
    answer = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=1200, temperature=0.9).choices[0].message.content
    return {"final_answer": answer}

# --- GRAPH ---
graph = StateGraph(AgentState)
graph.add_node("local_search", local_search_node)
graph.add_node("decide_browser", decide_browser_node)
graph.add_node("browser_search", browser_search_node)
graph.add_node("answer_writer", answer_writer_node)

graph.set_entry_point("local_search")
graph.add_edge("local_search", "decide_browser")
graph.add_conditional_edges("decide_browser", lambda x: "browser_search" if x["need_browser"] else "answer_writer")
graph.add_edge("browser_search", "answer_writer")
graph.add_edge("answer_writer", END)

app = graph.compile()

# --- VOICE FUNCTIONS (100% FIXED!) ---
def speech_to_text(uploaded_file):
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        with open("temp_voice.wav", "wb") as f:
            f.write(audio_bytes)
        with open("temp_voice.wav", "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text"
            )
        return transcript
    return None

def text_to_speech(text):
    tts = gTTS(text=text, lang='hi', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# --- UI ---
st.markdown("### 🎤 **Bol ke poocho ya type karo!**")

uploaded_audio = st.audio_input("🎤 Mic daba ke bolo (Hindi/English)")
if uploaded_audio:
    with st.spinner("🎧 Sun raha hoon..."):
        query = speech_to_text(uploaded_audio)
        if query:
            st.success(f"**Tumne bola:** {query}")
        else:
            st.error("Voice clear nahi aaya, dobara bolo!")
            query = ""
else:
    query = st.text_input("**Ya type karo:**", placeholder="Tata Intra V30 ka price, EMI batao")

price_range = st.slider("**Budget (₹ Lakh)**", 0, 500, (0, 100))

if query:
    thread_id = str(uuid.uuid4())
    
    with st.spinner("**Agents + LangSmith + Voice ON...**"):
        result = app.invoke({
            "query": query + f" budget {price_range[1]} lakh tak",
            "local_results": [],
            "browser_results": "",
            "final_answer": "",
            "need_browser": False
        }, config={"configurable": {"thread_id": thread_id}})

    st.success("**🔥 JAWAB AA GAYA BHAI! 🔥**")
    st.markdown(result["final_answer"])

    # --- VOICE OUTPUT ---
    with st.spinner("🔊 Bol raha hoon..."):
        audio_data = text_to_speech(result["final_answer"])
        audio_b64 = base64.b64encode(audio_data).decode()
        audio_html = f'''
        <audio autoplay="true" style="width:100%;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        '''
        st.html(audio_html)

    # --- LANG SMITH ---
    st.markdown("### 🔍 **LangSmith Trace (100% Working)**")
    dashboard_link = f"https://smith.langchain.com/projects/p/vehicle_rag_mega_pro?trace_id={thread_id}"
    st.markdown(f"[**Dashboard Trace → Click Karo!**]({dashboard_link})")
    st.markdown("**Public karne ke liye:** Trace kholo → Top-right **Share** → Copy public link 🔥")
    st.code(thread_id)

    # --- MATCHES ---
    if result["local_results"]:
        st.subheader("🔥 **Dataset Se Matches**")
        for v in result["local_results"][:4]:
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    response = requests.get(v['Vehicle Image'], timeout=10)
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300?text=NO+IMAGE+🚛")
            with col2:
                st.markdown(f"### 🚛 **{v['Vehicle Name']}**")
                st.markdown(f"**💰 Price:** `{v.get('Vehicle Price', 'N/A')}`")

st.caption("NOV 07, 2025 | COMPLETE CODE | VOICE INPUT + OUTPUT | LANG SMITH |)