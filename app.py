import streamlit as st
import time
import yt_dlp
import requests

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ==============================
# 🎨 Page Config
# ==============================
st.set_page_config(page_title="AskTube 🎥", layout="centered")

st.markdown("""
<style>
.stChatMessage {
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎥 AskTube")
st.caption("Chat with any YouTube video using AI 🚀")

# ==============================
# 📌 Sidebar
# ==============================
with st.sidebar:
    st.header("🎥 Load Video")
    video_url = st.text_input("Enter YouTube URL")
    process_btn = st.button("Process Video")

# ==============================
# 🔧 Extract Video ID
# ==============================
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return url

# ==============================
# 🎬 Get Video Title
# ==============================
def get_video_title(url):
    try:
        ydl_opts = {"quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "YouTube Video")
    except:
        return "YouTube Video"

# ==============================
# 📄 Get Transcript (with fallback)
# ==============================
def get_transcript(video_id, video_url):

    # Try youtube-transcript-api
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id)
        return " ".join(chunk.text for chunk in transcript_data)
    except:
        pass

    # Fallback: yt-dlp subtitles
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if subtitles and "en" in subtitles:
                sub_url = subtitles["en"][0]["url"]
                res = requests.get(sub_url)
                return res.text

    except:
        pass

    return None

# ==============================
# 🚀 Process Video
# ==============================
if process_btn and video_url:
    video_id = extract_video_id(video_url)

    try:
        progress = st.progress(0, text="Starting...")

        # 🎬 Title + Thumbnail
        progress.progress(10, text="Fetching video info...")
        video_title = get_video_title(video_url)
        st.markdown(f"### 🎬 {video_title}")
        st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg")

        # 📄 Transcript
        progress.progress(30, text="Fetching transcript...")
        transcript = get_transcript(video_id, video_url)

        if not transcript:
            st.error("❌ Could not fetch transcript for this video.")
            st.stop()

        # ✂️ Split
        progress.progress(50, text="Splitting text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        # 🧠 Embeddings
        progress.progress(70, text="Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 🤖 LLM
        progress.progress(85, text="Loading AI model...")
        llm = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model="llama-3.1-8b-instant",
            temperature=0.2
        )

        # 📜 Prompt
        prompt = PromptTemplate(
            template="""
Answer ONLY from context.
If not found, say "I don't know".

{context}
Question: {question}
""",
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state.chain = chain
        st.success("✅ Video ready! Ask questions below 👇")
        progress.progress(100, text="Done!")

    except Exception as e:
        st.error(f"Error: {e}")

# ==============================
# 💬 Chat UI
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "chain" in st.session_state:
    user_input = st.chat_input("Message AskTube...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            answer = st.session_state.chain.invoke(user_input)

            for word in answer.split():
                full_response += word + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)

            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==============================
# 🗑️ Clear Chat
# ==============================
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
