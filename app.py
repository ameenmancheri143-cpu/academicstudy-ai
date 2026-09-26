import streamlit as st
import ollama
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Makes the app wider and adds a browser tab icon
st.set_page_config(page_title="My AI Assistant", page_icon="🤖", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("⚙️ AI Dashboard")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    # 2. Settings Menu
    with st.expander("🛠️ Advanced Settings"):
        st.session_state.temperature = st.slider(
            "Temperature (Creativity)", 
            min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1,
            help="Higher values make the AI more creative, lower values make it more strict/factual."
        )
        chunk_size = st.number_input("PDF Chunk Size", min_value=500, max_value=2000, value=1000, step=100)

    st.divider()

    # 3. Mode Switcher
    st.header("🧠 Interaction Mode")
    mode = st.radio("Select Mode:", ["🌍 General Knowledge", "📖 Study my PDF"], label_visibility="collapsed")
    
    # 4. PDF Upload (Only shows if Study Mode is selected)
    if mode == "📖 Study my PDF":
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload your PDF textbook", type="pdf")
        
        if uploaded_file:
            if "vector_db" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
                with st.spinner("Analyzing document..."):
                    try:
                        pdf_reader = PdfReader(uploaded_file)
                        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                        
                        if not text.strip():
                            st.error("No text found. Is this a scanned image?")
                        else:
                            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
                            chunks = splitter.split_text(text)
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                            st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)
                            st.session_state.current_file = uploaded_file.name
                            st.success("Document ready!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

# --- MAIN CHAT UI ---
st.title("🎓 Academic Study AI")
st.caption("Your local, privacy-first engineering assistant.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Logic
if prompt := st.chat_input("Type your question here..."):
    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Safely copy messages so we don't permanently mess up the visual history
    ai_messages = st.session_state.messages.copy()
    
    if mode == "📖 Study my PDF":
        if "vector_db" in st.session_state:
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Secretly inject the PDF context into the very last message sent to the AI
            ai_messages[-1] = {
                "role": "user",
                "content": f"Use the following text to answer the question.\n\nContext:\n{context}\n\nQuestion: {prompt}"
            }
        else:
            st.warning("Upload a PDF in the sidebar first! Answering from general knowledge instead.")
    
    # Generate Response from Ollama
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # We pass the temperature setting directly into the Ollama options
            stream = ollama.chat(
                model='llama3.2', 
                messages=ai_messages, 
                stream=True,
                options={"temperature": st.session_state.temperature}
            )
            for chunk in stream:
                full_response += chunk['message']['content']
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error connecting to Ollama: {e}"
            response_placeholder.error(full_response)
            
    # Save the final response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})