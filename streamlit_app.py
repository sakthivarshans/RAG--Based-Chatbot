import streamlit as st
import time
from rag_graph import app

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")

st.title("Agentic AI RAG Chatbot")
st.markdown("Strictly grounded answers from `Ebook-Agentic-AI.pdf`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("Details"):
                st.write(f"**Confidence:** {message['metadata']['confidence']:.2f}")
                st.write("**Sources:**")
                for source in message['metadata']['used_context']:
                    st.text(source[:200] + "...")

if prompt := st.chat_input("Ask a question about Agentic AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            inputs = {"question": prompt}
            result = app.invoke(inputs)
            
            answer = result["answer"]
            confidence = result.get("confidence", 0.0)
            used_context = result.get("used_context", [])
            
            message_placeholder.markdown(answer)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "metadata": {
                    "confidence": confidence,
                    "used_context": used_context
                }
            })
            
            with st.expander("Details"):
                st.write(f"**Confidence:** {confidence:.2f}")
                st.write("**Sources:**")
                for source in used_context:
                    st.text(source[:200] + "...")
                    
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            message_placeholder.error(error_msg)
