import os
import boto3
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load env variables
load_dotenv()

st.set_page_config(page_title="KB + Gemini Chat", page_icon="💬")
st.title("Bedrock KB + Gemini 2.5 Flash")

# AWS config
aws_region = os.getenv("AWS_REGION", "us-east-1")
knowledge_base_id = '7W18VTUWOF'

# ✅ FIX: Proper API key handling
api_key = os.getenv("gemini_key")

if not api_key:
    st.error("❌ GEMINI API KEY not found. Set it in .env file")
    st.stop()

genai.configure(api_key=api_key)

st.write("This app retrieves context from Amazon Bedrock Knowledge Base and uses Gemini to answer.")

# User input
question = st.text_input("Enter your question")

if st.button("Ask"):
    if not knowledge_base_id:
        st.error("Set KNOWLEDGE_BASE_ID.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        # Bedrock client
        bedrock = boto3.client("bedrock-agent-runtime", region_name=aws_region)

        kb_response = bedrock.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            },
        )

        # Extract context
        chunks = []
        for item in kb_response.get("retrievalResults", []):
            text = item.get("content", {}).get("text", "")
            if text:
                chunks.append(text)

        context = "\n\n".join(chunks)

        prompt = f"""
Answer the user's question using only the context below.

Context:
{context}

Question:
{question}
"""

        # ✅ FIX: Correct Gemini usage (OLD SDK)
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content(prompt)

        st.subheader("Answer")
        st.write(response.text)

        with st.expander("Retrieved Context"):
            st.write(context if context else "No context returned.")