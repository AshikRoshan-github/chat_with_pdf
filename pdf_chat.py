import streamlit as st
import toml
from google import genai
from google.genai import types

# Load configuration using Streamlit's environment method
GEMINI_API_KEY = st.secrets["api"]["GEMINI_API_KEY"]

def generate_response(file_path, user_prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Upload file to Gemini API
    uploaded_file = client.files.upload(file=file_path)
    
    model = "gemini-2.0-flash-thinking-exp-01-21"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type,
                ),
                types.Part.from_text(text=f"""**Prompt:**\n\n{user_prompt}"""),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=65536,
        response_mime_type="text/plain",
    )
    
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        response_text += chunk.text
    
    return response_text

st.title("PDF Query with Gemini AI")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    user_prompt = st.text_area("Enter your question about the PDF:")
    
    if st.button("Get Response"):
        with st.spinner("Processing..."):
            response = generate_response("temp_uploaded.pdf", user_prompt)
        
        st.subheader("Response:")
        st.write(response)
