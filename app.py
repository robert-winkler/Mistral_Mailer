import streamlit as st
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get the Mistral API key from the environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Define the Mistral API endpoint
MISTRAL_API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Define the model to use
MODEL_NAME = "mistral-medium-latest"

# Define the maximum number of new tokens to generate
MAX_NEW_TOKENS = 350

# Define the temperature for the model
TEMPERATURE = 0.7

# Define the function to generate email
def generate_email(outline, context, email_type, language, tone):
    st.info(f"Generating email: type={email_type}, language={language}, tone={tone}")
    if not outline.strip():
        return ""

    tone_instructions = {
        "Casual": "Casual language with contractions. Short sentences.",
        "Friendly": "Warm tone, positive phrasing. Personal touches.",
        "Professional": "Formal but not stiff. Complete sentences."
    }

    system_prompt = (
        f"Write an email in {language} with the following tone: {tone_instructions[tone]}. "
        "Be concise yet nuanced. Maximum of 3 paragraphs. "
        f"Email type: {email_type}. "
        "Respond ONLY with the email body text."
    )

    task_map = {
        "New": f"Compose new email: {outline}",
        "Reply": f"Reply to this: {context}\n\nResponse outline: {outline}",
        "Forward": f"Forward this with comment: {context}\n\nComment: {outline}",
    }

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{task_map[email_type]}<|im_end|>\n<|im_start|>assistant\n"

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "do_sample": True,
        "eos_token_id": None,
        "pad_token_id": None,
    }

    try:
        response = requests.post(MISTRAL_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()["choices"][0]["text"]
        return output.strip()
    except Exception as e:
        st.error(f"Model inference error: {e}")
        return f"Error generating email. ({str(e)})"

# Streamlit app
st.title("Magic Mail Writer (Mistral Medium)")

st.markdown(
    "*Powered by [Mistral Medium](https://mistral.ai)*"
)

email_type = st.radio(
    "Email Type",
    ["New", "Reply", "Forward"],
    index=0,
)

language = st.radio(
    "Language",
    ["English", "Spanish", "German"],
    index=0,
)

tone = st.radio(
    "Tone",
    ["Casual", "Friendly", "Professional"],
    index=1,
)

outline_input = st.text_area(
    "Message Outline",
    placeholder=(
        "To: [Recipient]\n"
        "From: [Your Name]\n\n"
        "Key Points:\n"
        "- Know: [Essential information]\n"
        "- Feel: [Desired emotion]\n"
        "- Do: [Requested action]"
    ),
    height=150,
)

context_input = st.text_area(
    "Context (Previous communication)",
    placeholder="Paste relevant context here...",
    height=100,
)

if st.button("Generate Email"):
    output_text = generate_email(outline_input, context_input, email_type, language, tone)
    st.text_area("Generated Email", value=output_text, height=300)