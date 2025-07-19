# Magic Mail Writer - ZeroGPU Optimized
# pip install transformers==4.41.0 gradio==4.24.0 torch==2.5.1

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for ZeroGPU compatibility
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
CACHE_DIR = "model_cache"
MAX_TOKENS = 350
TEMPERATURE = 0.7

# Load tokenizer once (CPU operation)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

# ZeroGPU decorator for GPU-dependent function
@gpu
def generate_email(outline, context, email_type, language, tone):
    """Generate email with ZeroGPU optimizations"""
    logger.info("GPU function triggered - allocating GPU resources")
    
    if not outline.strip():
        return ""
    
    # Load model only when GPU is available
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    # Create generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Tone mapping with concise style
    tone_instructions = {
        "Casual": "Casual language with contractions. Short sentences.",
        "Friendly": "Warm tone, positive phrasing. Personal touches.",
        "Professional": "Formal but not stiff. Complete sentences."
    }
    
    # System prompt for nuanced multilingual writing
    system_prompt = (
        f"Write email in {language}. Tone: {tone_instructions[tone]}. "
        f"Be concise yet nuanced. Max 3 paragraphs. "
        f"Email type: {email_type}. "
        f"Respond ONLY with email body text."
    )
    
    # Dynamic task instructions
    task_map = {
        "New": f"Compose new email: {outline}",
        "Reply": f"Reply to this: {context}\n\nResponse outline: {outline}",
        "Forward": f"Forward this with comment: {context}\n\nComment: {outline}"
    }
    
    # Format messages using Mistral instruct template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_map[email_type]}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate with controlled parameters
    outputs = generator(
        prompt,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    
    # Extract response
    response = outputs[0]['generated_text']
    return response.split("assistant\n")[-1].strip()

# Gradio interface with ZeroGPU optimizations
with gr.Blocks(title="Magic Mail Writer", theme=gr.themes.Soft()) as app:
    gr.Markdown("## ✉️ ZeroGPU Email Writer")
    gr.Markdown("*Optimized for Mistral/Zephyr model with friendly-concise tone*")
    
    with gr.Row():
        email_type = gr.Radio(
            ["New", "Reply", "Forward"],
            value="New",
            label="Email Type",
            scale=1
        )
        language = gr.Radio(
            ["English", "Spanish", "German"],
            value="English",
            label="Language",
            scale=1
        )
        tone = gr.Radio(
            ["Casual", "Friendly", "Professional"],
            value="Friendly",
            label="Tone",
            scale=1
        )
    
    outline_input = gr.Textbox(
        label="Message Outline",
        placeholder=(
            "To: [Recipient]\n"
            "From: [Your Name]\n\n"
            "Key Points:\n"
            "- Know: [Essential information]\n"
            "- Feel: [Desired emotion]\n"
            "- Do: [Requested action]"
        ),
        lines=5
    )
    
    context_input = gr.Textbox(
        label="Context (Previous communication)",
        placeholder="Paste relevant context here...",
        lines=3
    )
    
    output_text = gr.Textbox(
        label="Generated Email",
        interactive=False,
        lines=10,
        show_copy_button=True
    )
    
    with gr.Row():
        generate_btn = gr.Button("Generate Email", variant="primary")
        clear_btn = gr.Button("Clear Output")
    
    generate_btn.click(
        fn=generate_email,
        inputs=[outline_input, context_input, email_type, language, tone],
        outputs=output_text
    )
    clear_btn.click(lambda: "", None, output_text)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        pwa=True
    )
# Magic Mail Writer - Mistral API Version
import os
import gradio as gr
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_TOKENS = 350
TEMPERATURE = 0.7
MODEL = "mistral-medium"  # Suitable for multilingual emails

# Initialize Mistral client
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    logger.error("MISTRAL_API_KEY not found in .env file")
    raise ValueError("API key missing")
    
client = MistralClient(api_key=mistral_api_key)

def generate_email(outline, context, email_type, language, tone):
    """Generate email using Mistral API"""
    logger.info(f"Generating email with type={email_type}, language={language}, tone={tone}")
    
    if not outline.strip():
        return ""
    
    # Tone mapping
    tone_instructions = {
        "Casual": "Casual language with contractions. Short sentences.",
        "Friendly": "Warm tone, positive phrasing. Personal touches.",
        "Professional": "Formal but not stiff. Complete sentences."
    }
    
    # System prompt
    system_prompt = (
        f"Write email in {language}. Tone: {tone_instructions[tone]}. "
        f"Be concise yet nuanced. Max 3 paragraphs. "
        f"Email type: {email_type}. "
        f"Respond ONLY with email body text."
    )
    
    # Task instructions
    task_map = {
        "New": f"Compose new email: {outline}",
        "Reply": f"Reply to this: {context}\n\nResponse outline: {outline}",
        "Forward": f"Forward this with comment: {context}\n\nComment: {outline}"
    }
    
    # Create messages
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=task_map[email_type])
    ]
    
    # Generate response
    try:
        response = client.chat(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return "Error generating email. Please try again."

# Gradio interface
with gr.Blocks(title="Magic Mail Writer", theme=gr.themes.Soft()) as app:
    gr.Markdown("## ✉️ Mistral API Email Writer")
    gr.Markdown("*Using Mistral API for multilingual email generation*")
    
    with gr.Row():
        email_type = gr.Radio(
            ["New", "Reply", "Forward"],
            value="New",
            label="Email Type",
            scale=1
        )
        language = gr.Radio(
            ["English", "Spanish", "German", "French", "Italian"],
            value="English",
            label="Language",
            scale=1
        )
        tone = gr.Radio(
            ["Casual", "Friendly", "Professional"],
            value="Friendly",
            label="Tone",
            scale=1
        )
    
    outline_input = gr.Textbox(
        label="Message Outline",
        placeholder=(
            "To: [Recipient]\n"
            "From: [Your Name]\n\n"
            "Key Points:\n"
            "- Know: [Essential information]\n"
            "- Feel: [Desired emotion]\n"
            "- Do: [Requested action]"
        ),
        lines=5
    )
    
    context_input = gr.Textbox(
        label="Context (Previous communication)",
        placeholder="Paste relevant context here...",
        lines=3
    )
    
    output_text = gr.Textbox(
        label="Generated Email",
        interactive=False,
        lines=10,
        show_copy_button=True
    )
    
    with gr.Row():
        generate_btn = gr.Button("Generate Email", variant="primary")
        clear_btn = gr.Button("Clear Output")
    
    generate_btn.click(
        fn=generate_email,
        inputs=[outline_input, context_input, email_type, language, tone],
        outputs=output_text
    )
    clear_btn.click(lambda: "", None, output_text)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want public sharing
        pwa=True
    )