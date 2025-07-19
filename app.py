import gradio as gr
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
MAX_NEW_TOKENS = 350
TEMPERATURE = 0.7

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if model.device.type == "cuda" else -1,
)

def build_prompt(system_prompt, user_prompt):
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def generate_email(outline, context, email_type, language, tone):
    logger.info(f"Generating email: type={email_type}, language={language}, tone={tone}")
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
    prompt = build_prompt(system_prompt, task_map[email_type])
    try:
        output = generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )[0]["generated_text"]
        if prompt in output:
            output = output[len(prompt):].strip()
        return output.strip()
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        return f"Error generating email. ({str(e)})"

with gr.Blocks(title="Magic Mail Writer (Open Hermes 2)", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ✉️ Magic Mail Writer (Open Nous Hermes 2 Mistral 7B)")
    gr.Markdown(
        "*Powered by [NousResearch/Nous-Hermes-2-Mistral-7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)*"
    )
    with gr.Row():
        email_type = gr.Radio(
            ["New", "Reply", "Forward"],
            value="New",
            label="Email Type",
            scale=1,
        )
        language = gr.Radio(
            ["English", "Spanish", "German"],
            value="English",
            label="Language",
            scale=1,
        )
        tone = gr.Radio(
            ["Casual", "Friendly", "Professional"],
            value="Friendly",
            label="Tone",
            scale=1,
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
        lines=5,
    )
    context_input = gr.Textbox(
        label="Context (Previous communication)",
        placeholder="Paste relevant context here...",
        lines=3,
    )
    output_text = gr.Textbox(
        label="Generated Email",
        interactive=False,
        lines=10,
        show_copy_button=True,
    )
    with gr.Row():
        generate_btn = gr.Button("Generate Email", variant="primary")
        clear_btn = gr.Button("Clear Output")
    generate_btn.click(
        fn=generate_email,
        inputs=[outline_input, context_input, email_type, language, tone],
        outputs=output_text,
    )
    clear_btn.click(lambda: "", None, output_text)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
