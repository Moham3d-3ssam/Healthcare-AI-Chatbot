import gradio as gr
import os
import re
import torch
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

# =========================
# Model
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tokenizer_path = os.path.join(BASE_DIR, "saved_files", "tokenizer")
model_path = os.path.join(BASE_DIR, "saved_files", "chatbot")

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# Clean text
# =========================
def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

# =========================
# Chat function (Streaming)
# =========================
def chat(message, history):

    message = clean_text(message)

    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    yield "Thinking..."

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    streamed = ""
    for word in response.split():
        streamed += word + " "
        time.sleep(0.02)
        yield streamed.strip()

# =========================
# Reset chat (FIXED)
# =========================
def reset_chat():
    return None

# =========================
# UI
# =========================
css = """
.gradio-container {
    background: radial-gradient(circle at top, #111827, #0b1220);
    color: white;
}

footer {display:none !important;}
"""

theme = gr.themes.Soft(primary_hue="blue", radius_size="lg")

with gr.Blocks(theme=theme, css=css) as demo:

    gr.Markdown("# 🏥 Healthcare AI Chatbot")

    # ================= CHAT =================
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=None
    )

demo.queue().launch()
