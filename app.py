"""
Gradio demosu: prompt -> model (GPU) -> gomulu three.js viewer otomatik oynatir.
Colab'da: pip install gradio + (CLIP) -> python app.py  (share=True public link verir)
Ayni dosya HF Spaces'e oldugu gibi tasinir.

Checkpoint: CKPT yolunu ayarla (lokal dosya ya da hf_hub_download ile).
"""
import os
import sys
import json
import html

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import gradio as gr
from infer import MotionGenerator

CKPT = os.environ.get("CKPT", "motion_denoiser_h3d.pt")   # gerekirse degistir
gen = MotionGenerator(CKPT)
print("model yuklendi:", CKPT, "| device:", gen.device)

# --- gomulu viewer (kendi head/importmap'iyle izole calissin diye iframe srcdoc) ---
_viewer_path = os.path.join(os.path.dirname(__file__), "web", "viewer.html")
_viewer_html = open(_viewer_path, encoding="utf-8").read()
IFRAME = (
    '<iframe id="mviewer" style="width:100%;height:620px;border:0;border-radius:14px" '
    'srcdoc="' + html.escape(_viewer_html, quote=True) + '"></iframe>'
)

# parent -> iframe veri kopru (uretilen JSON'u viewer'a postMessage ile yolla)
PUSH_JS = "(j)=>{document.getElementById('mviewer').contentWindow.postMessage({motion:JSON.parse(j)},'*'); return [];}"
# Generate'e basinca viewer'da yukleme halkasini ac
LOAD_JS = "()=>{document.getElementById('mviewer').contentWindow.postMessage({loading:true},'*');}"

CSS = "footer{display:none !important}"


def run(prompt, guidance, seq_len):
    data = gen.generate(prompt.strip(), seq_len=int(seq_len), guidance=float(guidance))
    return json.dumps(data)


with gr.Blocks(title="Text to 3D Human Motion", css=CSS) as demo:
    gr.Markdown(
        "## Text-to-3D Human Motion\n"
        "Sıfırdan eğitilmiş bir diffusion modeli. İngilizce bir prompt yaz ve **Generate**'e bas; "
        "üretilen hareket aşağıdaki 3B sahnede oynar."
    )
    with gr.Row():
        prompt = gr.Textbox(label="Prompt (English)", scale=4,
                            value="a person walks in a circle")
        btn = gr.Button("Generate", variant="primary", scale=1)
    with gr.Row():
        guidance = gr.Slider(1.0, 5.0, value=2.5, step=0.5, label="Guidance (hareket prompta ne kadar sadık kalsın)")
        seq_len = gr.Slider(40, 196, value=120, step=4, label="Length (frames)")
    gr.Examples(
        ["a person walks forward", "a person walks in a circle", "a person runs",
         "a person jumps", "a person sits down", "a person waves","a person kicks something",
         "a person fights with someone"],
        inputs=prompt)
    out = gr.Textbox(visible=False)        # JSON tasiyici (gizli)
    gr.HTML(IFRAME)                         # gomulu three.js viewer

    btn.click(None, None, None, js=LOAD_JS) \
       .then(run, [prompt, guidance, seq_len], out) \
       .then(None, out, None, js=PUSH_JS)


if __name__ == "__main__":
    demo.launch(share=True, show_api=False)
