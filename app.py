"""
Gradio demosu: prompt -> model (GPU) -> gomulu three.js viewer otomatik oynatir.
Colab'da: pip install gradio + (CLIP) -> python app.py  (share=True public link verir)
  --kr  -> Korece arayuz (prompt + ornekler yine Ingilizce). Aksi halde Turkce.
Ayni dosya HF Spaces'e oldugu gibi tasinir.
"""
import os
import sys
import json
import html
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import gradio as gr
from infer import MotionGenerator

# ---------------- dil ----------------
_p = argparse.ArgumentParser()
_p.add_argument("--kr", action="store_true", help="Korece arayuz")
LANG = "kr" if _p.parse_known_args()[0].kr else "tr"

# Gradio arayuz metinleri (prompt/ornekler haric)
STR = {
    "tr": {
        "intro": "## Text-to-3D Human Motion\n"
                 "Sıfırdan eğitilmiş bir diffusion modeli. İngilizce bir prompt yaz ve "
                 "**Generate**'e bas; üretilen hareket aşağıdaki 3B sahnede oynar.",
        "prompt": "Prompt (İngilizce)",
        "generate": "Generate",
        "guidance": "Guidance (hareket prompta ne kadar sadık kalsın)",
        "length": "Uzunluk (kare)",
        "steps": "Adım sayısı (az = hızlı, çok = kaliteli)",
        "examples": "Örnekler",
    },
    "kr": {
        "intro": "## 텍스트 기반 3D 휴먼 모션 생성\n"
                 "처음부터 직접 학습시킨 diffusion 모델입니다. 영어로 프롬프트를 입력하고 "
                 "**생성** 버튼을 누르면 생성된 모션이 아래 3D 화면에서 재생됩니다.",
        "prompt": "프롬프트 (영어)",
        "generate": "생성",
        "guidance": "Guidance (모션이 프롬프트를 얼마나 충실히 따를지)",
        "length": "길이 (프레임)",
        "steps": "스텝 수 (적을수록 빠름, 많을수록 고품질)",
        "examples": "예시",
    },
}
# viewer (iframe) ic metinleri -> window.__I18N olarak enjekte edilir
VIEWER_STR = {
    "tr": {"play": "Oynat", "pause": "Durdur", "load": "JSON Yükle", "save": "JSON Kaydet",
           "hint": "Bir prompt yaz ve Generate’e bas", "generating": "üretiliyor…"},
    "kr": {"play": "재생", "pause": "일시정지", "load": "JSON 불러오기", "save": "JSON 저장",
           "hint": "프롬프트를 입력하고 생성을 누르세요", "generating": "생성 중…"},
}
T = STR[LANG]

CKPT = os.environ.get("CKPT", "motion_denoiser_h3d.pt")   # gerekirse degistir
gen = MotionGenerator(CKPT)
print("model yuklendi:", CKPT, "| device:", gen.device, "| lang:", LANG)

# --- gomulu viewer (kendi head/importmap'iyle izole calissin diye iframe srcdoc) ---
_viewer_path = os.path.join(os.path.dirname(__file__), "web", "viewer.html")
_viewer_html = open(_viewer_path, encoding="utf-8").read()
# viewer'a dil enjekte et (module script'ten ONCE calisir -> window.__I18N hazir olur)
_inject = "<script>window.__I18N = " + json.dumps(VIEWER_STR[LANG], ensure_ascii=False) + ";</script>"
_viewer_html = _viewer_html.replace("<body>", "<body>\n" + _inject, 1)
IFRAME = (
    '<iframe id="mviewer" style="width:100%;height:620px;border:0;border-radius:14px" '
    'srcdoc="' + html.escape(_viewer_html, quote=True) + '"></iframe>'
)

# parent -> iframe veri kopru (uretilen JSON'u viewer'a postMessage ile yolla)
PUSH_JS = "(j)=>{try{var d=(typeof j==='string')?JSON.parse(j):j;document.getElementById('mviewer').contentWindow.postMessage({motion:d},'*');}catch(e){console.error('push hata',e);}}"
# Generate'e basinca yukleme halkasini ac (js fn'den ONCE calisir -> girdileri aynen geri dondur)
LOAD_JS = "(...a)=>{document.getElementById('mviewer').contentWindow.postMessage({loading:true},'*'); return a;}"

CSS = "footer{display:none !important}"


def run(prompt, guidance, seq_len, steps):
    data = gen.generate(prompt.strip(), seq_len=int(seq_len),
                        guidance=float(guidance), ddim_steps=int(steps))
    return json.dumps(data)


with gr.Blocks(title="Text to 3D Human Motion", css=CSS) as demo:
    gr.Markdown(T["intro"])
    with gr.Row():
        prompt = gr.Textbox(label=T["prompt"], scale=4, value="a person walks in a circle")
        btn = gr.Button(T["generate"], variant="primary", scale=1)
    with gr.Row():
        guidance = gr.Slider(1.0, 5.0, value=2.5, step=0.5, label=T["guidance"])
        seq_len = gr.Slider(40, 196, value=120, step=4, label=T["length"])
        steps = gr.Slider(20, 100, value=50, step=5, label=T["steps"])
    gr.Examples(
        ["a person walks forward", "a person walks in a circle", "a person runs",
         "a person jumps", "a person sits down", "a person waves", "a person kicks something",
         "a person fights with someone"],
        inputs=prompt, label=T["examples"])
    out = gr.Textbox(visible=False)        # JSON tasiyici (gizli)
    gr.HTML(IFRAME)                        # gomulu three.js viewer

    # tek click: js (spinner) fn'den once -> run -> .then ile PUSH
    btn.click(run, [prompt, guidance, seq_len, steps], out, js=LOAD_JS) \
       .then(None, out, None, js=PUSH_JS)


if __name__ == "__main__":
    demo.launch(share=True, show_api=False)
