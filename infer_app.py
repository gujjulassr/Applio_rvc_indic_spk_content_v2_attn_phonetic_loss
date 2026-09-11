import os, uuid, gradio as gr
from core import run_infer_script

os.makedirs("infer_outputs", exist_ok=True)


def walk(ext, skip=()):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk("logs")
                  for f in fs if f.endswith(ext) and not f.startswith(skip))


def convert(model, index, audio, pitch, sid, index_rate):
    if not (model and audio):
        raise gr.Error("Select a checkpoint and an audio clip.")
    out = f"infer_outputs/{uuid.uuid4().hex}.wav"
    _, path = run_infer_script(
        pitch=int(pitch), index_rate=float(index_rate), volume_envelope=1.0, protect=0.33,
        f0_method="rmvpe", input_path=audio, output_path=out, pth_path=model,
        index_path=index or "", split_audio=False, f0_autotune=False,
        f0_autotune_strength=1.0, proposed_pitch=False, proposed_pitch_threshold=155.0,
        clean_audio=False, clean_strength=0.5, export_format="WAV",
        embedder_model="contentvec", sid=int(sid))
    return path


with gr.Blocks(title="Applio Inference") as demo:
    gr.Markdown("### Applio Inference")
    model = gr.Dropdown(walk(".pth", ("G_", "D_")), label="Checkpoint (.pth)")
    index = gr.Dropdown([""] + walk(".index"), value="", label="Index (.index)")
    audio = gr.Audio(type="filepath", label="Audio")
    with gr.Row():
        pitch = gr.Slider(-12, 12, 0, step=1, label="Pitch")
        sid = gr.Number(0, precision=0, label="Speaker ID")
        index_rate = gr.Slider(0, 1, 0.5, label="Index rate")
    out = gr.Audio(label="Output")
    gr.Button("Convert", variant="primary").click(
        convert, [model, index, audio, pitch, sid, index_rate], out)

demo.launch(server_name="0.0.0.0", server_port=7860)
