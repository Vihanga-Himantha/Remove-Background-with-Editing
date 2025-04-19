import gradio as gr
import os
import cv2
from rembg import new_session, remove
from rembg.sessions import sessions_class

def inference(file, mask, model, x, y):
    im = cv2.imread(file, cv2.IMREAD_COLOR)
    input_path = "input.png"
    output_path = "output.png"
    cv2.imwrite(input_path, im)

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            session = new_session(model)

            output = remove(
                input,
                session=session,
                **{ "sam_prompt": [{"type": "point", "data": [x, y], "label": 1}] },
                only_mask=(mask == "Mask only")
            )
            o.write(output)

    return output_path

title = "RemBG"
description = "Gradio demo for **[RemBG](https://github.com/danielgatis/rembg)**. To use it, simply upload your image, select a model, click Process, and wait."
badge = """
    <div style="position: fixed; left: 50%; text-align: center;">
        <a href="https://github.com/danielgatis/rembg" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/RemBG-Github-blue" alt="RemBG Github" />
        </a>
    </div>
"""
def get_coords(evt: gr.SelectData) -> tuple:
    return evt.index[0], evt.index[1]

def show_coords(model: str):
    visible = model == "sam"
    return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)

for session in sessions_class:
    session.download_models()

with gr.Blocks() as app:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        inputs = gr.Image(type="filepath", label="Input Image")
        outputs = gr.Image(type="filepath", label="Output Image")
        
    with gr.Row():
        mask_option = gr.Radio(
            ["Default", "Mask only"],
            value="Default",
            label="Output Type"
        )
        model_selector = gr.Dropdown(
            [
                "u2net",
                "u2netp",
                "u2net_human_seg",
                "u2net_cloth_seg",
                "silueta",
                "isnet-general-use",
                "isnet-anime",
                "sam",
                "birefnet-general",
                "birefnet-general-lite",
                "birefnet-portrait",
                "birefnet-dis",
                "birefnet-hrsod",
                "birefnet-cod",
                "birefnet-massive"
            ],
            value="isnet-general-use",
            label="Model Selection"
        )

    extra = gr.Markdown("## Click on the image to capture coordinates (for SAM model)", visible=False)

    x = gr.Number(label="Mouse X Coordinate", visible=False)
    y = gr.Number(label="Mouse Y Coordinate", visible=False)

    model_selector.change(show_coords, inputs=model_selector, outputs=[x, y, extra])
    inputs.select(get_coords, None, [x, y])


    gr.Button("Process Image").click(
        inference,
        inputs=[inputs, mask_option, model_selector, x, y],
        outputs=outputs
    )

    gr.Examples(
        examples=[
            ["lion.png", "Default", "u2net", None, None],
            ["girl.jpg", "Default", "u2net", None, None],
            ["anime-girl.jpg", "Default", "isnet-anime", None, None]
        ],
        inputs=[inputs, mask_option, model_selector, x, y],
        outputs=outputs
    )
    gr.HTML(badge)

app.launch()
