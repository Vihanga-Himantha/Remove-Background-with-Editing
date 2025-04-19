import gradio as gr
import os
import io
from PIL import Image, ImageDraw, ImageFont, ImageOps # Added ImageOps for resizing background
from rembg import new_session, remove
from rembg.sessions import sessions_class
import numpy as np
import traceback # For detailed error printing

# --- Font Configuration ---
# Dictionary mapping display names to font file paths (regular, bold)
# *** YOU MUST ENSURE THESE FILES EXIST (or update paths) ***
# Common locations: Windows (C:/Windows/Fonts), Linux (/usr/share/fonts/truetype/msttcorefonts/), macOS (/Library/Fonts)
# Or place the .ttf files in the same directory as the script.
FONT_MAP = {
    "Arial": ("arial.ttf", "arialbd.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf"),
    "Courier New": ("cour.ttf", "courbd.ttf"),
    # Add more fonts here if available (Name: (regular_path, bold_path))
    # Example: "Comic Sans MS": ("comic.ttf", "comicbd.ttf")
}

# --- Default Settings ---
DEFAULT_FONT_NAME = "Arial" # Make sure this is a key in FONT_MAP
DEFAULT_FONT_SIZE = 50
DEFAULT_TEXT_COLOR = "#FFFFFF"
DEFAULT_TEXT = "Your Text Here"
DEFAULT_TEXT_X_PERCENT = 50 # Center X
DEFAULT_TEXT_Y_PERCENT = 50 # Center Y
DEFAULT_BACKGROUND_TYPE = "Transparent"
DEFAULT_ALPHA_MATTING = False # Start with alpha matting OFF
DEFAULT_MATTING_FG_THRESHOLD = 240
DEFAULT_MATTING_BG_THRESHOLD = 10
DEFAULT_MATTING_ERODE_SIZE = 10


# --- Helper Function to Find Fonts ---
def get_font_path(font_name, is_bold=False):
    """Gets the path for the requested font, handling bold variants."""
    if font_name not in FONT_MAP:
        print(f"Warning: Font '{font_name}' not in FONT_MAP. Falling back to default pillow font.")
        return None # Will trigger Pillow's default

    regular_path, bold_path = FONT_MAP[font_name]
    path_to_try = bold_path if is_bold else regular_path

    # Simple check for existence (useful for debug, Pillow handles errors)
    # Note: This requires the exact path or font name resolvable by the system/Pillow
    # if not os.path.exists(path_to_try):
    #     print(f"Debug: Font file check - Path '{path_to_try}' does not seem to exist directly.")
        # Pillow might still find it if it's a system font name like 'arial.ttf'

    # Attempt to load bold, fallback to regular if bold fails or isn't requested
    if is_bold:
        try:
            # Try loading bold directly
            ImageFont.truetype(bold_path, 10) # Test load bold
            print(f"Debug: Using bold font path: {bold_path}")
            return bold_path
        except Exception as e_bold:
            print(f"Warning: Could not load bold font '{bold_path}': {e_bold}. Trying regular.")
            try:
                # Try loading regular as fallback
                ImageFont.truetype(regular_path, 10) # Test load regular
                print(f"Debug: Using regular font path as fallback: {regular_path}")
                return regular_path
            except Exception as e_regular:
                 print(f"Error: Could not load regular font '{regular_path}' either: {e_regular}. Falling back to Pillow default.")
                 return None # Fallback to Pillow default
    else:
         # Try loading regular font
        try:
            ImageFont.truetype(regular_path, 10) # Test load regular
            print(f"Debug: Using regular font path: {regular_path}")
            return regular_path
        except Exception as e_regular:
             print(f"Error: Could not load regular font '{regular_path}': {e_regular}. Falling back to Pillow default.")
             return None # Fallback to Pillow default


# --- Model List ---
AVAILABLE_MODELS = [
    "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg",
    "silueta", "isnet-general-use", "isnet-anime", "sam"
]
try:
    actual_models = list(sessions_class.keys())
    # Ensure list only contains models rembg actually knows
    AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m in actual_models]
    # Add any other models rembg has that weren't in the initial list
    for model in actual_models:
        if model not in AVAILABLE_MODELS:
             AVAILABLE_MODELS.append(model)
    print("Available models found by rembg:", AVAILABLE_MODELS)
    # Select a default that is guaranteed to exist now
    DEFAULT_MODEL = "isnet-general-use" if "isnet-general-use" in AVAILABLE_MODELS else AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None

except Exception as e:
    print(f"Warning: Could not dynamically verify models: {e}. Using predefined list.")
    DEFAULT_MODEL = "isnet-general-use" # Fallback default


# --- Core Processing Function ---
def inference_with_text_and_background(
    # Input Image
    input_pil_image,
    # Background Options
    background_type,
    custom_background_pil_image,
    # Text Options
    user_text,
    font_name,
    font_size,
    is_bold,
    text_color,
    text_x_percent,
    text_y_percent,
    # Rembg Options
    mask_option,
    model_name,
    sam_x,
    sam_y,
    # Alpha Matting Inputs
    enable_alpha_matting,
    matting_fg_thresh,
    matting_bg_thresh,
    matting_erode_size
):
    print("--- Processing Start ---")
    print(f"Params: BG Type={background_type}, Text='{user_text}', Font={font_name}, Bold={is_bold}, Size={font_size}, Color={text_color}, Pos=({text_x_percent}%, {text_y_percent}%)")
    print(f"Model: {model_name}, MaskOnly={mask_option=='Mask only'}")
    print(f"Edge Params: AlphaMatting={enable_alpha_matting}, FG={matting_fg_thresh}, BG={matting_bg_thresh}, Erode={matting_erode_size}")

    # --- Input Validation ---
    if input_pil_image is None: print("Error: No input image."); return Image.new('RGBA', (100, 100), (0,0,0,0))
    if isinstance(input_pil_image, np.ndarray): input_pil_image = Image.fromarray(input_pil_image)
    elif not isinstance(input_pil_image, Image.Image):
        if isinstance(input_pil_image, str) and os.path.exists(input_pil_image): input_pil_image = Image.open(input_pil_image)
        else: print(f"Error: Invalid input: {type(input_pil_image)}"); return Image.new('RGBA', (100, 100), (0,0,0,0))
    input_pil_image = input_pil_image.convert("RGBA")
    width, height = input_pil_image.size
    print(f"Input Size: {width}x{height}")

    # --- Background Removal ---
    foreground_layer = None
    mask_image = None
    rembg_kwargs = {"session": new_session(model_name), "only_mask": (mask_option == "Mask only")}
    if model_name == "sam" and sam_x is not None and sam_y is not None: rembg_kwargs["sam_prompt"] = [{"type": "point", "data": [int(sam_x), int(sam_y)], "label": 1}]
    if enable_alpha_matting and not rembg_kwargs["only_mask"]:
        print("Alpha matting enabled.")
        rembg_kwargs.update({
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": int(matting_fg_thresh),
            "alpha_matting_background_threshold": int(matting_bg_thresh),
            "alpha_matting_erode_size": int(matting_erode_size)
        })
    elif enable_alpha_matting and rembg_kwargs["only_mask"]: print("Note: Alpha matting skipped for 'Mask only'.")

    try:
        img_byte_arr = io.BytesIO(); input_pil_image.save(img_byte_arr, format='PNG'); input_bytes = img_byte_arr.getvalue()
        print(f"Calling rembg.remove with args: {rembg_kwargs}")
        output_bytes = remove(input_bytes, **rembg_kwargs)
        if rembg_kwargs["only_mask"]: mask_image = Image.open(io.BytesIO(output_bytes)); print("Mask generated."); return mask_image
        else: foreground_layer = Image.open(io.BytesIO(output_bytes)).convert("RGBA"); print("Foreground extracted.")
    except Exception as e:
        print(f"Error during rembg: {e}"); traceback.print_exc()
        error_img = input_pil_image.copy(); draw = ImageDraw.Draw(error_img)
        try: font_err = ImageFont.load_default(); draw.text((10, 10), f"BG Removal Failed: {e}", fill="red", font=font_err)
        except Exception: pass
        return error_img # Return error state

    # --- Prepare Base Background Layer ---
    base_image = None
    print(f"Preparing background: {background_type}")
    if background_type == "Transparent": base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    elif background_type == "Original Image": base_image = input_pil_image.copy()
    elif background_type == "Upload New":
        if custom_background_pil_image is None: print("Warning: No custom BG uploaded. Using transparent."); base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        else:
            print("Processing custom background.")
            if isinstance(custom_background_pil_image, np.ndarray): custom_background_pil_image = Image.fromarray(custom_background_pil_image)
            elif not isinstance(custom_background_pil_image, Image.Image):
                 if isinstance(custom_background_pil_image, str) and os.path.exists(custom_background_pil_image): custom_background_pil_image = Image.open(custom_background_pil_image)
                 else: print(f"Error: Invalid custom BG type: {type(custom_background_pil_image)}"); base_image = Image.new('RGBA', (width, height), (255,0,0,128)); custom_background_pil_image = None
            if custom_background_pil_image:
                print(f"Resizing custom background to {width}x{height}"); base_image = custom_background_pil_image.resize((width, height), Image.Resampling.LANCZOS); base_image = base_image.convert("RGBA")
    if base_image is None: print("Error: Base image prep failed."); base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0)) # Failsafe


    # --- Draw Text onto Base Background ---
    if user_text:
        print(f"Drawing text: '{user_text}'")
        draw = ImageDraw.Draw(base_image)
        font = None
        try:
            target_font_path = get_font_path(font_name, is_bold) # Use helper to get path
            if target_font_path:
                font = ImageFont.truetype(target_font_path, int(font_size))
            else: # Fallback if helper returned None
                print("Using Pillow's default bitmap font.")
                font = ImageFont.load_default()
        except IOError as e: print(f"IOError loading font '{target_font_path}': {e}. Using default."); font = ImageFont.load_default()
        except Exception as e: print(f"Error loading font size {font_size}: {e}. Using default."); font = ImageFont.load_default()
        try:
             bbox = draw.textbbox((0, 0), user_text, font=font); text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]
             text_x = (width * text_x_percent / 100); text_y = (height * text_y_percent / 100)
             print(f"Text BBox: {bbox}, Size={text_width}x{text_height}. Pos={text_x},{text_y}")
             # Use anchor='lt' (left, top) so (X%, Y%) refers to the top-left corner of text bbox
             draw.text((text_x, text_y), user_text, fill=text_color, font=font, anchor="lt")
             # If anchor fails on older Pillow, text might be slightly offset depending on font metrics
             print("Text drawn.")
        except Exception as e: print(f"Error drawing text: {e}"); traceback.print_exc()


    # --- Composite Foreground over Base ---
    final_image = None
    try:
        if base_image.mode != 'RGBA': base_image = base_image.convert('RGBA')
        if foreground_layer.mode != 'RGBA': foreground_layer = foreground_layer.convert('RGBA')
        final_image = Image.alpha_composite(base_image, foreground_layer)
        print("Final composition successful.")
    except Exception as e:
        print(f"Error during final compositing: {e}"); traceback.print_exc()
        final_image = base_image.copy(); draw = ImageDraw.Draw(final_image) # Fallback
        try: font_err = ImageFont.load_default(); draw.text((10, height - 20), f"Compositing Failed: {e}", fill="red", font=font_err)
        except Exception: pass

    print("--- Processing End ---")
    return final_image


# --- Gradio UI Elements ---
title = "Advanced Image Editor (BG Removal + Text)"
description = """
1. Upload image.
2. Choose BG removal model & options. **Try 'Advanced Edge Refining' for cleaner edges.**
3. Select background type. Upload if needed.
4. Customize text style & position.
5. Click 'Process Image'.
"""
badge = """<div style='text-align: center; margin-top: 15px;'><a href='https://github.com/danielgatis/rembg' target='_blank'><img src='https://img.shields.io/badge/RemBG-Powered-blue' alt='RemBG Github'/></a></div>"""

def get_coords(evt: gr.SelectData) -> tuple: return int(evt.index[0]), int(evt.index[1])
def show_coords(model: str): visible = model == "sam"; return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)
def update_background_upload_visibility(choice): return gr.update(visible=(choice == "Upload New"))
def update_alpha_matting_visibility(enabled): return gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)


# --- Main Gradio App Definition ---
with gr.Blocks(css="footer {display: none !important}") as app:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)

    with gr.Row():
        # --- Left Column (Inputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Image")
            inputs_img = gr.Image(type="pil", label="Input Image", image_mode="RGBA", sources=["upload", "webcam", "clipboard"], container=True)

            gr.Markdown("### 2. Background Removal")
            model_selector = gr.Dropdown(choices=AVAILABLE_MODELS, value=DEFAULT_MODEL, label="Model", info="Select BG removal algorithm.")
            mask_option = gr.Radio(["Foreground + Text", "Mask only"], value="Foreground + Text", label="Output Type", info="Choose 'Mask only' to see the removal area.")

            # --- ADVANCED EDGE REFINING SECTION ---
            with gr.Accordion("Advanced Edge Refining", open=False): # Collapsed by default
                 alpha_matting_checkbox = gr.Checkbox(label="Enable Alpha Matting (Improves Edges, Slower)", value=DEFAULT_ALPHA_MATTING)
                 with gr.Column(visible=DEFAULT_ALPHA_MATTING) as alpha_sliders_col: # Group sliders
                      alpha_fg_slider = gr.Slider(minimum=0, maximum=255, step=1, label="Foreground Threshold", info="Lower values trim foreground more. Default: 240", value=DEFAULT_MATTING_FG_THRESHOLD)
                      alpha_bg_slider = gr.Slider(minimum=0, maximum=255, step=1, label="Background Threshold", info="Higher values trim background more. Default: 10", value=DEFAULT_MATTING_BG_THRESHOLD)
                      alpha_erode_slider = gr.Slider(minimum=0, maximum=50, step=1, label="Erode Size", info="Larger values smooth/shrink edges more. Default: 10", value=DEFAULT_MATTING_ERODE_SIZE)

            # SAM Coordinates (conditionally visible)
            extra_sam = gr.Markdown("Click input image to select subject (for SAM model).", visible=False)
            x_coord = gr.Number(label="SAM X", visible=False)
            y_coord = gr.Number(label="SAM Y", visible=False)


            gr.Markdown("### 3. Background Options")
            bg_type_selector = gr.Radio(choices=["Transparent", "Original Image", "Upload New"], value=DEFAULT_BACKGROUND_TYPE, label="Background Type")
            custom_bg_upload = gr.Image(type="pil", label="Upload Custom Background", image_mode="RGBA", sources=["upload"], visible=(DEFAULT_BACKGROUND_TYPE == "Upload New"), container=False)


            gr.Markdown("### 4. Text Options")
            with gr.Accordion("Customize Text", open=True):
                input_text = gr.Textbox(label="Text", value=DEFAULT_TEXT)
                with gr.Row():
                    font_selector = gr.Dropdown(choices=list(FONT_MAP.keys()), value=DEFAULT_FONT_NAME, label="Font")
                    font_bold_checkbox = gr.Checkbox(label="Bold", value=False)
                with gr.Row():
                    input_font_size = gr.Slider(minimum=10, maximum=400, step=1, value=DEFAULT_FONT_SIZE, label="Size")
                    input_text_color = gr.ColorPicker(value=DEFAULT_TEXT_COLOR, label="Color")
                gr.Markdown("Text Position (Top-Left Corner %)")
                with gr.Row():
                     input_text_x = gr.Slider(minimum=0, maximum=100, step=1, value=DEFAULT_TEXT_X_PERCENT, label="X %")
                     input_text_y = gr.Slider(minimum=0, maximum=100, step=1, value=DEFAULT_TEXT_Y_PERCENT, label="Y %")

            process_button = gr.Button("Process Image", variant="primary", scale=2)


        # --- Right Column (Output) ---
        with gr.Column(scale=1):
            gr.Markdown("### Output Image")
            outputs_img = gr.Image(type="pil", label="Output", interactive=False, image_mode="RGBA", container=True)

    # --- Interactions ---
    model_selector.change(show_coords, inputs=model_selector, outputs=[x_coord, y_coord, extra_sam])
    inputs_img.select(get_coords, None, [x_coord, y_coord])
    bg_type_selector.change(update_background_upload_visibility, inputs=bg_type_selector, outputs=custom_bg_upload)

    # Interaction for Alpha Matting Checkbox to show/hide sliders
    # Using a lambda to unpack the single column visibility update to the individual sliders
    # (Alternative: update the column's visibility directly if Gradio version supports it well)
    alpha_matting_checkbox.change(
        lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)),
        inputs=alpha_matting_checkbox,
        outputs=[alpha_fg_slider, alpha_bg_slider, alpha_erode_slider]
    )
    # Simpler alternative if updating column visibility works reliably:
    # alpha_matting_checkbox.change(lambda enabled: gr.update(visible=enabled), inputs=alpha_matting_checkbox, outputs=alpha_sliders_col)


    # Define the list of inputs for the button click IN ORDER
    process_inputs = [
        inputs_img,
        bg_type_selector, custom_bg_upload,
        input_text, font_selector, input_font_size, font_bold_checkbox, input_text_color, input_text_x, input_text_y,
        mask_option, model_selector, x_coord, y_coord,
        alpha_matting_checkbox, alpha_fg_slider, alpha_bg_slider, alpha_erode_slider # Ensure these are last and in order
    ]

    process_button.click(
        inference_with_text_and_background,
        inputs=process_inputs,
        outputs=outputs_img
    )

    # --- Examples --- (Removed for clarity, update if needed)
    # gr.Examples(...)

    gr.HTML(badge)

# --- Launch ---
if __name__ == "__main__":
    print("Verifying default font...")
    try:
        def_font_path = get_font_path(DEFAULT_FONT_NAME, False) # Test regular
        # Pillow raises IOError if font file not found, but might find system font by name
        # A simple load test is better than just checking path
        if def_font_path: # Only test if we got a path/name from our map
            ImageFont.truetype(def_font_path, 10)
            print(f"Default font '{DEFAULT_FONT_NAME}' seems loadable via path/name: {def_font_path}")
        else:
             print(f"Default font '{DEFAULT_FONT_NAME}' not found in map. Pillow default will be used.")
    except Exception as e:
        print(f"Warning: Could not preload/verify default font '{DEFAULT_FONT_NAME}': {e}. Pillow default will be used.")

    app.launch(debug=True, share=False)