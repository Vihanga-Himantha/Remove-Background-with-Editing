# --- Imports ---
import gradio as gr
import os
import io
# **** ADDED ImageFilter and ImageEnhance ****
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
from rembg import new_session, remove
from rembg.sessions import sessions_class
import numpy as np
import traceback

# --- Font Configuration --- (Keep your FONT_MAP)
FONT_MAP = {
    "Arial": ("arial.ttf", "arialbd.ttf"),
    "Times New Roman": ("times.ttf", "timesbd.ttf"),
    "Courier New": ("cour.ttf", "courbd.ttf"),
}

# --- Default Settings ---
DEFAULT_FONT_NAME = "Arial"
DEFAULT_FONT_SIZE = 50
DEFAULT_TEXT_COLOR = "#FFFFFF"
DEFAULT_TEXT = "Your Text Here"
DEFAULT_TEXT_X_PERCENT = 50
DEFAULT_TEXT_Y_PERCENT = 50
DEFAULT_BACKGROUND_TYPE = "Transparent"
DEFAULT_ALPHA_MATTING = False
DEFAULT_MATTING_FG_THRESHOLD = 240
DEFAULT_MATTING_BG_THRESHOLD = 10
DEFAULT_MATTING_ERODE_SIZE = 10
DEFAULT_BLUR_BACKGROUND = False
DEFAULT_BLUR_RADIUS = 5
DEFAULT_ADD_TEXT_OUTLINE = False
DEFAULT_OUTLINE_COLOR = "#000000" # Black
DEFAULT_OUTLINE_WIDTH = 2
# *** NEW Defaults for Final Adjustments ***
DEFAULT_BRIGHTNESS = 1.0 # 1.0 means no change
DEFAULT_CONTRAST = 1.0   # 1.0 means no change

# --- Helper Function to Find Fonts --- (Keep as is)
def get_font_path(font_name, is_bold=False):
    if font_name not in FONT_MAP: print(f"Warn: Font '{font_name}' !in FONT_MAP."); return None
    regular_path, bold_path = FONT_MAP[font_name]
    path_to_try = bold_path if is_bold else regular_path
    if is_bold:
        try: ImageFont.truetype(bold_path, 10); print(f"DBG: Using bold: {bold_path}"); return bold_path
        except Exception as e: print(f"Warn: !Load bold '{bold_path}': {e}. Try regular."); path_to_try = regular_path
    try: ImageFont.truetype(path_to_try, 10); print(f"DBG: Using regular/fallback: {path_to_try}"); return path_to_try
    except Exception as e: print(f"ERR: !Load font '{path_to_try}': {e}. Pillow default."); return None

# *** NEW: Helper function to convert Hex to RGBA ***
def hex_to_rgba(hex_color, alpha=255):
    """Converts a hex color string (e.g., #RRGGBB) to an RGBA tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        print(f"Warn: Invalid hex color format: '{hex_color}'. Using black.")
        return (0, 0, 0, alpha) # Default to black on error
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)
    except ValueError:
        print(f"Warn: Could not parse hex color: '{hex_color}'. Using black.")
        return (0, 0, 0, alpha)


# --- Model List --- (Keep as is)
AVAILABLE_MODELS = ["u2net","u2netp","u2net_human_seg","u2net_cloth_seg","silueta","isnet-general-use","isnet-anime","sam"]
try:
    actual_models = list(sessions_class.keys()); AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m in actual_models];
    for model in actual_models:
        if model not in AVAILABLE_MODELS: AVAILABLE_MODELS.append(model)
    print("Available models:", AVAILABLE_MODELS); DEFAULT_MODEL = "isnet-general-use" if "isnet-general-use" in AVAILABLE_MODELS else AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None
except Exception as e: print(f"Warn: !Verify models: {e}."); DEFAULT_MODEL = "isnet-general-use"


# --- Core Processing Function ---
def inference_with_enhancements( # Renamed function
    # Input Image
    input_pil_image,
    # Background Options
    background_type,
    custom_background_pil_image,
    blur_background,
    blur_radius,
    # Text Options
    user_text,
    font_name,
    font_size,
    is_bold,
    text_color_hex, # Renamed to indicate hex input
    text_x_percent,
    text_y_percent,
    add_text_outline,
    outline_color_hex, # Renamed to indicate hex input
    outline_width,
    # Rembg Options
    mask_option,
    model_name,
    sam_x,
    sam_y,
    # Alpha Matting Inputs
    enable_alpha_matting,
    matting_fg_thresh,
    matting_bg_thresh,
    matting_erode_size,
    # Final Adjustments
    brightness_factor,
    contrast_factor
):
    print("--- Processing Start ---")
    # Print key params... (brightness/contrast added)
    print(f"Params: BG Type={background_type}, Blur={blur_background}({blur_radius}), Text='{user_text}', Font={font_name}, Bold={is_bold}, Size={font_size}, Color={text_color_hex}, Outline={add_text_outline}({outline_width},{outline_color_hex}), Pos=({text_x_percent}%, {text_y_percent}%)")
    print(f"Model: {model_name}, MaskOnly={mask_option=='Mask only'}, AlphaMatting={enable_alpha_matting}({matting_fg_thresh},{matting_bg_thresh},{matting_erode_size})")
    print(f"Final Adjust: Brightness={brightness_factor}, Contrast={contrast_factor}")


    # --- Input Validation --- (Keep as is)
    if input_pil_image is None: print("ERR: No input image."); return Image.new('RGBA', (100, 100), (0,0,0,0))
    if isinstance(input_pil_image, np.ndarray): input_pil_image = Image.fromarray(input_pil_image)
    elif not isinstance(input_pil_image, Image.Image):
        if isinstance(input_pil_image, str) and os.path.exists(input_pil_image): input_pil_image = Image.open(input_pil_image)
        else: print(f"ERR: Invalid input: {type(input_pil_image)}"); return Image.new('RGBA', (100, 100), (0,0,0,0))
    input_pil_image = input_pil_image.convert("RGBA"); width, height = input_pil_image.size; print(f"Input Size: {width}x{height}")

    # --- Background Removal --- (Keep as is, using rembg_kwargs)
    foreground_layer = None; mask_image = None
    rembg_kwargs = {"session": new_session(model_name), "only_mask": (mask_option == "Mask only")}
    if model_name == "sam" and sam_x is not None and sam_y is not None: rembg_kwargs["sam_prompt"] = [{"type": "point", "data": [int(sam_x), int(sam_y)], "label": 1}]
    if enable_alpha_matting and not rembg_kwargs["only_mask"]: print("Alpha matting enabled."); rembg_kwargs.update({"alpha_matting": True, "alpha_matting_foreground_threshold": int(matting_fg_thresh), "alpha_matting_background_threshold": int(matting_bg_thresh), "alpha_matting_erode_size": int(matting_erode_size)})
    elif enable_alpha_matting and rembg_kwargs["only_mask"]: print("Note: Alpha matting skipped for 'Mask only'.")
    try:
        img_byte_arr = io.BytesIO(); input_pil_image.save(img_byte_arr, format='PNG'); input_bytes = img_byte_arr.getvalue()
        print(f"Calling rembg.remove with args: {rembg_kwargs}")
        output_bytes = remove(input_bytes, **rembg_kwargs)
        if rembg_kwargs["only_mask"]: mask_image = Image.open(io.BytesIO(output_bytes)); print("Mask generated."); return mask_image
        else: foreground_layer = Image.open(io.BytesIO(output_bytes)).convert("RGBA"); print("Foreground extracted.")
    except Exception as e: print(f"ERR: rembg failed: {e}"); traceback.print_exc(); return input_pil_image

    # --- Prepare Base Background Layer --- (Keep as is)
    base_image = None; print(f"Preparing background: {background_type}")
    if background_type == "Transparent": base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    elif background_type == "Original Image": base_image = input_pil_image.copy()
    elif background_type == "Upload New":
        if custom_background_pil_image is None: print("Warn: No custom BG. Using transparent."); base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        else:
            print("Processing custom background.");
            if isinstance(custom_background_pil_image, np.ndarray): custom_background_pil_image = Image.fromarray(custom_background_pil_image)
            elif not isinstance(custom_background_pil_image, Image.Image):
                if isinstance(custom_background_pil_image, str) and os.path.exists(custom_background_pil_image): custom_background_pil_image = Image.open(custom_background_pil_image)
                else: print(f"ERR: Invalid custom BG type: {type(custom_background_pil_image)}"); base_image = Image.new('RGBA', (width, height), (255,0,0,128)); custom_background_pil_image = None
            if custom_background_pil_image: print(f"Resizing custom BG to {width}x{height}"); base_image = custom_background_pil_image.resize((width, height), Image.Resampling.LANCZOS); base_image = base_image.convert("RGBA")
    if base_image is None: print("ERR: Base image prep failed."); base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # --- Apply Background Blur (BEFORE Text) --- (Keep as is)
    if blur_background and background_type != "Transparent" and blur_radius > 0:
        print(f"Applying BG Blur (Radius: {blur_radius})");
        try: base_image = base_image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
        except Exception as e: print(f"Warn: Blur failed: {e}")

    # --- Draw Text onto Base Background ---
    if user_text:
        print(f"Drawing text: '{user_text}'")
        draw = ImageDraw.Draw(base_image)
        font = None
        try: # (Keep font loading logic)
            target_font_path = get_font_path(font_name, is_bold);
            if target_font_path: font = ImageFont.truetype(target_font_path, int(font_size))
            else: print("Using Pillow default font."); font = ImageFont.load_default()
        except Exception as e: print(f"ERR loading font: {e}. Using default."); font = ImageFont.load_default()

        # Convert Hex Colors to RGBA Tuples
        text_fill_color = hex_to_rgba(text_color_hex)
        outline_fill_color = hex_to_rgba(outline_color_hex)
        print(f"Converted Text Color: {text_fill_color}")
        if add_text_outline: print(f"Converted Outline Color: {outline_fill_color}")

        try: # (Keep text position calculation)
             bbox = draw.textbbox((0, 0), user_text, font=font); text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]
             text_x = (width * text_x_percent / 100); text_y = (height * text_y_percent / 100)
             print(f"Text BBox:{bbox}, Size={text_width}x{text_height}. Pos={text_x:.1f},{text_y:.1f}")

             # Draw Text with Optional Outline (using converted colors)
             if add_text_outline and outline_width > 0:
                 print(f"Drawing text outline (Width: {outline_width})")
                 try: # Try modern Pillow approach first
                      draw.text((text_x, text_y), user_text, fill=text_fill_color, font=font, anchor="lt", stroke_width=int(outline_width), stroke_fill=outline_fill_color)
                      print("Text drawn with stroke.")
                 except TypeError: # Fallback for older Pillow
                      print("Warn: Using offset drawing for outline."); offsets = []; ow = int(outline_width);
                      for dx in range(-ow, ow + 1):
                          for dy in range(-ow, ow + 1):
                              if dx*dx + dy*dy <= ow*ow: offsets.append((dx, dy))
                      if (0,0) in offsets: offsets.remove((0,0))
                      for dx, dy in offsets: draw.text((text_x + dx, text_y + dy), user_text, fill=outline_fill_color, font=font, anchor="lt")
                      draw.text((text_x, text_y), user_text, fill=text_fill_color, font=font, anchor="lt") # Main text on top
                      print("Text drawn with fallback outline.")
             else: # Draw text normally without outline
                 draw.text((text_x, text_y), user_text, fill=text_fill_color, font=font, anchor="lt")
                 print("Text drawn without outline.")
        except Exception as e: print(f"ERR drawing text: {e}"); traceback.print_exc()

    # --- Composite Foreground over Base --- (Keep as is)
    composited_image = None
    try:
        if base_image.mode != 'RGBA': base_image = base_image.convert('RGBA')
        if foreground_layer.mode != 'RGBA': foreground_layer = foreground_layer.convert('RGBA')
        composited_image = Image.alpha_composite(base_image, foreground_layer)
        print("Composition successful.")
    except Exception as e: print(f"ERR during compositing: {e}"); traceback.print_exc(); composited_image = base_image

    # --- Apply Final Image Adjustments --- (Keep as is)
    final_image = composited_image
    try:
        if brightness_factor != 1.0: print(f"Applying Brightness: {brightness_factor}"); enhancer = ImageEnhance.Brightness(final_image); final_image = enhancer.enhance(float(brightness_factor))
        if contrast_factor != 1.0: print(f"Applying Contrast: {contrast_factor}"); enhancer = ImageEnhance.Contrast(final_image); final_image = enhancer.enhance(float(contrast_factor))
    except Exception as e: print(f"Warn: Error applying final adjustments: {e}"); final_image = composited_image # Use pre-adjustment image on error

    print("--- Processing End ---")
    return final_image


# --- Gradio UI Elements ---
title = "Enhanced Image Editor v2"
description = "Remove BG, add styled text, use custom/blurred BGs, adjust brightness/contrast."
badge = """<div style='text-align: center; margin-top: 15px;'><a href='https://github.com/danielgatis/rembg' target='_blank'><img src='https://img.shields.io/badge/RemBG-Powered-blue' alt='RemBG Github'/></a></div>"""

# --- Gradio Helper Functions --- (Keep all helpers)
def get_coords(evt: gr.SelectData) -> tuple: return int(evt.index[0]), int(evt.index[1])
def show_coords(model: str): visible = model == "sam"; return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)
def update_background_upload_visibility(choice): return gr.update(visible=(choice == "Upload New"))
def update_alpha_matting_visibility(enabled): return gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)
def update_blur_radius_visibility(enabled): return gr.update(visible=enabled)
def update_text_outline_visibility(enabled): return gr.update(visible=enabled), gr.update(visible=enabled)

# --- Main Gradio App Definition ---
with gr.Blocks(css="footer {display: none !important}") as app:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown(description)

    with gr.Row():
        # --- Left Column (Inputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Image"); inputs_img = gr.Image(type="pil", label="Input", image_mode="RGBA", sources=["upload","webcam","clipboard"], container=True)
            gr.Markdown("### 2. BG Removal"); model_selector = gr.Dropdown(choices=AVAILABLE_MODELS, value=DEFAULT_MODEL, label="Model"); mask_option = gr.Radio(["Foreground + Text", "Mask only"], value="Foreground + Text", label="Output Type")
            with gr.Accordion("Advanced Edge Refining", open=False):
                 alpha_matting_checkbox = gr.Checkbox(label="Enable Alpha Matting", value=DEFAULT_ALPHA_MATTING)
                 with gr.Column(visible=DEFAULT_ALPHA_MATTING) as alpha_sliders_col: alpha_fg_slider = gr.Slider(0, 255, step=1, label="FG Thresh", value=DEFAULT_MATTING_FG_THRESHOLD); alpha_bg_slider = gr.Slider(0, 255, step=1, label="BG Thresh", value=DEFAULT_MATTING_BG_THRESHOLD); alpha_erode_slider = gr.Slider(0, 50, step=1, label="Erode Size", value=DEFAULT_MATTING_ERODE_SIZE)
            extra_sam = gr.Markdown("Click input image (SAM model).", visible=False); x_coord = gr.Number(label="SAM X", visible=False); y_coord = gr.Number(label="SAM Y", visible=False)

            gr.Markdown("### 3. Background Options"); bg_type_selector = gr.Radio(["Transparent", "Original Image", "Upload New"], value=DEFAULT_BACKGROUND_TYPE, label="BG Type")
            with gr.Row(visible=(DEFAULT_BACKGROUND_TYPE != 'Transparent')): bg_blur_checkbox = gr.Checkbox(label="Blur BG", value=DEFAULT_BLUR_BACKGROUND); bg_blur_slider = gr.Slider(0, 50, step=0.5, label="Blur Radius", value=DEFAULT_BLUR_RADIUS, visible=DEFAULT_BLUR_BACKGROUND)
            custom_bg_upload = gr.Image(type="pil", label="Upload Custom BG", image_mode="RGBA", sources=["upload"], visible=(DEFAULT_BACKGROUND_TYPE == "Upload New"), container=False)

            gr.Markdown("### 4. Text Options")
            with gr.Accordion("Customize Text", open=True):
                input_text = gr.Textbox(label="Text", value=DEFAULT_TEXT)
                with gr.Row(): font_selector = gr.Dropdown(list(FONT_MAP.keys()), value=DEFAULT_FONT_NAME, label="Font"); font_bold_checkbox = gr.Checkbox(label="Bold", value=False)
                with gr.Row(): input_font_size = gr.Slider(10, 400, step=1, value=DEFAULT_FONT_SIZE, label="Size"); input_text_color = gr.ColorPicker(value=DEFAULT_TEXT_COLOR, label="Color")
                with gr.Row(): text_outline_checkbox = gr.Checkbox(label="Add Outline", value=DEFAULT_ADD_TEXT_OUTLINE)
                with gr.Row(visible=DEFAULT_ADD_TEXT_OUTLINE) as text_outline_options_row: outline_color_picker = gr.ColorPicker(value=DEFAULT_OUTLINE_COLOR, label="Outline Color"); outline_width_slider = gr.Slider(1, 20, step=1, label="Outline Width", value=DEFAULT_OUTLINE_WIDTH)

                # *** CORRECTED SYNTAX: Markdown and Row on separate lines ***
                gr.Markdown("Text Position (Top-Left Corner %)")
                with gr.Row():
                    input_text_x = gr.Slider(0, 100, step=1, value=DEFAULT_TEXT_X_PERCENT, label="X %")
                    input_text_y = gr.Slider(0, 100, step=1, value=DEFAULT_TEXT_Y_PERCENT, label="Y %")

            # Final Adjustments Section
            gr.Markdown("### 5. Final Adjustments")
            with gr.Accordion("Brightness / Contrast", open=False):
                 brightness_slider = gr.Slider(minimum=0.1, maximum=3.0, step=0.05, label="Brightness", value=DEFAULT_BRIGHTNESS, info="1.0 is original")
                 contrast_slider = gr.Slider(minimum=0.1, maximum=3.0, step=0.05, label="Contrast", value=DEFAULT_CONTRAST, info="1.0 is original")

            process_button = gr.Button("Process Image", variant="primary", scale=2)

        # --- Right Column (Output) ---
        with gr.Column(scale=1): gr.Markdown("### Output Image"); outputs_img = gr.Image(type="pil", label="Output", interactive=False, image_mode="RGBA", container=True)

    # --- Interactions --- (Keep previous ones)
    model_selector.change(show_coords, inputs=model_selector, outputs=[x_coord, y_coord, extra_sam])
    inputs_img.select(get_coords, None, [x_coord, y_coord])
    bg_type_selector.change(update_background_upload_visibility, inputs=bg_type_selector, outputs=custom_bg_upload)
    alpha_matting_checkbox.change(lambda en: (gr.update(visible=en),)*3, inputs=alpha_matting_checkbox, outputs=[alpha_fg_slider, alpha_bg_slider, alpha_erode_slider])
    bg_blur_checkbox.change(update_blur_radius_visibility, inputs=bg_blur_checkbox, outputs=bg_blur_slider)
    text_outline_checkbox.change(update_text_outline_visibility, inputs=text_outline_checkbox, outputs=[outline_color_picker, outline_width_slider])

    # Define the list of inputs for the button click IN ORDER (Added brightness/contrast)
    process_inputs = [
        inputs_img,
        bg_type_selector, custom_bg_upload, bg_blur_checkbox, bg_blur_slider, # BG inputs
        input_text, font_selector, input_font_size, font_bold_checkbox, input_text_color, input_text_x, input_text_y, # Text inputs
        text_outline_checkbox, outline_color_picker, outline_width_slider, # Text enhancement inputs
        mask_option, model_selector, x_coord, y_coord, # Rembg inputs
        alpha_matting_checkbox, alpha_fg_slider, alpha_bg_slider, alpha_erode_slider, # Alpha matting inputs
        brightness_slider, contrast_slider # Final adjustment inputs
    ]

    process_button.click(
        inference_with_enhancements, # Use the renamed function
        inputs=process_inputs,
        outputs=outputs_img
    )
    gr.HTML(badge)

# --- Launch ---
if __name__ == "__main__":
    print("Verifying default font...") # (Keep font verification)
    try:
        def_font_path = get_font_path(DEFAULT_FONT_NAME, False)
        if def_font_path: ImageFont.truetype(def_font_path, 10); print(f"Default font '{DEFAULT_FONT_NAME}' loadable.")
        else: print(f"Default font '{DEFAULT_FONT_NAME}'!in map/!loadable.")
    except Exception as e: print(f"Warn: !preload font '{DEFAULT_FONT_NAME}': {e}.")
    app.launch(debug=True, share=False)