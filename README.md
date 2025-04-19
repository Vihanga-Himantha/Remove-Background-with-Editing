# Remove-Background-with-Editing

A web-based application for removing backgrounds from images and adding customized text overlays. Built with Gradio, OpenCV, and RemBG.

## Features

- Multiple AI models for background removal
- Text overlay with customizable options:
  - Multiple font choices (Arial, Times New Roman, Courier New)
  - Font size and color customization
  - Bold text support
  - Precise text positioning using percentage coordinates
- Background options:
  - Transparent background
  - Keep original background
  - Upload custom background
- Advanced edge refining with Alpha Matting
- Support for different output types (Foreground + Text/Mask only)
- Interactive web interface
- Special SAM model support with coordinate selection
- Support for various image formats

## Models Available

- u2net
- u2netp
- u2net_human_seg
- u2net_cloth_seg
- silueta
- isnet-general-use
- isnet-anime
- sam

## Setup Instructions

1. Create a virtual environment:

   ```bash
   python -m venv myenv
   ```

2. Activate the virtual environment:

   - Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source myenv/bin/activate
     ```

3. Install required dependencies:
   ```bash
   pip install gradio opencv-python rembg Pillow
   ```

## Usage

1. Run the application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://127.0.0.1:7860)

3. Upload an image using the interface

4. Configure background removal:

   - Select your desired model
   - Choose output type (Foreground + Text/Mask only)
   - Enable Advanced Edge Refining if needed
   - If using the SAM model, click on the image to set coordinates

5. Choose background options:

   - Transparent background
   - Keep original image
   - Upload a custom background

6. Customize text (optional):

   - Enter your text
   - Select font, size, and color
   - Toggle bold option
   - Adjust text position using X% and Y% sliders

7. Click "Process Image" to generate the final image

## Notes

- The first run may take some time as it downloads the required AI models
- Font availability depends on your system fonts
- Advanced Edge Refining (Alpha Matting) provides cleaner edges but increases processing time
- Text positioning is based on the top-left corner of the text box
- The interface provides real-time preview of your settings
- All processing is done locally on your machine
