# Remove-Background-with-Editing

A web-based application for removing backgrounds from images using various AI models. Built with Gradio, OpenCV, and RemBG.

## Features

- Multiple AI models for background removal
- Support for different output types (Default/Mask only)
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
- birefnet variants (general, lite, portrait, etc.)

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
   pip install gradio opencv-python rembg
   ```

## Usage

1. Run the application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:7860)

3. Upload an image using the interface

4. Select your desired model and output type

5. If using the SAM model, click on the image to set coordinates

6. Click "Process Image" to remove the background

## Example Images

The application comes with example images:

- lion.png
- girl.jpg
- anime-girl.jpg

## Notes

- The first run may take some time as it downloads the required AI models
- Output images are saved as 'output.png' in the project directory
- Input images are temporarily saved as 'input.png'
