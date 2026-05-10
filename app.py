import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont
from skimage.filters import threshold_local

# ---------------------------------------------------
# Configuration & Asset Prep
# ---------------------------------------------------

ASCII_PALETTES = {
    "complex": np.array(list(" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")),
    "standard": np.array(list(" .:-=+*#%@")),
    "blocks": np.array(list(" ░▒▓█")),
}

def build_density_map(chars):
    """Calculates actual visual weight of characters for better mapping."""
    try:
        font = ImageFont.load_default()
    except:
        font = None # Fallback for environments without default fonts
    
    densities = []
    for ch in chars:
        img = Image.new("L", (16, 16), 255)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), ch, fill=0, font=font)
        densities.append(1.0 - (np.array(img).mean() / 255.0))
    return np.array(densities)

DENSITY_CACHE = {name: build_density_map(chars) for name, chars in ASCII_PALETTES.items()}

# ---------------------------------------------------
# Core Research Engine
# ---------------------------------------------------

class AdvancedASCIIPipeline:
    @staticmethod
    def analyze_and_process(image, width, palette_key, edge_strength, gamma, contrast, adaptive_block=35):
        # 1. Grayscale & Contrast Boost
        gray = image.convert("L")
        
        # 2. Adaptive Thresholding (from Research Pipeline)
        # Helps isolate hardware components from shadows
        gray_np = np.array(gray)
        adaptive_thresh = threshold_local(gray_np, adaptive_block, offset=10)
        binary_mask = (gray_np > adaptive_thresh).astype(float)

        # 3. Structural Edge Detection
        edges = gray.filter(ImageFilter.GaussianBlur(1)).filter(ImageFilter.FIND_EDGES)
        edges = ImageEnhance.Contrast(edges).enhance(2.0)
        edge_arr = np.array(edges).astype(float) / 255.0

        # 4. Resize with Terminal Correction (0.55 aspect)
        aspect = image.height / image.width
        new_height = max(1, int(width * aspect * 0.55))
        
        # Scale our maps
        # We convert back to PIL for high-quality LANCZOS resizing
        mask_res = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize((width, new_height), Image.Resampling.LANCZOS)
        edge_res = Image.fromarray((edge_arr * 255).astype(np.uint8)).resize((width, new_height), Image.Resampling.LANCZOS)
        
        mask_final = np.array(mask_res).astype(float) / 255.0
        edge_final = np.array(edge_res).astype(float) / 255.0

        # 5. Tone Adjustment & Combination
        # Combine the adaptive mask with the raw edges for "Structural Mapping"
        combined = np.clip((mask_final**gamma * contrast) - (edge_final * edge_strength), 0, 1)

        # 6. Perceptual Density Mapping
        palette = ASCII_PALETTES[palette_key]
        densities = DENSITY_CACHE[palette_key]
        
        # Find closest character match for every pixel
        target = 1.0 - combined
        idx = np.abs(target[..., None] - densities).argmin(axis=-1)
        
        ascii_rows = palette[idx]
        return "\n".join("".join(row) for row in ascii_rows)

# ---------------------------------------------------
# Gradio Interface Logic
# ---------------------------------------------------

def ui_wrapper(image, width, palette, edge_s, gamma, contrast):
    if image is None: return "Please upload an image."
    pipeline = AdvancedASCIIPipeline()
    return pipeline.analyze_and_process(image, width, palette, edge_s, gamma, contrast)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 Advanced Hardware ASCII Research Lab")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Hardware Input")
            
            with gr.Accordion("Fine-Tuning Controls", open=True):
                width_slider = gr.Slider(50, 250, value=120, step=1, label="Map Resolution (Width)")
                edge_slider = gr.Slider(0, 1.5, value=0.6, step=0.1, label="Structural Edge Intensity")
                gamma_slider = gr.Slider(0.1, 3.0, value=1.0, step=0.1, label="Luminance Gamma")
                contrast_slider = gr.Slider(0.5, 2.5, value=1.2, step=0.1, label="Adaptive Contrast")
                palette_choice = gr.Dropdown(choices=list(ASCII_PALETTES.keys()), value="standard", label="Character Set")
            
            btn = gr.Button("Render Structural Map", variant="primary")

        with gr.Column(scale=2):
            output = gr.Textbox(label="ASCII Output", lines=30, max_lines=100, show_copy_button=True, elem_id="ascii_out")
            # Adding CSS for monospace display
            gr.Markdown("""<style> #ascii_out textarea { font-family: 'Courier New', monospace !important; line-height: 1 !important; font-size: 8px !important; } </style>""")

    btn.click(
        fn=ui_wrapper,
        inputs=[img_input, width_slider, palette_choice, edge_slider, gamma_slider, contrast_slider],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
