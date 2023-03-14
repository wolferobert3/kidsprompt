import gradio as gr
from diffusion_model import TextToImageModel

# Define a text to image model
model = TextToImageModel()

with gr.Blocks() as diffusion_demo:

    gr.Markdown('## Stable Diffusion 2 Demo - AI Social Bias Experiment - User-Defined Prompts')
    
    with gr.Row():
        
        with gr.Column():
            prompt1 = gr.Textbox(lines=1, label="Prompt 1")
            prompt2 = gr.Textbox(lines=1, label="Prompt 2")
            prompt3 = gr.Textbox(lines=1, label="Prompt 3")

            generate = gr.Button(label="Generate")
        
        image = gr.Image(type="numpy", label="Image")

    generate.click(model.generate_image_from_multiple, [prompt1,prompt2,prompt3], image)

diffusion_demo.launch(share=True)