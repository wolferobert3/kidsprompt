import gradio as gr

from PIL import Image
from dalle_model import TextToImageModel

# Define a text to image model
model = TextToImageModel()

def generate_image_from_multiple(mood,subject,style):

    prompt = f'Generate an image of {mood} {subject} {style}.'
    model.generate_and_retrieve_images(prompt, n=1, size="256x256")
    image = Image.open(f'./{prompt}_0.png')

    return image

with gr.Blocks() as diffusion_demo:

    gr.Markdown('## DALL-E 2 Demo - AI Social Bias - User-Defined Prompts')
    
    with gr.Row():
        
        with gr.Column():

            prompt1 = gr.Textbox(lines=1, label="Prompt 1")
            prompt2 = gr.Textbox(lines=1, label="Prompt 2")
            prompt3 = gr.Textbox(lines=1, label="Prompt 3")

            generate = gr.Button(label="Generate")
        
        image = gr.Image(type="numpy", label="Image")

    generate.click(generate_image_from_multiple, [prompt1,prompt2,prompt3], image)

diffusion_demo.launch(share=True)