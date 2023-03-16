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

    gr.Markdown('## Image Generation - AI Social Bias - Predefined Prompts')
    
    with gr.Row():
        
        with gr.Column():

            mood = gr.Dropdown(choices=['a happy','a grumpy','a curious'],value='a happy',type='value',label='Mood')
            subject = gr.Dropdown(choices=['monkey','chicken','cow'],value='monkey',type='value',label='Subject')
            style = gr.Dropdown(choices=['in the style of Studio Ghibli','in the style of a polaroid photograph','in the style of film noir'],value='in the style of Studio Ghibli',type='value',label='Style')

            generate = gr.Button(label="Generate")
        
        image = gr.Image(type="numpy", label="Image")

    generate.click(generate_image_from_multiple, [mood,subject,style], image)

diffusion_demo.launch(share=True)