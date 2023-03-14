import gradio as gr
from diffusion_model import TextToImageModel

# Define a text to image model
model = TextToImageModel()

with gr.Blocks() as diffusion_demo:

    gr.Markdown('## Stable Diffusion 2 Demo - AI Social Bias Experiment - Predefined Prompts')
    
    with gr.Row():
        
        with gr.Column():

            mood = gr.Dropdown(choices=['a happy','a grumpy','a curious'],value='a happy',type='value',label='Mood')
            subject = gr.Dropdown(choices=['monkey','chicken','cow'],value='monkey',type='value',label='Subject')
            style = gr.Dropdown(choices=['in the style of Studio Ghibli','in the style of a polaroid photograph','in the style of film noir'],value='in the style of Studio Ghibli',type='value',label='Style')

            generate = gr.Button(label="Generate")
        
        image = gr.Image(type="numpy", label="Image")

    generate.click(model.generate_image_from_multiple, [mood,subject,style], image)

diffusion_demo.launch(share=True)