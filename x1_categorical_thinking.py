import gradio as gr

from clip_model import ImageClassifier
from os import listdir, path

# Upload a file from your computer
def upload_file(file_obj):
    return file_obj.name

IMG_DIR = 'example_images'
LOGFILE = 'logs/test_log.txt'

# Define an image classifier
model = ImageClassifier()
model.init_logfile(LOGFILE)

# Define classes for each of the sets of five example images
animal_classes = '\n'.join(['cat','dog','cow','monkey','bird'])
description_classes = '\n'.join(['a furry animal', 'an animal with claws', 'a domestic animal', 'a pet', 'a mammal'])
feelings_classes = '\n'.join(['happy', 'sad', 'angry', 'nervous', 'amused'])
colors_classes = '\n'.join(['grey', 'pink', 'black', 'white', 'orange'])

# Interface definition and deployment
with gr.Blocks() as demo:

    # Title and description
    gr.Markdown("## CLIP Image Classification Demo")
    gr.Markdown('Experiment with different image classes and see how the model\'s predictions change.')

    with gr.Row():

        with gr.Column():
            # Set default image to monkey
            image_1 = gr.Image(path.join(IMG_DIR, 'monkey.jpg'), label="Image")

            # Create upload button
            upload_button = gr.UploadButton("Upload an Image", file_types=["image"], file_count="single")

        # Create image classes viewable in tabs
        with gr.Tab(label='Animals'):
            text_classes_animals = gr.Textbox(label="Image Classes", lines = 5, value=animal_classes, interactive=True)
            submission_animals = gr.Button(value="Classify Image")
        
        with gr.Tab(label='Descriptions'):
            text_classes_descriptions = gr.Textbox(label="Image Classes", lines = 5, value=description_classes, interactive=True)
            submission_descriptions = gr.Button(value="Classify Image")
        
        with gr.Tab(label='Feelings'):
            text_classes_feelings = gr.Textbox(label="Image Classes", lines = 5, value=feelings_classes, interactive=True)
            submission_feelings = gr.Button(value="Classify Image")
        
        with gr.Tab(label='Colors'):
            text_classes_colors = gr.Textbox(label="Image Classes", lines = 5, value=colors_classes, interactive=True)
            submission_colors = gr.Button(value="Classify Image")
        
        # Use the Label component to visualize the model's predictions
        outputs = gr.Label(num_top_classes=5, label="Output")

    # Load in downloaded images to use as examples
    gr.Examples([path.join(IMG_DIR, img) for img in listdir(IMG_DIR)], image_1)

    # Define image upload button behavior
    upload_button.upload(upload_file, upload_button, image_1)

    # Run the classification function when the classify button is clicked - corresponds to each tab defined above
    submission_animals.click(model.predict, [image_1,text_classes_animals], outputs)
    submission_descriptions.click(model.predict, [image_1,text_classes_descriptions], outputs)
    submission_feelings.click(model.predict, [image_1,text_classes_feelings], outputs)
    submission_colors.click(model.predict, [image_1,text_classes_colors], outputs)

# Run the demo and create a shareable public link
demo.launch(share=True)