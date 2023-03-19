import gradio as gr

from clip_model import ImageClassifier, log_json
from os import listdir, path

# Upload a file from your computer
def upload_file(file_obj):
    return file_obj.name, path.split(file_obj.name)[1]

IMG_DIR = 'example_images'

# Define an image classifier
model = ImageClassifier()

# Define classes for each of the sets of five example images
animal_classes = '\n'.join(['cat','dog','cow','monkey','bird'])
description_classes = '\n'.join(['a furry animal', 'an animal with claws', 'a domestic animal', 'a pet', 'a mammal'])
feelings_classes = '\n'.join(['happy', 'sad', 'angry', 'nervous', 'amused'])
colors_classes = '\n'.join(['grey', 'pink', 'black', 'white', 'orange'])

def predict_and_log(image, text_classes, station, user, session_ID, experiment, image_ID, textID, logfile=None):

    dict_probs, log_dict = model.predict(image, text_classes)
    log_json(station, user, session_ID, experiment, image_ID, textID, log_dict, logfile)

    return dict_probs

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
            text_id_animal = gr.Textbox(label="Image ID", value='animals', interactive=False, visible=False)
            text_classes_animals = gr.Textbox(label="Image Classes", lines = 5, value=animal_classes, interactive=True)
            submission_animals = gr.Button(value="Classify Image")
        
        with gr.Tab(label='Descriptions'):
            text_id_description = gr.Textbox(label="Image ID", value='descriptions', interactive=False, visible=False)
            text_classes_descriptions = gr.Textbox(label="Image Classes", lines = 5, value=description_classes, interactive=True)
            submission_descriptions = gr.Button(value="Classify Image")
        
        with gr.Tab(label='Feelings'):
            text_id_feelings = gr.Textbox(label="Image ID", value='feelings', interactive=False, visible=False)
            text_classes_feelings = gr.Textbox(label="Image Classes", lines = 5, value=feelings_classes, interactive=True)
            submission_feelings = gr.Button(value="Classify Image")
        
        with gr.Tab(label='Colors'):
            text_id_colors = gr.Textbox(label="Image ID", value='colors', interactive=False, visible=False)
            text_classes_colors = gr.Textbox(label="Image Classes", lines = 5, value=colors_classes, interactive=True)
            submission_colors = gr.Button(value="Classify Image")
        
        # Use the Label component to visualize the model's predictions
        outputs = gr.Label(num_top_classes=5, label="Output")

    with gr.Row():
        with gr.Column():
            with gr.Accordion(label='Logging Info', open=False):
                station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                name = gr.Textbox(label="User Name", lines=1, interactive=True)
                session = gr.Dropdown(label="Session ID", value='S1', choices=['S1','S2','S3'], interactive=True, visible=False)
                experiment = gr.Dropdown(label="Experiment ID", value='X1', choices=['X1','X2','X3'], interactive=True, visible=False)
                image_id = gr.Textbox(label="Image ID", value='monkey.jpg', interactive=True, visible=False)
                log_file = gr.Textbox(label="Log File", interactive=True, visible=False)

        with gr.Column():
            # Load in downloaded images to use as examples
            with gr.Accordion(label="Images", open=False):
                examples = gr.Examples([[path.join(IMG_DIR, img), img] for img in listdir(IMG_DIR) if img not in ['.DS_Store']], [image_1, image_id], None, examples_per_page=2)    

    # Define image upload button behavior
    upload_button.upload(upload_file, upload_button, [image_1, image_id])

    # Run the classification function when the classify button is clicked - corresponds to each tab defined above
    submission_animals.click(predict_and_log, [image_1, text_classes_animals, station, name, session, experiment, image_id, text_id_animal], outputs)
    submission_descriptions.click(predict_and_log, [image_1, text_classes_descriptions, station, name, session, experiment, image_id, text_id_description], outputs)
    submission_feelings.click(predict_and_log, [image_1, text_classes_feelings, station, name, session, experiment, image_id, text_id_feelings], outputs)
    submission_colors.click(predict_and_log, [image_1, text_classes_colors, station, name, session, experiment, image_id, text_id_colors], outputs)

# Run the demo and create a shareable public link
demo.launch(share=True)