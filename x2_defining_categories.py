# Import libraries
import gradio as gr

from clip_model import ImageClassifier, log_json
from os import listdir, path

from utils import upload_file_by_station

# Set image directories for example images and storing new images
NEW_IMG_DIR = 'user_images'
IMG_DIR = 'example_images/boundary'

# List of example images
imgs = sorted([img for img in listdir(IMG_DIR) if img not in ['.DS_Store']], key=lambda x: int(x.split('.')[0]))

# Define an image classifier
model = ImageClassifier()

# Define a function that takes an image and text classes, logs the state of the UI and model output, and returns the model's predictions
def collect_classes_and_predict(image, text_1, text_2, text_3, text_4, text_5, station, user, session_ID, experiment, image_ID, textID, logfile=None):
    
    dict_probs, log_dict = model.predict(image, [text_1, text_2, text_3, text_4, text_5])
    log_json(station, user, session_ID, experiment, image_ID, textID, log_dict, logfile)

    return dict_probs

# Interface definition and deployment
with gr.Blocks() as demo:

    header = gr.Markdown("## UW Kids Team - Defining Categories")

    with gr.Row():

        with gr.Column():

            image_1 = gr.Image(path.join(IMG_DIR, imgs[0]), label="Image")

            # Create upload button
            upload_button = gr.UploadButton("Upload an Image", file_types=["image"], file_count="single")

            with gr.Accordion(label='Logging Info', open=False):

                station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                name = gr.Textbox(label="User Name", lines=1, interactive=True)
                session = gr.Dropdown(label="Session ID", value='S1', choices=['S1','S2','S3'], interactive=False, visible=False)
                experiment = gr.Dropdown(label="Experiment ID", value='X1', choices=['X1','X2','X3'], interactive=False, visible=False)
                image_id = gr.Textbox(label="Image ID", value=imgs[0], interactive=False, visible=False)
                user_image_dir = gr.Textbox(label="User Image Directory", value=NEW_IMG_DIR, interactive=False, visible=False)

        with gr.Column():

            class_header = gr.Markdown("Use 1-3 words to describe what you see in the pictures and click 'Predict'")

            text_id = gr.Textbox(label="Image ID", value='user_defined', interactive=False, visible=False)
            text_1 = gr.Textbox(label="Description 1", lines=1)
            text_2 = gr.Textbox(label="Description 2", lines=1)
            text_3 = gr.Textbox(label="Description 3", lines=1)
            text_4 = gr.Textbox(label="Description 4", lines=1)
            text_5 = gr.Textbox(label="Description 5", lines=1)

            submission_button = gr.Button('Predict!')

        with gr.Column():
            outputs = gr.Label(num_top_classes=5, label="Output")

            with gr.Accordion(label="Images", open=False):
                examples = gr.Examples([[path.join(IMG_DIR, img), img] for img in imgs], [image_1, image_id], None, examples_per_page=3)    

    submission_button.click(collect_classes_and_predict, [image_1, text_1, text_2, text_3, text_4, text_5, station, name, session, experiment, image_id, text_id], outputs, show_progress=False)

    # Define image upload button behavior
    upload_button.upload(upload_file_by_station, [upload_button, user_image_dir, station, experiment], [image_1, image_id])

demo.launch(share=True)