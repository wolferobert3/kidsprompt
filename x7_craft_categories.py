# Import libraries
import datetime
import gradio as gr

from clip_model import ImageClassifier
from utils import log_json, save_log_image

# Set image directories for example images and storing new images
NEW_IMG_DIR = 'user_images'

# Define an image classifier
model = ImageClassifier()

# Define a function that takes an image and text classes, logs the state of the UI and model output, and returns the model's predictions
def collect_classes_and_predict(image, text_1, text_2, text_3, station, user, session_ID, experiment, image_ID, textID, logfile=None):
    
    if sum([bool(i) for i in [text_1, text_2, text_3]]) < 2:
        return None

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dict_probs, log_dict = model.predict(image, [text_1, text_2, text_3])
    log_json(station, user, session_ID, experiment, image_ID, textID, log_dict, logfile)
    save_log_image(image, station, experiment, date_time, NEW_IMG_DIR, fromarray=True)

    return dict_probs

# Interface definition and deployment
with gr.Blocks() as demo:

    header = gr.Markdown("## UW Kids Team - Defining Craft Categories")

    with gr.Row():

        with gr.Column():

            image_1 = gr.Image(source='webcam', streaming=True)

            with gr.Accordion(label='Logging Info', open=False):

                station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                name = gr.Textbox(label="User Name", lines=1, interactive=True)
                session = gr.Dropdown(label="Session ID", value='S1', choices=['S1','S2','S3'], interactive=False, visible=False)
                experiment = gr.Dropdown(label="Experiment ID", value='X7', choices=['X1','X2','X3','X7'], interactive=False, visible=False)
                image_id = gr.Textbox(label="Image ID", value='', interactive=False, visible=False)
                user_image_dir = gr.Textbox(label="User Image Directory", value=NEW_IMG_DIR, interactive=False, visible=False)

        with gr.Column():

            class_header = gr.Markdown("Use 1-3 words to describe what you see in the pictures and click 'Predict'")

            text_id = gr.Textbox(label="Image ID", value='user_defined', interactive=False, visible=False)
            text_1 = gr.Textbox(label="Description 1", lines=1)
            text_2 = gr.Textbox(label="Description 2", lines=1)
            text_3 = gr.Textbox(label="Description 3", lines=1)

            submission_button = gr.Button('Predict!')

        with gr.Column():
            outputs = gr.Label(num_top_classes=3, label="Output")

    submission_button.click(collect_classes_and_predict, [image_1, text_1, text_2, text_3, station, name, session, experiment, image_id, text_id], outputs, show_progress=False)

demo.launch(share=True)