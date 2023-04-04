import gradio as gr

from os import path, listdir

from clip_model import ImageClassifier
from utils import log_json_resources, upload_file_by_station

# Define parameters
NEW_IMG_DIR = 'user_images'
IMG_DIR = 'example_images/pets'
LOGFILE = 'logs/test_log.txt'

# List of example images
imgs = sorted([img for img in listdir(IMG_DIR) if img not in ['.DS_Store']], key=lambda x: int(x.split('.')[0]))

# Define an image classifier
model = ImageClassifier()

# Set model parameters
model.init_logfile(LOGFILE)

IMAGE_DICT = {'positive': 'admission_images/door_open.jpg', 'negative': 'admission_images/door_close.jpg'}#, 'maybe a monster': 'admission_images/question_mark.png'}

def collect_classes_and_predict(image, positive_text, negative_text, decision_boundary, station, user, session_ID, experiment, image_ID, textID, logfile=None):
    
    if not positive_text or not negative_text:
        return None, f'admission_images/background_starting.jpg'

    class_map = {positive_text: 'positive', negative_text: 'negative'}
    text_classes = [positive_text, negative_text]

    dict_probs, log_dict = model.predict(image, text_classes)

    if dict_probs[positive_text]*100 > decision_boundary:
        output_image = IMAGE_DICT[class_map[positive_text]]
    else:
        output_image = IMAGE_DICT[class_map[negative_text]]

    log_json_resources(station, user, session_ID, experiment, image_ID, textID, log_dict, decision_boundary, logfile)

    return dict_probs, output_image

# Interface definition and deployment
with gr.Blocks() as demo:

    admit = gr.Markdown('### UW Kids Team: Designing Buddy\'s Smart Door')

    with gr.Accordion(label="Smart Door Camera", open=False):

        with gr.Row():

            with gr.Column():

                image_1 = gr.Image(path.join(IMG_DIR, imgs[0]), label="Image")
                upload_button = gr.UploadButton("Upload an Image", file_types=["image"], file_count="single")
                image_id = gr.Textbox(label="Image ID", value=imgs[0], interactive=False, visible=False)

                positive_text = gr.Textbox(label="Describe Buddy's Friend Group", interactive=True, visible=True)
                negative_text = gr.Textbox(label="Describe Animals Outside of Buddy's Friend Group", interactive=True, visible=True)
                text_id = gr.Textbox(label="Image ID", value='user_defined', interactive=False, visible=False)

                submission_button = gr.Button(value="Check Admission")

                with gr.Accordion(label='Logging Info', open=False):

                    station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                    name = gr.Textbox(label="User Name", lines=1, interactive=True)
                    session = gr.Dropdown(label="Session ID", value='S2', choices=['S1','S2','S3'], interactive=False, visible=False)
                    experiment = gr.Dropdown(label="Experiment ID", value='X2', choices=['X1','X2','X3'], interactive=False, visible=False)
                    image_id = gr.Textbox(label="Image ID", value=imgs[0], interactive=False, visible=False)
                    user_image_dir = gr.Textbox(label="User Image Directory", value=NEW_IMG_DIR, interactive=False, visible=False)

            with gr.Column():

                outputs = gr.Label(num_top_classes=3, label="Output")
                slider_instructions = gr.Markdown('Use the slider to choose the point at which the smart door should open.')
                decision_boundary = gr.Slider(0, 100, 50, step=10, label='Decision Slider', interactive=True, visible=True)

            with gr.Column():

                output_image = gr.Image(label="Output", value='admission_images/background_starting.jpg')

                with gr.Accordion(label="Images", open=False):
                    
                    examples = gr.Examples([[path.join(IMG_DIR, img), img] for img in imgs if img not in ['.DS_Store']], [image_1, image_id], None, examples_per_page=3)    

        submission_button.click(collect_classes_and_predict, [image_1, positive_text, negative_text, decision_boundary, station, name, session, experiment, image_id, text_id], [outputs, output_image], show_progress=False)
        upload_button.upload(upload_file_by_station, [upload_button, user_image_dir, station, experiment], [image_1, image_id])

demo.launch(share=True)