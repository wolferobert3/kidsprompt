import gradio as gr

from os import path, listdir

from clip_model import ImageClassifier

# Define parameters
IMG_DIR = 'example_images/pets'
LOGFILE = 'logs/test_log.txt'

# List of example images
imgs = sorted([img for img in listdir(IMG_DIR) if img not in ['.DS_Store']], key=lambda x: int(x.split('.')[0]))

# Define an image classifier
model = ImageClassifier()

# Set model parameters
model.init_logfile(LOGFILE)

TEXT_CLASSES = ['monster', 'not a monster', 'maybe a monster']
TEXT_CLASSES = ['dog', 'not a dog']
IMAGE_DICT = {'monster': 'admission_images/green_check.png', 'not a monster': 'admission_images/stop_sign.png', 'maybe a monster': 'admission_images/question_mark.png'}
IMAGE_DICT = {'positive': 'admission_images/door_open.jpg', 'negative': 'admission_images/door_close.jpg'}#, 'maybe a monster': 'admission_images/question_mark.png'}

def prediction_wrapper(image, positive_text, negative_text, decision_boundary):

    class_map = {positive_text: 'positive', negative_text: 'negative'}
    text_classes = [positive_text, negative_text]
    dict_probs, dict_logs = model.predict(image, text_classes)

    if dict_probs [positive_text] > decision_boundary:
        output_image = IMAGE_DICT[class_map[positive_text]]
    else:
        output_image = IMAGE_DICT[class_map[negative_text]]
    
    return dict_probs, output_image

# Interface definition and deployment
with gr.Blocks() as demo:

    admit = gr.Markdown('### Buddy and Luna\'s Dog Door')

    with gr.Accordion(label="Dog Door", open=False):

        with gr.Row():

            with gr.Column():
                #image_1 = gr.Image(source='webcam', streaming=True)
                image_1 = gr.Image(path.join(IMG_DIR, imgs[0]), label="Image")
                image_id = gr.Textbox(label="Image ID", value=imgs[0], interactive=False, visible=False)
                positive_text = gr.Textbox(label="Positive Class", value="dog", interactive=True, visible=True)
                negative_text = gr.Textbox(label="Negative Class", value="not a dog", interactive=True, visible=True)
                submission_button = gr.Button(value="Check Admission")

            with gr.Column():
                outputs = gr.Label(num_top_classes=3, label="Output")
                decision_boundary = gr.Slider(0, 100, 50, step=10, label='Decision Boundary')

            output_image = gr.Image(label="Output", value='admission_images/background_starting.jpg')

        submission_button.click(prediction_wrapper, [image_1, positive_text, negative_text, decision_boundary], [outputs, output_image])

        with gr.Accordion(label="Images", open=False):
            examples = gr.Examples([[path.join(IMG_DIR, img), img] for img in imgs], [image_1, image_id], None, examples_per_page=3)    

demo.launch(share=True)