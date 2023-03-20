import gradio as gr

from clip_model import ImageClassifier

# Define parameters
IMG_DIR = 'example_images'
LOGFILE = 'logs/test_log.txt'

# Define an image classifier
model = ImageClassifier()

# Set model parameters
model.init_logfile(LOGFILE)

TEXT_CLASSES = ['monster', 'not a monster', 'maybe a monster']
TEXT_CLASSES = ['dog', 'not a dog']
IMAGE_DICT = {'monster': 'admission_images/green_check.png', 'not a monster': 'admission_images/stop_sign.png', 'maybe a monster': 'admission_images/question_mark.png'}
IMAGE_DICT = {'dog': 'admission_images/door_open.jpg', 'not a dog': 'admission_images/door_close.jpg'}#, 'maybe a monster': 'admission_images/question_mark.png'}

def monster_wrapper(image):

    dict_probs, dict_logs = model.predict(image, TEXT_CLASSES)
    top_class = max(dict_probs, key=dict_probs.get)
    
    return dict_probs, IMAGE_DICT[top_class]

# Interface definition and deployment
with gr.Blocks() as demo:

    admit = gr.Markdown('### Welcome to the party!')

    with gr.Accordion(label="Scan your face to check admission status!", open=False):

        with gr.Row():

            with gr.Column():
                image_1 = gr.Image(source='webcam', streaming=True)
                submission_button = gr.Button(value="Check Admission")

            outputs = gr.Label(num_top_classes=3, label="Output")
            output_image = gr.Image(label="Output", value='admission_images/background_starting.jpg')

        submission_button.click(monster_wrapper, image_1, [outputs, output_image])
        
demo.launch(share=True)