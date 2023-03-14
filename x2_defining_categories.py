import gradio as gr

from clip_model import ImageClassifier
from os import listdir, path

IMG_DIR = 'example_images'
LOGFILE = 'logs/test_log.txt'

# Define an image classifier
model = ImageClassifier()
model.init_logfile(LOGFILE)

# Wrapper to collect text classes and send to model for prediction
def collect_classes_and_predict(image, text_1, text_2, text_3, text_4, text_5):
    return model.predict(image, [text_1, text_2, text_3, text_4, text_5])

# Upload a file from your computer
def upload_file(file_obj):
    return file_obj.name

# Interface definition and deployment
with gr.Blocks() as demo:

    header = gr.Markdown("## CLIP Experiment 2 - Defining Categories")

    with gr.Row():

        with gr.Column():

            image_1 = gr.Image(path.join(IMG_DIR, 'monkey.jpg'), label="Image")

            # Create upload button
            upload_button = gr.UploadButton("Upload an Image", file_types=["image"], file_count="single")

            ids = gr.Markdown('Identify your station and type your name')
            station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
            name = gr.Textbox(label="Name", lines=1, interactive=True)

        with gr.Column():

            class_header = gr.Markdown("Type your classes in the boxes below and click 'Predict!'")

            text_1 = gr.Textbox(label="Class 1", lines=1)
            text_2 = gr.Textbox(label="Class 2", lines=1)
            text_3 = gr.Textbox(label="Class 3", lines=1)
            text_4 = gr.Textbox(label="Class 4", lines=1)
            text_5 = gr.Textbox(label="Class 5", lines=1)

            button = gr.Button('Predict!')
        
        #text_classes = gr.Textbox(label="Text Classes", interactive=True, lines = 5)
        outputs = gr.Label(num_top_classes=5, label="Output")

    # Load in downloaded images to use as examples
    gr.Examples([path.join(IMG_DIR, img) for img in listdir(IMG_DIR)], image_1)

    button.click(collect_classes_and_predict, [image_1, text_1, text_2, text_3, text_4, text_5], outputs, show_progress=False)

    # Define image upload button behavior
    upload_button.upload(upload_file, upload_button, image_1)

    #button.click(model.predict, [station, name, [image_1,text_1,text_2,text_3,text_4,text_5]], outputs, show_progress=False)
    #text_classes.change(predict, [name,image_1,text_classes], outputs, show_progress=False)

demo.launch(share=True)