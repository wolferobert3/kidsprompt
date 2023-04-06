import torch
import datetime

import gradio as gr
import pandas as pd

from clip_model import TextRetriever, EmojiRetriever, GestureRetriever
from utils import log_json_text_retrieval, save_similarity_tensor, save_regularized_tensor, save_log_image

# Define parameters
VOCAB_FILE = 'word_lexica/Wordlist_Kidsteam.csv'
IMAGE_ID = 'webcam_stream'
TENSOR_DIR = 'tensors'
WORDCLOUD_DIR = 'word_clouds'
WEBCAM_DIR = 'webcam_images'
K_VALUE = 10

# Load in ANEW data
#nrc_vad = pd.read_csv(VOCAB_FILE, sep='\t', header=None, names=['term', 'pleasure', 'arousal', 'dominance'], na_filter=False, nrows=1024)
#vocab = [word.strip() for word in nrc_vad['term'].tolist() if word]

words = pd.read_csv(VOCAB_FILE)
vocab = [word.strip() for word in words['term'].tolist() if word]

# Define regularization vectors
pleasantness = torch.tensor(words['pleasure'])
inverse_arousal = torch.tensor(1-words['arousal'])

# Define regularization dictionary
regularization_dict = {'pleasantness': pleasantness, 'inverse_arousal': inverse_arousal}

# Define an image classifier
text_model = TextRetriever(vocab, regularization_dict=regularization_dict)
emoji_model = EmojiRetriever()
gesture_model = GestureRetriever()

# Set model parameters
text_model.set_k_value(K_VALUE)
emoji_model.set_k_value(1)
gesture_model.set_k_value(1)

MODEL_DICT = {'text': text_model, 'emoji': emoji_model, 'gesture': gesture_model}

def compute_word_cloud_and_log(image, regularizers, station, user, session_ID, experiment, image_ID, text_ID, model_type, logfile=None):

    current_model = MODEL_DICT[model_type]

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cosine_similarities = current_model.get_cosine_similarities(image)
    #save_similarity_tensor(cosine_similarities, station, experiment, date_time, TENSOR_DIR)

    for regularizer in regularizers:
        cosine_similarities = current_model.regularize_cosine_similarities(cosine_similarities, regularizer)
        #save_regularized_tensor(cosine_similarities, station, experiment, date_time, TENSOR_DIR, regularizer)

    top_words, top_weights = current_model.return_top_k_words(cosine_similarities)

    if model_type == 'text':
        cloud_image = current_model.compute_word_cloud(top_words, top_weights)
    elif model_type == 'emoji':
        cloud_image = current_model.compute_emoji_cloud(top_words, top_weights)
    elif model_type == 'gesture':
        cloud_image = current_model.compute_gesture_cloud(top_words, top_weights)

    #log_json_text_retrieval(station, user, session_ID, experiment, image_ID, text_ID, model_type, top_words, top_weights, logfile, date_time)
    #save_log_image(cloud_image, station, experiment, date_time, WORDCLOUD_DIR)
    #save_log_image(image, station, experiment, date_time, WEBCAM_DIR, fromarray=True)

    return cloud_image

# Interface definition and deployment
with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column():

            image_1 = gr.Image(source='webcam', streaming=True)

            with gr.Accordion(label='Logging Info', open=False):

                station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                name = gr.Textbox(label="User Name", lines=1, interactive=True)
                regularizers = gr.CheckboxGroup(choices=['pleasantness', 'inverse_arousal'], label="Regularizers", interactive=True, visible=False, value=None)
                model_type = gr.Radio(choices=['text', 'emoji', 'gesture'], label="Model Type", interactive=True, value='text')
                session = gr.Dropdown(label="Session ID", value='S3', choices=['S1','S2','S3'], interactive=False, visible=False)
                experiment = gr.Dropdown(label="Experiment ID", value='X3', choices=['X1','X2','X3'], interactive=False, visible=False)
                image_id = gr.Textbox(label="Image ID", value=IMAGE_ID, interactive=False, visible=False)
                text_id = gr.Textbox(label="Text ID", value=VOCAB_FILE, interactive=False, visible=False)
    
        outputs = gr.Image(label="Output")

    image_1.change(compute_word_cloud_and_log, [image_1, regularizers, station, name, session, experiment, image_id, text_id, model_type], outputs, show_progress=False)

demo.launch(share=True)