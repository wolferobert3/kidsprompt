import torch
import datetime

import gradio as gr
import pandas as pd

from clip_model import TextRetriever
from utils import log_json_text_retrieval, save_similarity_tensor, save_regularized_tensor

# Define parameters
LOGFILE = 'logs/test_log.txt'
VOCAB_FILE = 'word_lexica/NRC-VAD-Lexicon.txt'
IMAGE_ID = 'webcam_stream'
TENSOR_DIR = 'tensors'
K_VALUE = 10

# Load in ANEW data
nrc_vad = pd.read_csv(VOCAB_FILE, sep='\t', header=None, names=['term', 'pleasure', 'arousal', 'dominance'], na_filter=False)
vocab = [word.strip() for word in nrc_vad['term'].tolist() if word]

# Define regularization vectors
pleasantness = torch.tensor(nrc_vad['pleasure'])
inverse_arousal = torch.tensor(1-nrc_vad['arousal'])

# Define regularization dictionary
regularization_dict = {'pleasantness': pleasantness, 'inverse_arousal': inverse_arousal}

# Define an image classifier
model = TextRetriever(vocab, regularization_dict=regularization_dict)

# Set model parameters
model.set_k_value(K_VALUE)
model.init_logfile(LOGFILE)

def compute_word_cloud_and_log(image, regularizers, station, user, session_ID, experiment, image_ID, text_ID, logfile=None):

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cosine_similarities = model.get_cosine_similarities(image)
    save_similarity_tensor(cosine_similarities, station, experiment, date_time, TENSOR_DIR)

    for regularizer in regularizers:
        cosine_similarities = model.regularize_cosine_similarities(cosine_similarities, regularizer)
        save_regularized_tensor(cosine_similarities, station, experiment, date_time, TENSOR_DIR, regularizer)

    top_words, top_weights = model.return_top_k_words(cosine_similarities)
    word_cloud = model.compute_word_cloud(top_words, top_weights)

    log_json_text_retrieval(station, user, session_ID, experiment, image_ID, text_ID, top_words, top_weights, logfile, date_time)

    return word_cloud

# Interface definition and deployment
with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column():

            image_1 = gr.Image(source='webcam', streaming=True)

            with gr.Accordion(label='Logging Info', open=False):

                station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                name = gr.Textbox(label="User Name", lines=1, interactive=True)
                regularizers = gr.CheckboxGroup(choices=['pleasantness', 'inverse_arousal'], label="Regularizers", interactive=True, value=None)
                session = gr.Dropdown(label="Session ID", value='S3', choices=['S1','S2','S3'], interactive=False, visible=False)
                experiment = gr.Dropdown(label="Experiment ID", value='X3', choices=['X1','X2','X3'], interactive=False, visible=False)
                image_id = gr.Textbox(label="Image ID", value=IMAGE_ID, interactive=False, visible=False)
                text_id = gr.Textbox(label="Text ID", value=VOCAB_FILE, interactive=False, visible=False)
    
        outputs = gr.Image(label="Output")

    image_1.change(compute_word_cloud_and_log, [image_1, regularizers, station, name, session, experiment, image_id, text_id], outputs, show_progress=False)

demo.launch(share=True)