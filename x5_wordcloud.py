import torch
import numpy as np

import gradio as gr
import pandas as pd

from clip_model import TextRetriever

# Define parameters
LOGFILE = 'logs/test_log.txt'
VOCAB_FILE = 'word_lexica/NRC-VAD-Lexicon.txt'
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

# Interface definition and deployment
with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column():

            image_1 = gr.Image(source='webcam', streaming=True)

            with gr.Accordion(label='Logging Info', open=False):

                station = gr.Dropdown(['iMac1','iMac2','iMac3','iMac4','iMac5'], value='iMac1', label="Station", interactive=True)
                name = gr.Textbox(label="User Name", lines=1, interactive=True)
                session = gr.Dropdown(label="Session ID", value='S1', choices=['S1','S2','S3'], interactive=False, visible=False)
                experiment = gr.Dropdown(label="Experiment ID", value='X1', choices=['X1','X2','X3'], interactive=False, visible=False)
                image_id = gr.Textbox(label="Image ID", value='monkey.jpg', interactive=False, visible=False)
                user_image_dir = gr.Textbox(label="User Image Directory", value=NEW_IMG_DIR, interactive=False, visible=False)
    
        outputs = gr.Image(label="Output")

    image_1.change(model.compute_word_cloud, image_1, outputs, show_progress=False)

demo.launch(share=True)