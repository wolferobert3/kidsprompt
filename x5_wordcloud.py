import gradio as gr

from clip_model import TextRetriever

# Define parameters
LOGFILE = 'logs/test_log.txt'
VOCAB_FILE = 'words.csv'
K_VALUE = 10

# Load in vocab
vocab = [word for word in open(VOCAB_FILE, 'r').read().split(',') if word not in ['',' ','\n']]

# Define an image classifier
model = TextRetriever(vocab)

# Set model parameters
model.set_k_value(K_VALUE)
model.init_logfile(LOGFILE)

# Interface definition and deployment
with gr.Blocks() as demo:

    with gr.Row():

        image_1 = gr.Image(source='webcam', streaming=True)
        outputs = gr.Image(label="Output")

    image_1.change(model.compute_word_cloud, image_1, outputs, show_progress=False)

demo.launch(share=True)