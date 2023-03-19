import torch
import time

from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from wordcloud import WordCloud

from utils import log_json

# Superclass for predicting the similarity between an image and a text based on CLIP similarities

class VLModel:
   
    def __init__(self, model_name='openai/clip-vit-base-patch32'):

        self.model = CLIPModel.from_pretrained(model_name)
        self.logit_scale = self.model.logit_scale.exp()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def get_text_embeddings(self, text, scale=False):

        with torch.no_grad():

          input_text = self.tokenizer(text, padding=True, return_tensors="pt")
          embeddings = self.model.get_text_features(**input_text).squeeze()

          if scale:
              embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings
  
    def get_image_embeddings(self, images, scale=False):
        
        with torch.no_grad():

          image_input = self.processor(images=images, return_tensors="pt")
          embeddings = self.model.get_image_features(**image_input).squeeze()

          if scale:
              embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings

    def scale_image_embeddings(self, image_embeddings):

        scaled_image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        return scaled_image_embeddings

    def scale_text_embeddings(self, text_embeddings):

        scaled_text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        return scaled_text_embeddings

    def scale_logits(self, logits):

        with torch.no_grad():
            scaled_logits = logits * self.logit_scale

        return scaled_logits

    def compute_softmax(self, logits):

        return logits.softmax(dim=0)

    def set_logit_scale(self, logit_scale):

        self.logit_scale = logit_scale

    def get_probability_dict(self, classes, probs):

        return {classes[i]: float(probs[i]) for i in range(len(probs))}

    def init_logfile(self, logfile):

        self.logfile = logfile

        with open(self.logfile, 'w') as f:
            f.write('Initiliazing logfile \n')

    def write_to_logfile(self, message):

        with open(self.logfile, 'a') as f:
            f.write(message + '\n')



# Class for computing the similarity between an image and a text; intended for use with a static image and dynamically changing text classes

class ImageClassifier(VLModel):

    def __init__(self, model_name='openai/clip-vit-base-patch32'):

        VLModel.__init__(self, model_name)

    def predict(self, image, text_classes):

        if type(text_classes) == str:
            text_classes = self.split_text_into_classes(text_classes)

        class_inputs = [i for i in text_classes if i not in ['',' ','\n']]

        scaled_image_embeddings = self.get_image_embeddings(image, scale=True)
        scaled_text_embeddings = self.get_text_embeddings(class_inputs, scale=True)

        cosine_similarities = torch.matmul(scaled_text_embeddings, scaled_image_embeddings)
        logits_per_image = self.scale_logits(cosine_similarities)
        probs = self.compute_softmax(logits_per_image)

        dict_probs = self.get_probability_dict(class_inputs, probs)
        dict_logs = self.create_logging_dict(class_inputs, cosine_similarities.tolist(), logits_per_image.tolist(), probs.tolist())
        
        return dict_probs, dict_logs

    def split_text_into_classes(self, text):

        return text.split('\n')

    def create_logging_dict(self, text_classes, cosine_similarities, logits_per_image, probabilities):

        logging_dict = {'text_classes': text_classes, 'cosine_similarities': cosine_similarities, 'logits_per_image': logits_per_image, 'probabilities': probabilities}

        return logging_dict

    def predict_and_log(self, image, text_classes, station, user, session_ID, experiment, image_ID, text_ID, log_dict, logfile=None):

        dict_probs, log_dict = self.predict(image, text_classes)
        log_json(station, user, session_ID, experiment, image_ID, text_ID, log_dict, logfile)

        return dict_probs


# Class for retrieving the most similar text to an image; intended for use with a dynamically changing image, and static text vocabulary

class TextRetriever(VLModel):

    def __init__(self, vocabulary, model_name='openai/clip-vit-base-patch32'):

        VLModel.__init__(self, model_name)

        self.vocab = vocabulary
        with torch.no_grad():
            self.scaled_reference_text_embeddings = self.get_text_embeddings(self.vocab, scale=True)
        self.k = 10
        self.sleep_time = 2

    def get_cosine_similarities(self, image):

        with torch.no_grad():
            scaled_image_embeddings = self.get_image_embeddings(image, scale=True)
            cosine_similarities = torch.matmul(self.scaled_reference_text_embeddings, scaled_image_embeddings)

        return cosine_similarities

    def return_top_k_words(self, image):

        similarities = self.get_cosine_similarities(image)
        top_k = similarities.topk(self.k, dim=0).indices.numpy()

        top_k_words = [self.vocab[i] for i in top_k]
        top_k_weights = [float(similarities[i].numpy()) for i in top_k]

        message = f'Top {self.k} words: {top_k_words} \nTop {self.k} weights: {top_k_weights}'
        self.write_to_logfile(message)

        return top_k_words, top_k_weights

    # Word cloud function with weights
    def compute_word_cloud(self, image):

        text, weights = self.return_top_k_words(image)
    
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Dark2", max_font_size=150, random_state=42)
        wordcloud.generate_from_frequencies(frequencies=dict(zip(text, weights)))

        time.sleep(self.sleep_time)

        image = wordcloud.to_image()
        return image

    def set_k_value(self, k):

        self.k = k

    def set_sleep_time(self, sleep_time):

        self.sleep_time = sleep_time
