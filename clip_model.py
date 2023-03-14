import torch
import time

from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from wordcloud import WordCloud

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

    def predict(self, image, text_classes, log=True):

        if type(text_classes) == str:
            text_classes = self.split_text_into_classes(text_classes)

        text_classes = [i for i in text_classes if i not in ['',' ','\n']]

        scaled_image_embeddings = self.get_image_embeddings(image, scale=True)
        scaled_text_embeddings = self.get_text_embeddings(text_classes, scale=True)

        cosine_similarities = torch.matmul(scaled_text_embeddings, scaled_image_embeddings)
        logits_per_image = self.scale_logits(cosine_similarities)

        if log:
            message = f'Cosine similarities: {cosine_similarities} \nLogits per image: {logits_per_image}'
            self.write_to_logfile(message)

        probs = self.compute_softmax(logits_per_image)
        dict_probs = self.get_probability_dict(text_classes, probs)

        return dict_probs

    def split_text_into_classes(self, text):

        return text.split('\n')

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

"""
# Write the probabilities to the log file as json
def log_probabilities(station, name, cosine_similarities, dict_probs, dict_logits):
   
   with open(LOGFILE, 'a') as f:
      f.write(f'Station:{station}\tName:{name}\tcosine_similarities:' + json.dumps(cosine_similarities) + '\tprobabilities:\t' + json.dumps(dict_probs) + '\tlogits:\t' + json.dumps(dict_logits) + '\n')
"""