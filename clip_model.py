import torch
import time

import numpy as np

from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from wordcloud import WordCloud
from os import path
from PIL import Image

from EmojiCloud.plot import plot_dense_emoji_cloud
from EmojiCloud.emoji import EmojiManager
from EmojiCloud.canvas import EllipseCanvas, RectangleCanvas
from EmojiCloud.vendors import TWITTER

from emoji_vocab import EmojiVocab, GestureVocab
from utils import log_json

# Superclass for predicting the similarity between an image and a text based on CLIP similarities

class VLModel:
   
    def __init__(self, model_name: str ='openai/clip-vit-base-patch32') -> None:

        self.model = CLIPModel.from_pretrained(model_name)
        self.logit_scale = self.model.logit_scale.exp()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def get_text_embeddings(self, text: str, scale: bool = False) -> torch.Tensor:

        with torch.no_grad():

          input_text = self.tokenizer(text, padding=True, return_tensors="pt")
          embeddings = self.model.get_text_features(**input_text).squeeze()

          if scale:
              embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings
  
    def get_image_embeddings(self, images: list, scale: bool = False) -> torch.Tensor:
        
        with torch.no_grad():

          image_input = self.processor(images=images, return_tensors="pt")
          embeddings = self.model.get_image_features(**image_input).squeeze()

          if scale:
              embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

        return embeddings

    def scale_image_embeddings(self, image_embeddings: torch.Tensor) -> torch.Tensor:

        scaled_image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)

        return scaled_image_embeddings

    def scale_text_embeddings(self, text_embeddings: torch.Tensor) -> torch.Tensor:

        scaled_text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        return scaled_text_embeddings

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            scaled_logits = logits * self.logit_scale

        return scaled_logits

    def compute_softmax(self, logits: torch.Tensor) -> torch.Tensor:

        return logits.softmax(dim=0)

    def set_logit_scale(self, logit_scale: float) -> None:

        self.logit_scale = logit_scale

    def get_probability_dict(self, classes: list, probs: list) -> dict:

        return {classes[i]: float(probs[i]) for i in range(len(probs))}

    def init_logfile(self, logfile: str) -> None:

        self.logfile = logfile

        with open(self.logfile, 'w') as f:
            f.write('Initiliazing logfile \n')

    def write_to_logfile(self, message: str) -> None:

        with open(self.logfile, 'a') as f:
            f.write(message + '\n')


# Class for computing the similarity between an image and a text; intended for use with a static image and dynamically changing text classes

class ImageClassifier(VLModel):

    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32') -> None:

        VLModel.__init__(self, model_name)

    def predict(self, image: np.array, text_classes: str or list) -> dict:

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

    def split_text_into_classes(self, text: str) -> list:

        return text.split('\n')

    def create_logging_dict(self, text_classes: list, cosine_similarities: list, logits_per_image: list, probabilities: list) -> dict:

        logging_dict = {'text_classes': text_classes, 'cosine_similarities': cosine_similarities, 'logits_per_image': logits_per_image, 'probabilities': probabilities}

        return logging_dict

    def predict_and_log(self, image: np.array, text_classes: list, station: str, user: str, session_ID: str, experiment: str, image_ID: str, text_ID: str, log_dict: dict, logfile: str = None) -> dict:

        dict_probs, log_dict = self.predict(image, text_classes)
        log_json(station, user, session_ID, experiment, image_ID, text_ID, log_dict, logfile)

        return dict_probs


# Class for retrieving the most similar text to an image; intended for use with a dynamically changing image, and static text vocabulary

class TextRetriever(VLModel):

    def __init__(self, vocabulary: list, model_name: str = 'openai/clip-vit-base-patch32', regularization_dict: dict = {}) -> None:

        VLModel.__init__(self, model_name)

        self.vocab = vocabulary
        self.regularization_dict = regularization_dict
        with torch.no_grad():
            self.scaled_reference_text_embeddings = self.get_text_embeddings(self.vocab, scale=True)
        self.k = 10
        self.sleep_time = 2

    def get_cosine_similarities(self, image: np.array) -> torch.Tensor:

        with torch.no_grad():
            scaled_image_embeddings = self.get_image_embeddings(image, scale=True)
            cosine_similarities = torch.matmul(self.scaled_reference_text_embeddings, scaled_image_embeddings)

        return cosine_similarities

    def save_cosine_similarities(self, cosine_similarities: torch.Tensor, save_dir: str) -> None:
            
            torch.save(cosine_similarities, path.join(save_dir, 'cosine_similarities.t'))

    def regularize_cosine_similarities(self, cosine_similarities: torch.Tensor, regularizer: str) -> torch.Tensor:

        regularization_vector = self.regularization_dict[regularizer]

        with torch.no_grad():
            cosine_similarities = cosine_similarities * regularization_vector

        return cosine_similarities

    def return_top_k_words(self, similarities: torch.Tensor) -> tuple:

        top_k = similarities.topk(self.k, dim=0).indices.numpy()

        top_k_words = [self.vocab[i] for i in top_k]
        top_k_weights = [float(similarities[i].numpy()) for i in top_k]

        #message = f'Top {self.k} words: {top_k_words} \nTop {self.k} weights: {top_k_weights}'
        #self.write_to_logfile(message)

        return top_k_words, top_k_weights

    # Word cloud function with weights
    def compute_word_cloud(self, text: list, weights: list) -> Image.Image:
    
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Dark2", max_font_size=150, random_state=42)
        wordcloud.generate_from_frequencies(frequencies=dict(zip(text, weights)))

        time.sleep(self.sleep_time)

        return wordcloud.to_image()

    def compute_word_cloud_from_image(self, image: np.array, regularizers: list = []) -> Image.Image:
            
            cosine_similarities = self.get_cosine_similarities(image)
            cosine_similarities = self.regularize_cosine_similarities(cosine_similarities, regularizers)
            top_k_words, top_k_weights = self.return_top_k_words(cosine_similarities)
            word_cloud = self.compute_word_cloud(top_k_words, top_k_weights)
    
            return word_cloud

    def set_k_value(self, k: int) -> None:

        self.k = k

    def set_sleep_time(self, sleep_time: int) -> None:

        self.sleep_time = sleep_time

    def set_regularization_dict(self, regularization_dict: dict) -> None:

        self.regularization_dict.update(regularization_dict)

class EmojiRetriever(TextRetriever):
    
    def __init__(self, vocabulary: list = EmojiVocab.vocab, model_name: str = 'openai/clip-vit-base-patch32', regularization_dict: dict = {}, emoji_dict: dict = EmojiVocab.emoji_dict_unicode) -> None:

        TextRetriever.__init__(self, vocabulary, model_name, regularization_dict)
        self.emoji_dict = emoji_dict
        self.canvas_width = 24*10
        self.canvas_height = 24*4

    # Word cloud function with weights
    def compute_emoji_cloud(self, text: list, weights: list) -> Image.Image:
    
        text = [self.emoji_dict[word].replace('\\U000','') for word in text]
        dict_weight = dict(zip(text, weights))
        emoji_list = EmojiManager.create_list_from_single_vendor(dict_weight, TWITTER)
        canvas = RectangleCanvas(self.canvas_width, self.canvas_height)

        return plot_dense_emoji_cloud(canvas, emoji_list)

    def compute_emoji_cloud_from_image(self, image: np.array, regularizers: list = []) -> Image.Image:
            
        cosine_similarities = self.get_cosine_similarities(image)
        cosine_similarities = self.regularize_cosine_similarities(cosine_similarities, regularizers)
        top_k_words, top_k_weights = self.return_top_k_words(cosine_similarities)
        top_k_emojis = [self.emoji_dict[word] for word in top_k_words]
        emoji_cloud = self.compute_emoji_cloud(top_k_emojis, top_k_weights)

        return emoji_cloud
    
    def set_canvas_size(self, width: int, height: int) -> None:

        self.canvas_width = width
        self.canvas_height = height


class GestureRetriever(TextRetriever):
    
    def __init__(self, vocabulary: list = GestureVocab.vocab, model_name: str = 'openai/clip-vit-base-patch32', regularization_dict: dict = {}, gesture_dict: dict = GestureVocab.gesture_dict_unicode) -> None:

        TextRetriever.__init__(self, vocabulary, model_name, regularization_dict)
        self.gesture_dict = gesture_dict
        self.canvas_width = 24*10
        self.canvas_height = 24*4

    # Word cloud function with weights
    def compute_gesture_cloud(self, text: list, weights: list) -> Image.Image:
    
        text = [self.gesture_dict[word].replace('\\U000','') for word in text]
        dict_weight = dict(zip(text, weights))
        emoji_list = EmojiManager.create_list_from_single_vendor(dict_weight, TWITTER)
        canvas = RectangleCanvas(self.canvas_width, self.canvas_height)

        return plot_dense_emoji_cloud(canvas, emoji_list)

    def compute_gesture_cloud_from_image(self, image: np.array, regularizers: list = []) -> Image.Image:
            
        cosine_similarities = self.get_cosine_similarities(image)
        cosine_similarities = self.regularize_cosine_similarities(cosine_similarities, regularizers)
        top_k_words, top_k_weights = self.return_top_k_words(cosine_similarities)
        top_k_gestures = [self.gesture_dict[word] for word in top_k_words]
        gesture_cloud = self.compute_gesture_cloud(top_k_gestures, top_k_weights)

        return gesture_cloud