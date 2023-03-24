import torch
import datetime
import json

from os import listdir, path, mkdir
from PIL import Image

# Function for logging session data to a json file
def log_json(station, user, session_ID, experiment, image_ID, text_ID, log_dict, logfile=None):

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_dict = {'station': station, 'user': user, 'session_ID': session_ID, 'experiment': experiment, 'image_ID': image_ID, 'text_ID': text_ID, 'date_time': date_time}
    write_dict.update(log_dict)

    if not logfile:
        logfile = path.join(f'logs', f'{station}_{user}_{session_ID}_{experiment}.log')

    with open(logfile, 'a') as f:
        f.write(json.dumps(write_dict) + '\n')

# Function for logging session data to a json file
def log_json_text_retrieval(station, user, session_ID, experiment, image_ID, text_ID, model_type, top_words, top_weights, logfile=None, date_time=None):

    if not date_time:
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    write_dict = {'station': station, 'user': user, 'session_ID': session_ID, 'experiment': experiment, 'image_ID': image_ID, 'text_ID': text_ID, 'model_type': model_type, 'date_time': date_time, 'top_words': top_words, 'top_weights': top_weights}

    if not logfile:
        logfile = path.join(f'logs', f'{station}_{user}_{session_ID}_{experiment}.log')

    with open(logfile, 'a') as f:
        f.write(json.dumps(write_dict) + '\n')

# Function for logging session data to a json file
def log_json_resources(station, user, session_ID, experiment, image_ID, text_ID, log_dict, decision_boundary, logfile=None):

    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_dict = {'station': station, 'user': user, 'session_ID': session_ID, 'experiment': experiment, 'image_ID': image_ID, 'text_ID': text_ID, 'date_time': date_time, 'decision_boundary': decision_boundary}
    write_dict.update(log_dict)

    if not logfile:
        logfile = path.join(f'logs', f'{station}_{user}_{session_ID}_{experiment}.log')

    with open(logfile, 'a') as f:
        f.write(json.dumps(write_dict) + '\n')


def save_similarity_tensor(prob_tensor, station, experiment, date_time, tensor_dir):

    child_dir = path.join(tensor_dir, f'{station}_{experiment}')

    if not path.exists(child_dir):
        mkdir(child_dir)

    save_file = path.join(child_dir, f'{date_time}.pt')        

    torch.save(prob_tensor, save_file)

def save_regularized_tensor(prob_tensor, station, experiment, date_time, tensor_dir, regularizer):

    child_dir = path.join(tensor_dir, f'{station}_{experiment}_{regularizer}')

    if not path.exists(child_dir):
        mkdir(child_dir)

    save_file = path.join(child_dir, f'{date_time}.pt')

    torch.save(prob_tensor, save_file)

def save_log_image(image_, station, experiment, date_time, save_dir, fromarray=False):

    child_dir = path.join(save_dir, f'{station}_{experiment}')

    if not path.exists(child_dir):
        mkdir(child_dir)

    save_file = path.join(child_dir, f'{date_time}.png')
    
    if fromarray:
        Image.fromarray(image_).save(save_file, 'PNG')
        return
    
    image_.save(save_file)

# Define a function that saves an image uploaded by the user
def save_novel_image(image_file, image_ID, img_dir):

    if image_ID not in listdir(img_dir):
        Image.open(image_file).save(path.join(img_dir, image_ID))

# Upload a file from your computer
def upload_file(file_obj, img_dir):

    img_file, image_ID = file_obj.name, path.split(file_obj.name)[1]
    save_novel_image(img_file, image_ID, img_dir)

    return img_file, image_ID

# Define a function that saves an image uploaded by the user
def save_novel_image_to_station_dir(image_file, image_ID, img_dir, station, experiment):

    child_dir = path.join(img_dir, f'{station}_{experiment}')
    
    if not path.exists(child_dir):
        mkdir(child_dir)

    if image_ID not in listdir(child_dir):
        Image.open(image_file).save(path.join(child_dir, image_ID))

# Upload a file from your computer
def upload_file_by_station(file_obj, img_dir, station, experiment):

    img_file, image_ID = file_obj.name, path.split(file_obj.name)[1]
    save_novel_image_to_station_dir(img_file, image_ID, img_dir, station, experiment)

    return img_file, image_ID