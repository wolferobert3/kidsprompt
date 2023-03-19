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
        logfile = f'logs/{station}_{user}_{session_ID}_{experiment}.log'

    with open(logfile, 'a') as f:
        f.write(json.dumps(write_dict) + '\n')

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