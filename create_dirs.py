from os import makedirs, path

DIRS = ['example_images', 'logs', 'tensors', 'user_images', 'webcam_images', 'word_clouds']

for dir in DIRS:
    if not path.exists(dir):
        makedirs(dir)