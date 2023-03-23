import openai
import requests
from os import path

openai.api_key = open(f'oai_key.txt', 'r').read().strip()

class TextToImageModel:

    def __init__(self, save_path: str = '') -> None:
        
        self.save_path = save_path

    def set_save_path(self, save_path: str) -> None:

        self.save_path = save_path

    def generate_images(self, prompt: str, n: int = 4, size: str = "256x256") -> list:

        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size,
        )

        image_urls = [response['data'][i]['url'] for i in range(n)]

        return image_urls

    def retrieve_images(self, image_urls: list, prompt: str) -> None:

        for i in range(len(image_urls)):
            local_img = requests.get(image_urls[i], allow_redirects=True)

            with open(path.join(self.save_path, f'{prompt}_{i}.png'), 'wb') as f:
                f.write(local_img.content)

    def generate_and_retrieve_images(self, prompt: str, n: int = 4, size: str = "256x256") -> None:

        image_urls = self.generate_images(prompt, n, size)
        self.retrieve_images(image_urls, prompt)

    def init_logfile(self, logfile: str) -> None:

        self.logfile = logfile

        with open(self.logfile, 'w') as f:
            f.write('Initiliazing logfile \n')

    def write_to_logfile(self, message: str) -> None:

        with open(self.logfile, 'a') as f:
            f.write(message + '\n')