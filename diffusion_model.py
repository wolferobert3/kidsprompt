import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

class TextToImageModel:

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-base") -> None:
        
        self.model_name = model_name

        self.pipe = DiffusionPipeline.from_pretrained(self.model_name, torch_dtype=torch.float16, revision='fp16')
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

    def generate_image(self, prompt: str) -> Image.Image:
        return self.pipe(prompt, num_inference_steps=25).images[0]

    def generate_images(self, prompt: str) -> Image.Image:
        return self.pipe(prompt, num_inference_steps=25).images

    def generate_image_from_multiple(self, prompt1: str, prompt2: str, prompt3: str) -> Image.Image:

        prompt = ' '.join([prompt1, prompt2, prompt3])
        return self.pipe(prompt, num_inference_steps=25).images[0]

    def generate_images_from_multiple(self, prompt1: str, prompt2: str, prompt3: str) -> Image.Image:

        prompt = ' '.join([prompt1, prompt2, prompt3])
        return self.pipe(prompt, num_inference_steps=25).images

    def init_logfile(self, logfile: str) -> None:

        self.logfile = logfile

        with open(self.logfile, 'w') as f:
            f.write('Initiliazing logfile \n')

    def write_to_logfile(self, message: str) -> None:

        with open(self.logfile, 'a') as f:
            f.write(message + '\n')