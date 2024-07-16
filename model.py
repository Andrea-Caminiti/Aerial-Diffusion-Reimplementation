import torch
import torch.nn.functional as F
import numpy as np
from dataset import create_dataloader
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from typing import Union, Optional
from tqdm.auto import tqdm

class AerialStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        feature_extractor: CLIPFeatureExtractor,
    ) -> None:
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
    
    def train(
        self,
        batch: tuple[torch.FloatTensor, list[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.001,
        diffusion_model_learning_rate: float = 2e-6,
        text_embedding_optimization_steps: int = 500,
        model_fine_tuning_optimization_steps: int = 1000,
    ):      
        image_tensor, captions = batch
        captions_updated = ['front view of ' + caption.lower() for caption in captions]

        text_inputs = [self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ) for prompt in captions_updated]

        input_ids = torch.LongTensor([text_in['input_ids'].tolist() for text_in in text_inputs])
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(input_ids.to(self.device))[0], requires_grad=True) # 77 x 768; 768 for each token
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()

        prompt_list_modified = ['aerial view of ' + caption.lower() for caption in captions] 
        text_input_modified = [self.tokenizer(
            prompt_modified,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ) for prompt_modified in prompt_list_modified]
        input_ids_modified = torch.LongTensor([text_in['input_ids'].tolist() for text_in in text_input_modified])
        text_embeddings_orig = torch.nn.Parameter(
            self.text_encoder(input_ids_modified.to(self.device))[0]
        )
        text_embeddings_orig = text_embeddings_orig.detach()
        
        optimizer = torch.optim.Adam(
            [text_embeddings],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )
        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        latents_dtype = text_embeddings.dtype
        image_tensor = image_tensor.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(image_tensor).latent_dist
        image_latents = init_latent_image_dist.sample(generator=generator)
        image_latents = 0.18215 * image_latents
        self.image_latents = image_latents

        progress_bar = tqdm(range(text_embedding_optimization_steps))
        progress_bar.set_description("Steps")

        for _ in range(text_embedding_optimization_steps):
            # Sample noise that we'll add to the latents
            print(_)
            noise = torch.randn(image_latents.shape).to(image_latents.device)
            timesteps = torch.randint(1000, (1,), device=image_latents.device)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)

            # Predict the noise residual
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            progress_bar.set_postfix(loss)
            
        text_embeddings.requires_grad_(False)

        # Now we fine tune the unet to better reconstruct the image
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
    
        for _ in range(model_fine_tuning_optimization_steps):
                torch.cuda.empty_cache()
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(1000, (1,), device=image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)
                
                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                torch.cuda.empty_cache()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

if __name__ == '__main__':
    d_loader = create_dataloader(16)
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        custom_pipeline='./model.py', cache_dir = 'dir_name',
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    )
    for batch in d_loader:
        pipe.train(batch)