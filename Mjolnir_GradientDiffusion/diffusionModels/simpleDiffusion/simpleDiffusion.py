from utils.networkHelper import *
from diffusionModels.simpleDiffusion.varianceSchedule import VarianceSchedule






class DiffusionModel(nn.Module):
    def __init__(self,
                 schedule_name="linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None,
                 recover_t=1000):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model


        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)


        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, noiseimg,shape,recover_t=1000):
        device = next(self.denoise_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)

        img = torch.randn(shape, device=device)

        noiseimg=noiseimg
        imgs = []
        imgs2 = []
        # #
        # for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
        #     img = self.p_sample(noiseimg, torch.full((b,), i, device=device, dtype=torch.long), i)
        #     imgs.append(noiseimg.cpu().numpy())

        for i in tqdm(reversed(range(0, recover_t)), desc='sampling loop time step', total=recover_t):
            img = self.p_sample(noiseimg, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, image_size, noiseimg, batch_size=16, channels=3,recover_t=1000):
        return self.p_sample_loop(noiseimg, recover_t=recover_t,shape=(batch_size, channels, image_size, image_size))

    def forward(self, mode,noiseimg=None,recover_t=1000, **kwargs):
        if mode == "train":

            if "x_start" and "t" in kwargs.keys():

                if "loss_type" and "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"],
                                             noise=kwargs["noise"], loss_type=kwargs["loss_type"])
                elif "loss_type" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], loss_type=kwargs["loss_type"])
                elif "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"])
                else:
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

            else:
                raise ValueError("Need to input x_start and t！")

        elif mode == "generate":
            if "image_size" and "batch_size" and "channels" in kwargs.keys():
                return self.sample(image_size=kwargs["image_size"],
                                   noiseimg=noiseimg,
                                   batch_size=kwargs["batch_size"],
                                   channels=kwargs["channels"],
                                   recover_t=recover_t)
            else:
                raise ValueError("Need to input image_size, batch_size, channels etc.")
        else:
            raise ValueError("mode should be {train} or {generate}")





