# @title Scheduler functions
from diffusers import (EulerAncestralDiscreteScheduler,
                       DPMSolverSDEScheduler,
                       DPMSolverMultistepScheduler,
                       DDIMScheduler,
                       PNDMScheduler,
                       LCMScheduler
                       )
import torchsde

def set_ddim_scheduler(pipeline):
  pipeline.scheduler = DDIMScheduler.from_config(
      pipeline.scheduler.config,
       rescale_betas_zero_snr=True
      )

def set_dpm_sde_scheduler(pipeline):
  pipeline.scheduler = DPMSolverSDEScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    timestep_spacing="linspace",
    prediction_type="epsilon",
    use_karras_sigmas=True
)

def set_pndm_scheduler(pipeline):
  pipeline.scheduler = PNDMScheduler(
    num_train_timesteps=1000,       # Number of diffusion steps to train the model
    beta_start=0.0001,             # Starting beta value of inference
    beta_end=0.02,                 # Final beta value
    beta_schedule="linear",        # Beta schedule: 'linear', 'scaled_linear', or 'squaredcos_cap_v2'
    trained_betas=None,            # Optional array of betas to override beta_start and beta_end
    skip_prk_steps=False,          # Skip Runge-Kutta steps if True
    set_alpha_to_one=False,        # Fix alpha product to 1 for the final step if True
    prediction_type="epsilon",     # Prediction type: 'epsilon' (predict noise) or 'v_prediction'
    timestep_spacing="leading",    # Timestep scaling method
    steps_offset=0                 # Offset added to inference steps
)

def set_euler_scheduler(pipeline):
  pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
      pipeline.scheduler.config,
      timestep_spacing='linspace',
     )

def set_dpm_scheduler(pipeline):
  pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
      pipeline.scheduler.config,
      algorithm_type="sde-dpmsolver++",
      use_karras_sigmas=True,
      )

def set_lcm_scheduler(pipeline):
    pipeline.scheduler = LCMScheduler.from_config(
            num_train_steps=1000,
            beta_start=8.5e-4,
            beta_end=0.012,
            original_inference_steps=50,
            prediction_type='epsilon'
            )

