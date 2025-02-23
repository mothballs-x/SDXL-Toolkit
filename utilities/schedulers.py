# Scheduler Functions
# @title Scheduler functions
from diffusers import (EulerAncestralDiscreteScheduler,
                       DPMSolverSDEScheduler,
                       DPMSolverMultistepScheduler,
                       DDIMScheduler,
                       PNDMScheduler, LCMScheduler)

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
        prediction_type="epsilon"
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


def set_euler_scheduler(pipeline, spacing):
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing=spacing,
     )


def set_dpm_scheduler(pipeline):
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
  )


def set_lcm_scheduler(pipeline):
    pipeline.scheduler = LCMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        prediction_type='v_prediction',
        timestep_spacing='trailing',
        steps_offset=1
    )


# Scheduler Dict
schedulers_dict = {
    'euler': set_euler_scheduler,
    'dpm': set_dpm_scheduler,
    'dpm_sde': set_dpm_sde_scheduler,
    'pndm': set_pndm_scheduler,
    'ddim': set_ddim_scheduler,
    'lcm': set_lcm_scheduler
}
