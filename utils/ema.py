import torch
import copy

class EMA:
    """
    Exponential Moving Average (EMA) for model weights and buffers.
    Covers both parameters and buffers (e.g., BatchNorm stats).
    """
    def __init__(self, beta=0.995, step_start_ema=2000):
        self.beta = beta
        self.step = 0
        self.step_start_ema = step_start_ema

    def update_model_average(self, ema_model, current_model):
        """
        Update EMA model weights and buffers using current model.
        """
        with torch.no_grad():
            ema_state = ema_model.state_dict()
            model_state = current_model.state_dict()
            for key in ema_state.keys():
                if key in model_state:
                    ema_state[key].copy_(
                        self.update_average(ema_state[key], model_state[key])
                    )

    def update_average(self, old, new):
        """
        EMA formula for a tensor.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        """
        Perform EMA update or initialize if before warmup.
        """
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Sync EMA model weights to match current model exactly.
        """
        ema_model.load_state_dict(model.state_dict())

def create_ema_model(model, beta=0.995, step_start_ema=2000):
    """
    Deepcopy model and create an EMA wrapper.
    """
    ema_model = copy.deepcopy(model)
    ema_model.eval()  # EMA model is always used in eval mode
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema = EMA(beta=beta, step_start_ema=step_start_ema)
    return ema_model, ema
