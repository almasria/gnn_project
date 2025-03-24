from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HyperparamConfig:
    # General parameters
    total_i: int = None
    dataset: str = None
    model_name: str = None
    in_dim: int = None
    hidden_dim: int = None
    out_dim: int = None
    num_layer: int = None
    init_weights: str = None
    bias_fill: float = None

    # Model-specific parameters (only relevant if model_name is "pinn_ff")
    init_activ_func: Optional[str] = None  # Only relevant for "pinn_ff"       # sin , tanh, gelu
    subseq_activ_func: Optional[str] = None  # Only relevant for "pinn_ff"     # tanh, gelu

    # Optimizer parameters
    optimizer: str = "adam"  # Options: "adam", "lbfgs"
    learning_rate: Optional[float] = field(default=None)  # Only relevant for Adam
    batch_size: Optional[int] = field(default=None)  # Only relevant for Adam

    # LBFGS-specific parameters
    max_iter: Optional[int] = field(default=None)  # Only relevant for LBFGS
    line_search_fn: Optional[str] = field(default=None)  # Only relevant for LBFGS

    # Normalization parameters
    normalize_res: bool = False
    normalize_ic: bool = False
    alpha: Optional[float] = None  # Only relevant if normalize_res or normalize_ic is True
    epsilon: Optional[float] = None  # Only relevant if normalize_res or normalize_ic is True

    def validate(self):
        """
        Validate the configuration to ensure all conditional parameters are set correctly.
        """
        # Validate optimizer-specific parameters
        if self.optimizer == "adam":
            if self.learning_rate is None:
                raise ValueError("`learning_rate` must be specified when optimizer is 'adam'.")
            if self.batch_size is None:
                raise ValueError("`batch_size` must be specified when optimizer is 'adam'.")
        elif self.optimizer == "lbfgs":
            if self.max_iter is None:
                raise ValueError("`max_iter` must be specified when optimizer is 'lbfgs'.")
            if self.line_search_fn is None:
                raise ValueError("`line_search_fn` must be specified when optimizer is 'lbfgs'.")

        # Validate model-specific parameters
        if self.model_name == "pinn_ff":
            if self.init_activ_func is None:
                raise ValueError("`init_activ_func` must be specified when model_name is 'pinn_ff'.")
            if self.subseq_activ_func is None:
                raise ValueError("`subseq_activ_func` must be specified when model_name is 'pinn_ff'.")

        # Validate normalization-specific parameters
        if self.normalize_res or self.normalize_ic:
            if self.alpha is None:
                raise ValueError("`alpha` must be specified when normalize_res or normalize_ic is True.")
            if self.epsilon is None:
                raise ValueError("`epsilon` must be specified when normalize_res or normalize_ic is True.")

    def to_dict(self):
        """
        Convert the dataclass to a dictionary for use with wandb or other logging tools.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}