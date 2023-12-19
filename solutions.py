# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
from pathlib import Path
from jaxtyping import Float
from typing import Optional, Union, Callable
from tqdm.auto import tqdm
from dataclasses import dataclass

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent

from plotly_utils import imshow, line, hist
from plotly_utils_toy_models import (
    plot_features_in_2d,
    plot_features_in_Nd,
    plot_correlated_features,
    plot_feature_geometry,
    frac_active_line_plot,
)

import tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%




# ======================================================
# ! 1 - TMS: Superposition in a nonprivileged basis
# ======================================================


def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device = device,
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None: feature_probability = t.ones(())
        if isinstance(feature_probability, float): feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_instances, cfg.n_features))
        if importance is None: importance = t.ones(())
        if isinstance(importance, float): importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_instances, cfg.n_features))

        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        return F.relu(out + self.b_final)


    def generate_batch(self, batch_size) -> Float[Tensor, "batch_size instances features"]:
        '''
        Generates a batch of data. We'll return to this function later when we apply correlations.
        '''
        pass # See below for solutions


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `model.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        '''
        pass # See below for solutions


    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        '''
        Optimizes the model using the given hyperparameters.
        '''
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item()/self.cfg.n_instances, lr=step_lr)


if MAIN:
    tests.test_model(Model)


# %%

def generate_batch(self: Model, batch_size) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of data. We'll return to this function later when we apply correlations.
    '''
    # Generate the features, before randomly setting some to zero
    feat = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)

    # Generate a random boolean array, which is 1 wherever we'll keep a feature, and zero where we'll set it to zero
    feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability

    # Create our batch from the features, where we set some to zero
    batch = t.where(feat_is_present, feat, 0.0)
    
    return batch

Model.generate_batch = generate_batch

if MAIN:
    tests.test_generate_batch(Model)


# %%

def calculate_loss(
    self: Model,
    out: Float[Tensor, "batch instances features"],
    batch: Float[Tensor, "batch instances features"],
) -> Float[Tensor, ""]:
    '''
    Calculates the loss for a given batch, using this loss described in the Toy Models paper:

        https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

    Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
    '''
    error = self.importance * ((batch - out) ** 2)
    loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
    return loss

Model.calculate_loss = calculate_loss

if MAIN:
    tests.test_calculate_loss(Model)

# %%

if MAIN:
    cfg = Config(
        n_instances = 8,
        n_features = 5,
        n_hidden = 2,
    )

    # importance varies within features for each instance
    importance = (0.9 ** t.arange(cfg.n_features))
    importance = einops.rearrange(importance, "features -> () features")

    # sparsity is the same for all features in a given instance, but varies over instances
    feature_probability = (50 ** -t.linspace(0, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize(steps=10_000)

    plot_features_in_2d(
        values = model.W.detach(),
        colors = model.importance,
        title = "Superposition: 5 features represented in 2D space",
        subplot_titles = [f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
    )

# %%

if MAIN:
    with t.inference_mode():
        batch = model.generate_batch(250)
        hidden = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> instances hidden batch_size")

    plot_features_in_2d(
        values = hidden,
        title = "Hidden state representation of a random batch of data",
    )


# %%

if MAIN:
    cfg = Config(
        n_instances = 20,
        n_features = 100,
        n_hidden = 20,
    )

    importance = (100 ** -t.linspace(0, 1, cfg.n_features))
    importance = einops.rearrange(importance, "features -> () features")

    feature_probability = (20 ** -t.linspace(0, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize(steps=10_000)

    plot_features_in_Nd(
        values = model.W[::2], # plot every other instance
        height = 1600,
        width = 800,
    )


# %%





# ======================================================
# ! 2 - TMS: Correlated / Anticorrelated features
# ======================================================


def generate_correlated_features(self: Model, batch_size, n_correlated_pairs) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_correlated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= self.feature_probability[:, [0]]
    feat_is_present = einops.repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
    return t.where(feat_is_present, feat, 0.0)


def generate_anticorrelated_features(self: Model, batch_size, n_anticorrelated_pairs) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of anti-correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
    first_feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, [0]]
    first_feat_is_present = first_feat_seeds <= 0.5
    first_feats = t.where(feat_set_is_present & first_feat_is_present, feat[:, :, :n_anticorrelated_pairs], 0.0)
    second_feats = t.where(feat_set_is_present & (~first_feat_is_present), feat[:, :, n_anticorrelated_pairs:], 0.0)
    return einops.rearrange(t.concat([first_feats, second_feats], dim=-1), "batch instances (pair features) -> batch instances (features pair)", pair=2)


def generate_uncorrelated_features(self: Model, batch_size, n_uncorrelated) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of uncorrelated features.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability[:, [0]]
    return t.where(feat_is_present, feat, 0.0)


def generate_batch(self: Model, batch_size):
    '''
    Generates a batch of data, with optional correslated & anticorrelated features.
    '''
    n_uncorrelated = self.cfg.n_features - 2 * self.cfg.n_correlated_pairs - 2 * self.cfg.n_anticorrelated_pairs
    data = []
    if self.cfg.n_correlated_pairs > 0:
        data.append(self.generate_correlated_features(batch_size, self.cfg.n_correlated_pairs))
    if self.cfg.n_anticorrelated_pairs > 0:
        data.append(self.generate_anticorrelated_features(batch_size, self.cfg.n_anticorrelated_pairs))
    if n_uncorrelated > 0:
        data.append(self.generate_uncorrelated_features(batch_size, n_uncorrelated))
    batch = t.cat(data, dim=-1)
    return batch


Model.generate_correlated_features = generate_correlated_features
Model.generate_anticorrelated_features = generate_anticorrelated_features
Model.generate_uncorrelated_features = generate_uncorrelated_features
Model.generate_batch = generate_batch

if MAIN:
    cfg = Config(
        n_instances = 30,
        n_features = 4,
        n_hidden = 2,
        n_correlated_pairs = 1,
        n_anticorrelated_pairs = 1,
    )

    feature_probability = 10 ** -t.linspace(0.5, 1, cfg.n_instances).to(device)

    model = Model(
        cfg = cfg,
        device = device,
        feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")
    )

    # Generate a batch of 4 features: first 2 are correlated, second 2 are anticorrelated
    batch = model.generate_batch(batch_size=100_000)
    corr0, corr1, anticorr0, anticorr1 = batch.unbind(dim=-1)
    corr0_is_active = corr0 != 0
    corr1_is_active = corr1 != 0
    anticorr0_is_active = anticorr0 != 0
    anticorr1_is_active = anticorr1 != 0

    assert (corr0_is_active == corr1_is_active).all(), "Correlated features should be active together"
    assert (corr0_is_active.float().mean(0) - feature_probability).abs().mean() < 0.002, "Each correlated feature should be active with probability `feature_probability`"

    assert (anticorr0_is_active & anticorr1_is_active).int().sum().item() == 0, "Anticorrelated features should never be active together"
    assert (anticorr0_is_active.float().mean(0) - feature_probability).abs().mean() < 0.002, "Each anticorrelated feature should be active with probability `feature_probability`"

    # Generate a batch of 4 features: first 2 are correlated, second 2 are anticorrelated
    batch = model.generate_batch(batch_size = 1)
    correlated_feature_batch, anticorrelated_feature_batch = batch[:, :, :2], batch[:, :, 2:]

    # Plot correlated features
    plot_correlated_features(correlated_feature_batch, title="Correlated Features: should always co-occur")
    plot_correlated_features(anticorrelated_feature_batch, title="Anti-correlated Features: should never co-occur")


# %%

if MAIN:
    cfg = Config(
        n_instances = 5,
        n_features = 4,
        n_hidden = 2,
        n_correlated_pairs = 2,
        n_anticorrelated_pairs = 0,
    )

    # All same importance, very low feature probabilities (ranging from 5% down to 0.25%)
    importance = t.ones(cfg.n_features, dtype=t.float, device=device)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (400 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize()

    plot_features_in_2d(
        values = model.W.detach(),
        colors = ["blue"] * 2 + ["limegreen"] * 2, # when colors is a list of strings, it's assumed to be the colors of features
        title = "Correlated feature sets are represented in local orthogonal bases",
        subplot_titles = [f"1 - S = {i:.3f}" for i in model.feature_probability[:, 0]],
    )

# %%

if MAIN:

    # Anticorrelated feature pairs

    cfg = Config(
        n_instances = 5,
        n_features = 4,
        n_hidden = 2,
        n_correlated_pairs = 0,
        n_anticorrelated_pairs = 2,
    )
    # All same importance, not-super-low feature probabilities (all >10%)
    importance = t.ones(cfg.n_features, dtype=t.float, device=device)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (10 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize()

    plot_features_in_2d(
        values = model.W.detach(),
        colors = ["red"] * 2 + ["orange"] * 2,
        title = "Anticorrelated feature sets are frequently represented as antipodal pairs",
        subplot_titles = [f"1 - S = {i:.3f}" for i in model.feature_probability[:, 0]],
    )


# %%

if MAIN:
    # 3 correlated feature pairs

    cfg = Config(
        n_instances = 5,
        n_features = 6,
        n_hidden = 2,
        n_correlated_pairs = 3,
        n_anticorrelated_pairs = 0,
    )
    # All same importance, very low feature probabilities (ranging from 5% down to 0.25%)
    importance = t.ones(cfg.n_features, dtype=t.float, device=device)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (100 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize()
    
    plot_features_in_2d(
        values = model.W.detach(),
        colors = ["blue"] * 2 + ["limegreen"] * 2 + ["purple"] * 2,
        title = "Correlated feature sets are represented in local orthogonal bases",
        subplot_titles = [f"1 - S = {i:.3f}" for i in model.feature_probability[:, 0]],
    )


# %%

# ======================================================
# ! 3 - TMS: Superposition in a Privileged Basis
# ======================================================





class NeuronModel(Model):
    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device=device
    ):
        super().__init__(cfg, feature_probability, importance, device)

    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        activations = F.relu(einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        ))
        out = F.relu(einops.einsum(
            activations, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        ) + self.b_final)
        return out
    

if MAIN:
    n_features = 10
    n_hidden = 5

    importance = einops.rearrange(0.75 ** t.arange(1, 1+n_features), "feats -> () feats")
    feature_probability = einops.rearrange(t.tensor([0.75, 0.35, 0.15, 0.1, 0.06, 0.02, 0.01]), "instances -> instances ()")

    cfg = Config(
        n_instances = len(feature_probability.squeeze()),
        n_features = n_features,
        n_hidden = n_hidden,
    )

    model = NeuronModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize(steps=10_000)

    plot_features_in_Nd(
        W = model.W,
        height = 600,
        width = 1000,
        title = "Neuron model: n_features = 10, d_hidden = 5, I<sub>i</sub> = 0.75<sup>i</sup>",
        subplot_titles = [f"1 - S = {i:.2f}" for i in feature_probability.squeeze()],
        neuron_plot = True,
    )


# %%


class NeuronComputationModel(Model):
    W1: Float[Tensor, "n_instances n_hidden n_features"]
    W2: Float[Tensor, "n_instances n_features n_hidden"]
    b_final: Float[Tensor, "n_instances n_features"]

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device=device
    ):
        super().__init__(cfg, feature_probability, importance, device)

        del self.W
        self.W1 = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.W2 = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_features, cfg.n_hidden))))
        self.to(device)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        activations = F.relu(einops.einsum(
           features, self.W1,
           "... instances features, instances hidden features -> ... instances hidden"
        ))
        out = F.relu(einops.einsum(
            activations, self.W2,
            "... instances hidden, instances features hidden -> ... instances features"
        ) + self.b_final)
        return out
    

    def generate_batch(self, batch_size) -> Tensor:
        feat = 2 * t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W1.device) - 1
        feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W1.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(feat_is_present, feat, 0.0)
        return batch


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        error = self.importance * ((batch.abs() - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss


if MAIN:
    n_features = 100
    n_hidden = 40

    importance = einops.rearrange(0.8 ** t.arange(1, 1+n_features), "feats -> () feats")
    feature_probability = einops.rearrange(t.tensor([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]), "instances -> instances ()")

    cfg = Config(
        n_instances = len(feature_probability.squeeze()),
        n_features = n_features,
        n_hidden = n_hidden,
    )

    model = NeuronComputationModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize(steps=30_000)
        
    plot_features_in_Nd(
        W = model.W1,
        height = 1200,
        width = 700,
        title = f"Neuron computation model: n_features = {n_features}, d_hidden = {n_hidden}, I<sub>i</sub> = 0.75<sup>i</sup>",
        subplot_titles = [f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
        neuron_plot = True,
        show_heatmaps = False,
    )


# %%



# ======================================================
# ! 4 - TMS: Feature Geometry
# ======================================================
    
@t.inference_mode()
def compute_dimensionality(
    W: Float[Tensor, "n_instances n_hidden n_features"]
) -> Float[Tensor, "n_instances n_features"]:

    # Compute numerator terms
    W_norms = W.norm(dim=1, keepdim=True)
    numerator = W_norms.squeeze() ** 2

    # Compute denominator terms
    W_normalized = W / (W_norms + 1e-8)
    denominator = einops.einsum(W_normalized, W, "i h f1, i h f2 -> i f1 f2").pow(2).sum(-1)

    return numerator / denominator

# %%



# ======================================================
# ! 5 - SAEs in Toy Models
# ======================================================

@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
        self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))
        self.to(device)

    def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):

        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.transpose(-1, -2) if self.cfg.tied_weights else self.W_dec),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean(-1) # shape [batch_size n_instances]
        l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @t.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        if self.cfg.tied_weights:
            self.W_enc.data = self.W_enc.data / self.W_enc.data.norm(dim=1, keepdim=True)
        else:
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)

    @t.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size n_instances n_hidden"],
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        pass # See below for a solution to this function

    def optimize(
        self,
        model: Model,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        '''
        Optimizes the autoencoder using the given hyperparameters.

        This function should take a trained model as input.
        '''
        if neuron_resample_window is not None:
            assert (dead_neuron_window is not None) and (dead_neuron_window < neuron_resample_window)

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"values": [], "colors": [], "titles": [], "frac_active": []}
        colors = None
        title = "no resampling yet"

        for step in progress_bar:

            # Normalize the decoder weights before each optimization step
            self.normalize_decoder()

            # Resample dead neurons
            if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
                # Get the fraction of neurons active in the previous window
                frac_active_in_window = t.stack(frac_active_list[-neuron_resample_window:], dim=0)
                # Compute batch of hidden activations which we'll use in resampling
                batch = model.generate_batch(batch_size)
                h = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> batch_size instances hidden")
                # Resample
                colors, title = self.resample_neurons(h, frac_active_in_window, neuron_resample_scale)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                features = model.generate_batch(batch_size)
                h = einops.einsum(features, model.W, "... instances features, instances hidden features -> ... instances hidden")

            # Optimize
            optimizer.zero_grad()
            l1_loss, l2_loss, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()

            # Calculate the sparsities, and add it to a list
            frac_active = einops.reduce((acts.abs() > 1e-8).float(), "batch_size instances hidden_ae -> instances hidden_ae", "mean")
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(l1_loss=l1_loss.mean(0).sum().item(), l2_loss=l2_loss.mean(0).sum().item(), lr=step_lr)
                data_log["values"].append(self.W_enc.detach().cpu())
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}: {title}")
                data_log["frac_active"].append(frac_active.detach().cpu())

        return data_log
    
# %%

# ======================================================
# ! 6 - SAEs in Language Models
# ======================================================
