# %%

# ! Note - these solutions are a bit out-of-date. The tests (which import the solutions) should all work, but the solutions haven't been updated
# ! to be consistent with the new contents of the Colab (e.g. SAE exercises don't yet have solutions here). These will be added shortly.

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
# assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part7_toy_models_of_superposition', not '{section_dir}'"
# if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line

# from utils1 import plot_W, plot_Ws_from_model, render_features, plot_feature_geometry
import tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%


if MAIN:
	W = t.randn(2, 5)
	W_normed = W / W.norm(dim=0, keepdim=True)
	
	imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600)

# %%


# if MAIN:
# 	plot_W(W_normed)

# %%

@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as 
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    # Ignore the correlation arguments for now.
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


def linear_lr(step, steps):
	return (1 - (step / steps))
	
def constant_lr(*_):
	return 1.0
	
def cosine_decay_lr(step, steps):
	return np.cos(0.5 * np.pi * step / (steps - 1))



class Model(nn.Module):

    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map (ignoring n_instances) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device = device,
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None: feature_probability = t.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None: importance = t.ones(())
        self.importance = importance.to(device)

        self.W = nn.Parameter(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features), device=device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features), device=device))



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
        feat = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(
            feat_is_present,
            feat,
            t.zeros((), device=self.W.device),
        )
        return batch


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
        # SOLUTION
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss


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
        optimizer = t.optim.AdamW(list(self.parameters()), lr=lr)

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
	tests.test_generate_batch(Model)
	tests.test_calculate_loss(Model)

# %%



if MAIN:
	cfg = Config(
		n_instances = 10,
		n_features = 5,
		n_hidden = 2,
	)
	
	importance = (0.9**t.arange(cfg.n_features))

	feature_probability = (20 ** -t.linspace(0, 1, cfg.n_instances))
	
	line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})

	line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})

# %%


if MAIN:
	model = Model(
		cfg=cfg,
		device=device,
		importance=importance[None, :],
		feature_probability=feature_probability[:, None]
	)
	model.optimize()

# %%


# if MAIN:
# 	plot_Ws_from_model(model, cfg)

# %% VISUALIZING FEATURES ACROSS VARYING SPARSITY


if MAIN:
	cfg = Config(
		n_instances = 20,
		n_features = 100,
		n_hidden = 20,
	)
	
	importance = (100 ** -t.linspace(0, 1, cfg.n_features))

	feature_probability = (20 ** -t.linspace(0, 1, cfg.n_instances))
	
	line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})

	line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})

# %%


if MAIN:
	model = Model(
		cfg=cfg,
		device=device,
		importance = importance[None, :],
		feature_probability = feature_probability[:, None]
	)


if MAIN:
	model.optimize()

# %%


# if MAIN:
# 	fig = render_features(model, np.s_[::2])
# 	fig.update_layout(width=1200, height=2000)

# %%

def generate_correlated_batch(self: Model, batch_size: int) -> Float[Tensor, "batch_size instances fetures"]:
    '''
    Generates a batch of data.

    There are `n_correlated_pairs` pairs of correlated features (i.e. they always co-occur), and 
    `n_anticorrelated` pairs of anticorrelated features (i.e. they never co-occur; they're
    always opposite).

    So the total number of features defined this way is `2 * n_correlated_pairs + 2 * n_anticorrelated`.

    You should stack the features in the order (correlated, anticorrelated, uncorrelated), where
    the uncorrelated ones are all the remaining features.

    Note, we assume the feature probability varies across instances but not features, i.e. all features
    in each instance have the same probability of being present.
    '''
    n_correlated_pairs = self.cfg.n_correlated_pairs
    n_anticorrelated_pairs = self.cfg.n_anticorrelated_pairs

    n_uncorrelated = self.cfg.n_features - 2 * (n_correlated_pairs + n_anticorrelated_pairs)
    assert n_uncorrelated >= 0, "Need to have number of paired correlated + anticorrelated features <= total features"
    assert self.feature_probability.shape == (self.cfg.n_instances, 1), "Feature probability should not vary across features in a single instance."

    # Define uncorrelated features, the standard way
    feat = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability
    batch_uncorrelated = t.where(
        feat_is_present,
        feat,
        t.zeros((), device=self.W.device),
    )

    # SOLUTION
    # Define correlated features: have the same sample determine if they're zero or not
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_correlated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= self.feature_probability
    feat_is_present = einops.repeat(
        feat_set_is_present,
        "batch instances features -> batch instances (features pair)", pair=2
    )
    batch_correlated = t.where(
        feat_is_present, 
        feat,
        t.zeros((), device=self.W.device),
    )

    # Define anticorrelated features: have them all be zero with probability `feature_probability`, and
    # have a single feature randomly chosen if they aren't all zero
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs), device=self.W.device)
    # First, generate seeds (both for entire feature set, and for features within the set)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
    first_feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
    # Create boolean mask for whether the entire set is zero
    # Note: the *2 here didn't seem to be used by the paper, but it makes more sense imo! You can leave it out and still get good results.
    feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability
    # Where it's not zero, create boolean mask for whether the first element is zero
    first_feat_is_present = first_feat_seeds <= 0.5
    # Now construct our actual features and stack them together, then rearrange
    first_feats = t.where(
        feat_set_is_present & first_feat_is_present, 
        feat[:, :, :n_anticorrelated_pairs],
        t.zeros((), device=self.W.device)
    )
    second_feats = t.where(
        feat_set_is_present & (~first_feat_is_present), 
        feat[:, :, n_anticorrelated_pairs:],
        t.zeros((), device=self.W.device)
    )
    batch_anticorrelated = einops.rearrange(
        t.concat([first_feats, second_feats], dim=-1),
        "batch instances (pair features) -> batch instances (features pair)", pair=2
    )

    return t.concat([batch_correlated, batch_anticorrelated, batch_uncorrelated], dim=-1)


if MAIN:
	Model.generate_batch = generate_correlated_batch

# %%

if MAIN:
	cfg = Config(
		n_instances = 10,
		n_features = 4,
		n_hidden = 2,
		n_correlated_pairs = 1,
		n_anticorrelated_pairs = 1,
	)

	importance = t.ones(cfg.n_features, dtype=t.float, device=device)
	feature_probability = (20 ** -t.linspace(0, 1, cfg.n_instances))

	model = Model(
		cfg=cfg,
		device=device,
		importance=importance[None, :],
		feature_probability=feature_probability[:, None]
	)

	batch = model.generate_batch(batch_size = 1)

	imshow(
		batch.squeeze(),
		labels={"x": "Feature", "y": "Instance"}, 
		title="Feature heatmap (first two features correlated, last two anticorrelated)"
	)

# %%

if MAIN:
	feature_probability = (20 ** -t.linspace(0.5, 1, cfg.n_instances))
	model.feature_probability = feature_probability[:, None].to(device)

	batch = model.generate_batch(batch_size = 10000)

	corr0, corr1, anticorr0, anticorr1 = batch.unbind(dim=-1)
	corr0_is_active = corr0 != 0
	corr1_is_active = corr1 != 0
	anticorr0_is_active = anticorr0 != 0
	anticorr1_is_active = anticorr1 != 0

	assert (corr0_is_active == corr1_is_active).all(), "Correlated features should be active together"
	assert (corr0_is_active.float().mean(0).cpu() - feature_probability).abs().mean() < 0.01, "Each correlated feature should be active with probability `feature_probability`"

	assert (anticorr0_is_active & anticorr1_is_active).int().sum().item() == 0, "Anticorrelated features should never be active together"
	assert (anticorr0_is_active.float().mean(0).cpu() - feature_probability).abs().mean() < 0.01, "Each anticorrelated feature should be active with probability `feature_probability`"



# %%

if MAIN:
	cfg = Config(
		n_instances = 5,
		n_features = 4,
		n_hidden = 2,
		n_correlated_pairs = 2,
		n_anticorrelated_pairs = 0,
	)

	# All same importance
	importance = t.ones(cfg.n_features, dtype=t.float, device=device)
	# We use very low feature probabilities, from 5% down to 0.25%
	feature_probability = (400 ** -t.linspace(0.5, 1, 5))

	model = Model(
		cfg=cfg,
		device=device,
		importance=importance[None, :],
		feature_probability=feature_probability[:, None]
	)

	model.optimize()

	# plot_Ws_from_model(model, cfg)

# %%

if MAIN:
	cfg = Config(
		n_instances = 5,
		n_features = 4,
		n_hidden = 2,
		n_correlated_pairs = 0,
		n_anticorrelated_pairs = 2,
	)

	# All same importance
	importance = t.ones(cfg.n_features, dtype=t.float, device=device)
	# We use very low feature probabilities, from 5% down to 0.25%
	feature_probability = (400 ** -t.linspace(0.5, 1, 5))

	model = Model(
		cfg=cfg,
		device=device,
		importance=importance[None, :],
		feature_probability=feature_probability[:, None]
	)

	model.optimize()

	# plot_Ws_from_model(model, cfg)

# %%

if MAIN:
	cfg = Config(
		n_instances = 5,
		n_features = 6,
		n_hidden = 2,
		n_correlated_pairs = 3,
		n_anticorrelated_pairs = 0,
	)

	# All same importance
	importance = t.ones(cfg.n_features, dtype=t.float, device=device)
	# We use very low feature probabilities, from 5% down to 0.25%
	feature_probability = (400 ** -t.linspace(0.5, 1, 5))

	model = Model(
		cfg=cfg,
		device=device,
		importance=importance[None, :],
		feature_probability=feature_probability[:, None]
	)

	model.optimize()

	# plot_Ws_from_model(model, cfg)

# %%

if MAIN:
	cfg = Config(
		n_features = 200,
		n_hidden = 20,
		n_instances = 20,
	)
	
	feature_probability = (20 ** -t.linspace(0, 1, cfg.n_instances))
	
	model = Model(
		cfg=cfg,
		device=device,
		# For this experiment, use constant importance.
		feature_probability = feature_probability[:, None]
	)


# if MAIN:
# 	model.optimize()
# 	plot_feature_geometry(model)

# %%

@t.inference_mode()
def compute_dimensionality(
    W: Float[Tensor, "n_instances n_hidden n_features"]
) -> Float[Tensor, "n_instances n_features"]:

    # Compute numerator terms
    W_norms = W.norm(dim=1, keepdim=True)
    numerator = W_norms.squeeze() ** 2

    # Compute denominator terms
    W_normalized = W / W_norms
    # t.clamp(W_norms, 1e-6, float("inf"))
    denominator = einops.einsum(W_normalized, W, "i h f1, i h f2 -> i f1 f2").pow(2).sum(-1)

    return numerator / denominator



if MAIN:
	W = model.W.detach()
	dim_fracs = compute_dimensionality(W)

	# plot_feature_geometry(model, dim_fracs=dim_fracs)

# %%



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

    def generate_batch(self, batch_size) -> Tensor:
        feat = 2 * t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device) - 1
        feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(
            feat_is_present,
            feat,
            t.zeros((), device=self.W.device),
        )
        return batch
    

def calculate_neuron_loss(
    out: Float[Tensor, "batch instances features"],
    batch: Float[Tensor, "batch instances features"],
    model: Model
) -> Float[Tensor, ""]:
    error = model.importance * ((batch.abs() - out) ** 2)
    loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
    return loss


# def optimize(
#     model: Union[Model, NeuronModel], 
#     batch_size: int = 1024,
#     steps: int = 10_000,
#     log_freq: int = 100,
#     lr: float = 1e-3,
#     lr_scale: Callable = constant_lr,
# ):
#     '''
#     Optimizes the model using the given hyperparameters.
    
#     This version can accept either a Model or NeuronModel instance.
#     '''
#     cfg = model.cfg

#     optimizer = t.optim.AdamW(list(model.parameters()), lr=lr)

#     progress_bar = tqdm(range(steps))
#     for step in progress_bar:
#         step_lr = lr * lr_scale(step, steps)
#         for group in optimizer.param_groups:
#             group['lr'] = step_lr
#             optimizer.zero_grad()
#             batch = model.generate_batch(batch_size)
#             out = model(batch)
#             if isinstance(model, NeuronModel):
#                 loss = calculate_neuron_loss(out, batch, model)
#             else:
#                 loss = calculate_loss(out, batch, model)
#             loss.backward()
#             optimizer.step()

#             if step % log_freq == 0 or (step + 1 == steps):
#                 progress_bar.set_postfix(loss=loss.item()/cfg.n_instances, lr=step_lr)

                

# for n_features in [5, 6, 8]:

#     cfg = Config(
#         n_instances = 1,
#         n_features = n_features,
#         n_hidden = 5,
#     )

#     model = NeuronModel(
#         cfg=cfg,
#         device=device,
#         feature_probability=t.ones(model.cfg.n_instances, device=device)[:, None],
#     )

#     optimize(model, steps=1000)

#     W = model.W[0]
#     W_normed = W / W.norm(dim=0, keepdim=True)
#     imshow(W_normed.T, width=600, color_continuous_scale="RdBu_r", zmin=-1.4, zmax=1.4)

# %%
