## torch-nac

This repository contains PyTorch implementations of the following research papers:

- FLUID: Continuous-Time Hyperconnected Sparse Transformer for Sink-Free Learning
- Neuronal Attention Circuit (NAC) for Representation Learning
- Neuronal Stochastic Attention Circuit (NSAC) for Probabilistic Representation Learning
---

### Installation

```bash
pip install torch-nac
```

---

### Requirements

- Python >= 3.10
- Pytorch >= 2.0

---

## Usage Examples [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11uou3vn_b0WeAFA0IIz5uO4m8GKxrlS-?usp=sharing)

These layers can be used as drop-in components inside PyTorch models.

---

### 1. Liquid Attention Network (LAN)

```python
import torch
from torch_nac import layers

class LAN_Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.lan = layers.LAN(
            input_dim=1,                # Dimension of the input features
            d_model=64,                 # Dimension of the model of LAN
            num_heads=16,               # Number of attention heads of LAN
            topk=8,                     # Number of top-k attention interactions
            euler_steps=6,              # Number of Euler steps 
            activation="sigmoid",       # Activation function
            use_sink_gate=True,         # Use Attention Sink Gate
            return_sequences=False,     # Return full sequences if True, else last output
            return_attention=False      # Return attention weights if True
            )

        self.out = torch.nn.Linear(64, 1)

    # call method
    def forward(self, x):
        x = self.lan(x)
        return self.out(x)


model = LAN_Model().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,)

print(model)
```

---

### 2. FLUID Transformer

```python
import torch
from torch_nac import layers

class FLUID_Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fluid = layers.FLUID(
            input_dim=1,                # Dimension of the input features
            d_model=64,                 # Dimension of the model of LAN
            num_heads=16,               # Number of attention heads of LAN
            num_layers=1,               # Number of stacked encoder/decoder layers
            ff_dim=32,                  # Dimension of the feed-forward network
            delta_t= 0.01,              # Time-step for the Liquid Attention
            euler_steps=5,              # Number of Euler steps for Liquid Attention
            topk=8,                     # Number of top-k attention interactions
            expansion_rate=2,           # Expansion factor for feed-forward layers
            use_sink_gate=True,         # Enable sink gate mechanism
            use_pairwise=False,         # disable top-k sparsity if True
            enable_hc=True,             # Enable hyper-connections if True, Otherwise -> Residual connections
            dynamic_hc=True,            # Enable Liquid hyper-connections if True, Otherwise -> Static
            dropout=0.0,                # Dropout rate
            max_len=1000,               # Maximum sequence length of positional encoder
            return_attention=False,     # Return attention weights if True
        )
        self.actv = nn.Sigmoid()
        self.flat = nn.Flatten()
        self.out = torch.nn.Linear(64, 1)

    # call method
    def forward(self, x):
        x = self.fluid(x)
        x = self.actv(x)
        x = self.flat(x)
        return self.out(x)


model = FLUID_Model().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,)

print(model)
```

---

### 3. Neuronal Attention Circuit (NAC)

```python
import torch
from torch_nac import layers

class NAC_Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        self.nac = layers.NAC(
            input_dim=1,                # Dimension of the input features
            d_model=64,                  # Dimension of the model
            num_heads=16,                # Number of attention heads
            mode='exact',                # Computation mode: 'exact', 'euler', or 'steady'
            topk=8,                      # Number of top-k pairwise interactions
            delta_t=0.5,                 # Time step for Euler mode
            sparsity=0.5,                # Sparsity level for NCP wiring
            euler_steps=6,               # Number of Euler integration steps
            dropout=0.0,                 # Dropout rate
            tau_epsilon=1e-5,            # Small positive value for temporal head
            activation='sigmoid',        # Activation function
            use_riemann_sum=True,         # Use Reimann-sum integration if True, else standard weighted sum
            return_sequences=False,      # Return full sequences if True, else last output
            return_attention=False,      # Return attention weights if True
            return_cell_state=False,      # Return cell-level state  if True
        )
        self.out = torch.nn.Linear(64, 1)

    # call method
    def forward(self, x):
        x = self.nac(x)
        return self.out(x)

model = NAC_Model().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,)

print(model)
```

---

### 4. Neuronal Stochastic Attention Circuit (NSAC)

```python
import torch 
from torch_nac import layers, models, losses

# Stochastic function
class Stochastic_Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = layers.OUWrap(
            layers.NAC(input_dim=1, d_model=64, num_heads=16, topk=8, sparsity=0.5),
            output_dim=1,                # Output dimension for regression 
            bn_mean=0.0,                 # Brownian mean
            bn_std=0.1,                  # Brownian standard deviation
            activation='sigmoid',        # Activation function
            return_sequences=False,      # Return full sequences if True, else last output
            return_attention=False,      # Return attention weights if True
            return_cell_state=False,     # Return cell potentials if True
        )

    def forward(self, x, training=None):
        return self.model(x)

# NSAC model
class NSAC_Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.nsac = models.NSAC(
            stochastic_model = Stochastic_Model(),
            mc_samples=1,           # Monte-Carlo steps 
            ood_mean=0.0,           # OOD generating noise mean 
            ood_std=5.0             # OOD generating noise standard deviation
        )

    def forward(self, x):
        return self.nsac(x)
    
model = NSAC_Model().to(device)
loss_fn = losses.NSACLoss(lambda_reg=0.5)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,)


print(model)

```

---
## Citation

```bibtex
@article{razzaq2025neuronal,
  title={Neuronal Attention Circuit (NAC) for Representation Learning},
  author={Razzaq, Waleed and Kanjaraway, Izis and Zhao, Yun-Bo},
  journal={arXiv preprint arXiv:2512.10282},
  year={2025}
}

@article{razzaq2026fluid,
  title={FLUID: Continuous-Time Hyperconnected Sparse Transformer for Sink-Free Learning},
  author={Razzaq, Waleed and Zhao, Yun-Bo},
  journal={arXiv preprint arXiv:2605.04421},
  year={2026}
}

@article{razzaq2026neuronal,
  title={Neuronal Stochastic Attention Circuit (NSAC) for Probabilistic Representation Learning},
  author={Razzaq, Waleed and Zhao, Yun-Bo},
  journal={arXiv preprint arXiv:2605.26061},
  year={2026}
}
