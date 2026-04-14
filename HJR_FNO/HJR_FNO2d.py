# =========================
# Standard library imports
# =========================
import os
import time
import math
import operator
import warnings
from functools import reduce, partial
from timeit import default_timer
from pathlib import Path
from typing import Tuple, List, Dict, Union

warnings.filterwarnings("ignore")

# =========================
# Third-party imports
# =========================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.ndimage import gaussian_filter

# =========================
# PyTorch imports
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset

# =========================
# Local / project imports
# =========================
from .neural_utils import *


################################################################
#  2D Fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        modes1: number of Fourier modes to keep along x-dimension
        modes2: number of Fourier modes to keep along y-dimension
                (at most floor(N/2) + 1 for each spatial dim)
        """
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)

        # Two weight tensors to cover both frequency corners in 2D.
        # Shape: (in_ch, out_ch, modes1, modes2) — complex valued.
        #   weights1 → top-left  corner: (+kx, +ky)
        #   weights2 → bottom-left corner: (-kx, +ky)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                    dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_ch, x, y), (in_ch, out_ch, x, y) -> (batch, out_ch, x, y)
        # Contracts over in_ch ('i'), keeps both spatial freq axes ('x','y')
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x shape: (batch, in_ch, X, Y)

        # 1. 2D real FFT → (batch, in_ch, X, Y//2+1)
        #    rfft2 halves only the LAST axis (y), giving the non-redundant
        #    half of the conjugate-symmetric 2D spectrum.
        x_ft = torch.fft.rfft2(x)

        # 2. Allocate output frequency tensor (same half-spectrum shape)
        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-2), x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # 3. Multiply the two low-frequency corners by learned weights.
        #
        #    Top-left    [:, :, :modes1,  :modes2] — low +kx, low +ky
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],
                             self.weights1)

        #    Bottom-left [:, :, -modes1:, :modes2] — low -kx, low +ky
        #    This captures the low-|kx| negative-frequency content that
        #    rfft2 retains in the upper rows of the x-axis spectrum.
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                             self.weights2)

        # 4. Inverse 2D real FFT back to physical space.
        #    s=(x.size(-2), x.size(-1)) ensures the output spatial size
        #    matches the input exactly, compensating for the halved y-axis.
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


################################################################
#  FNO2d — 4-layer Fourier Neural Operator for 2D problems
################################################################
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.lifting.
        2. 4 layers of the integral operators u' = (W + K)(u).
               W defined by self.w*  (pointwise Conv2d, kernel=1)
               K defined by self.conv*  (SpectralConv2d)
        3. Project from channel space to output space by self.projection.

        input:  solution + coordinates (a(x,y), x, y)  — arbitrary c_in channels
        input shape:  (batch, X, Y, c_in)
        output: solution at a later time / quantity of interest
        output shape: (batch, X, Y, 1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width

        # ------- Lifting -------
        # Maps c_in channels → width.  Conv2d(kernel=1) = pointwise operation,
        # identical in role to the 1D version but now over a 2D spatial grid.
        # Adjust in_channels (here 3: a(x,y), x, y) to match your actual input.
        self.lifting    = nn.Conv2d(4, self.width, 1)

        # ------- Fourier blocks -------
        # Spectral (K) paths
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Bypass (W) paths — pointwise Conv2d instead of Conv1d
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # ------- Projection -------
        self.projection = nn.Conv2d(self.width, 1, 1)

    def forward(self, x):
        # x: (batch, X, Y, c_in)

        # --- Lifting ---
        # Permute to channel-first for Conv2d: (batch, c_in, X, Y)
        x = x.permute(0, 3, 1, 2)
        x = self.lifting(x)                          # → (batch, width, X, Y)

        # --- Fourier Block 0 ---
        x1 = self.conv0(x)                           # spectral path  K(u)
        x2 = self.w0(x)                              # bypass path    W(u)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Fourier Block 1 ---
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Fourier Block 2 ---
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Fourier Block 3 ---
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x  = x1 + x2
        x  = F.relu(x)

        # --- Projection ---
        x = self.projection(x)                       # → (batch, 1, X, Y)
        x = x.permute(0, 2, 3, 1)                   # → (batch, X, Y, 1)
        return x