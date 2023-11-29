"""Torch implementations of two-area PPC-PFC networks from Murray, Jaramillo, and Wang (2017)"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union


# helper funcs

def current_to_frequency(
    I: torch.Tensor,
    a: float = 270., # Hz / nA
    b: float = 108., # Hz / nA
    c: float = 0.154, # sec
) -> torch.Tensor:
    """Input current to firing rate transfer function.
    Essentially a smoothed ReLU.

    Parameters
    ----------
    I : torch.Tensor
        Tensor of input currents, can be any shape
    a : float, default: 270.0
        Linear scaling applied before smoothed ReLU
    b : float, default: 108.0
        Linear offset applied before smoothed ReLU
    c : float, default: 0.154
        A parameter controlling smoothness of the 
        ReLU, with higher `c` giving less smoothing.
        Must be greater than 0
    """
    assert c > 0, f"`c` must be > 0, got {c}"
    act = a * I - b
    freq = torch.where(
        act == 0,
        1 / (c * torch.exp(-c * act)), # l'hopital's
        act / (1 - torch.exp(-c * act)),
    )
    return freq


# main classes

class MurraySingleArea(nn.Module):
    def __init__(
        self,
        Js: float = 0.35,
        Jt: float = 0.28387,
        dt: float = 0.001,
        I_0: float = 0.334,
        tau_ndma: float = 0.06,
        gamma: float = 0.641,
    ):
        """Single-area two-population model. 
        Defined by

        .. math::
            \dot{S} = -S/\tau_{NDMA} + \gamma (1 - S) f(J \cdot S + I_0 + I_{app})
        
        where $f$ is the f-I transfer function, $J$ is the weight 
        matrix, and $I_{app}$ is external applied input current.
        Note that the noise current is neglected here and
        instead incorporated into task simulation - see `task.py`.
        
        Parameters
        ----------
        Js : float, default: 0.35
            The "structure" of the network, basically
            the difference between the connection strength
            between a population and itself, and the connection
            strength between the two populations.
        Jt : float, default: 0.28387
            The "tone" of the network, equal to the
            sum of the connection strength between a population 
            and itself, and the connection strength between 
            the two populations.
        dt : float, default: 0.001
            The bin size of the inputs and outputs, in seconds.
            This is not necessarily equal to the simulation step 
            size, which can be controlled with `steps_per_bin`
            in the `forward()` call. Must be > 0
        I_0 : float, default: 0.334
            Constant input current bias
        tau_ndma : float, default: 0.06
            Time constant for NDMA synapses, in seconds.
            Essentially the decay timescale of model latent state.
            Must be > 0
        gamma : float, default: 0.641
            Parameter that controls saturation rate of 
            the model latent state. Must be > 0 and < 1
        """
        assert dt > 0, f"`dt` must be > 0, got {dt}"
        assert tau_ndma > 0, f"`tau_ndma` must be > 0, got {tau_ndma}"
        assert gamma > 0 and gamma < 1, f"`gamma` must be between 0 and 1, got {gamma}"
        super().__init__()
        self.J = nn.Parameter(torch.Tensor([
            [(Js + Jt) / 2, (Jt - Js) / 2],
            [(Jt - Js) / 2, (Js + Jt) / 2],
        ]).float())
        self.dt = dt
        self.I_0 = I_0
        self.tau_ndma = tau_ndma
        self.gamma = gamma
        self.batch_first = True

    def forward(
        self,
        input: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        steps_per_bin: int = 1, # integration resolution
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Simulate model forward

        Parameters
        ----------
        input : torch.Tensor
            External applied input current at each timestamp, in nA.
            Should be of shape Batch x Time x 2
        h0 : torch.Tensor, optional
            Initial state of the network. If provided, should be of
            shape 1 x Batch x 2 (to conform with PyTorch RNN conventions).
            If not provided, defaults to zeros
        steps_per_bin : int, default: 1
            Number of integration steps per data bin. Inputs are 
            assumed to stay constant during each bin if multiple
            steps are used.

        Returns
        -------
        states : tuple
            Tuple containing Batch x Time x 2 arrays containing 
            model latent state ($S$) and firing rates ($f(I)$)
            from simulation
        final_state : torch.Tensor
            Final state of the network, to conform with PyTorch
            conventions. Has shape 1 x Batch x 2
        """
        # input: B x T x 2
        # h0: 1 x B x 2
        assert len(input.shape) == 3, f"{input.shape}"
        if h0 is None:
            h0 = torch.zeros((1, input.shape[0], 2))
        else:
            assert len(h0.shape) == 3, f"{h0.shape}"
        input = input.to(self.J.device).float()
        state = h0[0].to(self.J.device).float()
        states = []
        rates = []
        for i in range(input.shape[1]):
            for n in range(steps_per_bin):
                current = state @ self.J + self.I_0 + input[:,i,:]
                rate = current_to_frequency(current)
                state = state + (self.dt / steps_per_bin) * (
                    -state / self.tau_ndma +
                    self.gamma * (1 - state) * rate
                )
            states.append(state)
            rates.append(rate)
        return (
            torch.stack(states, dim=1),
            torch.stack(rates, dim=1)
        ), states[-1].unsqueeze(0)


class MurrayTwoArea(nn.Module):
    def __init__(
        self,
        Js1: float = 0.35,
        Jt1: float = 0.28387,
        Js2: float = 0.4182,
        Jt2: float = 0.28387,
        Js12: float = 0.15,
        Jt12: float = 0.0,
        Js21: float = 0.04,
        Jt21: float = 0.0,
        dt: float = 0.001,
        I_0: float = 0.334,
        tau_ndma: float = 0.06,
        gamma: float = 0.641,
    ):
        """Two-area model, consisting of two connected MurraySingleArea models. 
        Defined by

        .. math::
            \dot{S} = -S/\tau_{NDMA} + \gamma (1 - S) f(J \cdot S + I_0 + I_{app})
        
        where $f$ is the f-I transfer function, $J$ is the weight 
        matrix, and $I_{app}$ is external applied input current.
        Note that the noise current is neglected here and
        instead incorporated into task simulation - see `task.py`.
        
        Parameters
        ----------
        Js1 : float, default: 0.35
            The "structure" of the first area. See 
            `MurraySingleArea` for a description of what
            it means.
        Jt1 : float, default: 0.28387
            The "tone" of the first area. See 
            `MurraySingleArea` for a description of what
            it means.
        Js2 : float, default: 0.4182
            The "structure" of the second area.
        Jt2 : float, default: 0.28387
            The "tone" of the second area.
        Js12 : float, default: 0.15
            The "structure" of the connection from the
            first to second area.
        Jt12 : float, default: 0.0
            The "tone" of the connection from the first
            to second area.
        Js21 : float, default: 0.04
            The "structure" of the connection from the
            second to first area.
        Jt21 : float, default: 0.0
            The "tone" of the connection from the second
            to first area.
        dt : float, default: 0.001
            The bin size of the inputs and outputs, in seconds.
            This is not necessarily equal to the simulation step 
            size, which can be controlled with `steps_per_bin`
            in the `forward()` call. Must be > 0
        I_0 : float, default: 0.334
            Constant input current bias
        tau_ndma : float, default: 0.06
            Time constant for NDMA synapses, in seconds.
            Essentially the decay timescale of model latent state.
            Must be > 0
        gamma : float, default: 0.641
            Parameter that controls saturation rate of 
            the model latent state. Must be > 0 and < 1
        """
        assert dt > 0, f"`dt` must be > 0, got {dt}"
        assert tau_ndma > 0, f"`tau_ndma` must be > 0, got {tau_ndma}"
        assert gamma > 0 and gamma < 1, f"`gamma` must be between 0 and 1, got {gamma}"
        super().__init__()
        J11 = torch.Tensor([
            [(Js1 + Jt1) / 2, (Jt1 - Js1) / 2],
            [(Jt1 - Js1) / 2, (Js1 + Jt1) / 2],
        ]).float()
        J12 = torch.Tensor([
            [(Js12 + Jt12) / 2, (Jt12 - Js12) / 2],
            [(Jt12 - Js12) / 2, (Js12 + Jt12) / 2],
        ]).float()
        J21 = torch.Tensor([
            [(Js21 + Jt21) / 2, (Jt21 - Js21) / 2],
            [(Jt21 - Js21) / 2, (Js21 + Jt21) / 2],
        ]).float()
        J22 = torch.Tensor([
            [(Js2 + Jt2) / 2, (Jt2 - Js2) / 2],
            [(Jt2 - Js2) / 2, (Js2 + Jt2) / 2],
        ]).float()
        self.J = nn.Parameter(
            torch.cat([
                torch.cat([J11, J12], dim=1),
                torch.cat([J21, J22], dim=1),
            ], dim=0)
        )
        self.dt = dt
        self.I_0 = I_0
        self.tau_ndma = tau_ndma
        self.gamma = gamma
        self.batch_first = True

    def forward(
        self,
        input,
        h0=None,
        steps_per_bin=1,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Simulate model forward

        Parameters
        ----------
        input : torch.Tensor
            External applied input current at each timestamp, in nA.
            Should be of shape Batch x Time x 2
        h0 : torch.Tensor, optional
            Initial state of the network. If provided, should be of
            shape 1 x Batch x 2 (to conform with PyTorch RNN conventions).
            If not provided, defaults to zeros
        steps_per_bin : int, default: 1
            Number of integration steps per data bin. Inputs are 
            assumed to stay constant during each bin if multiple
            steps are used.

        Returns
        -------
        states : tuple
            Tuple containing Batch x Time x 2 arrays containing 
            model latent state ($S$) and firing rates ($f(I)$)
            from simulation
        final_state : torch.Tensor
            Final state of the network, to conform with PyTorch
            conventions. Has shape 1 x Batch x 2
        """
        # input: B x T x 4
        # h0: 1 x B x 4
        assert len(input.shape) == 3, f"{input.shape}"
        if h0 is None:
            h0 = torch.zeros((1, input.shape[0], 4))
        else:
            assert len(h0.shape) == 3, f"{h0.shape}"
        input = input.to(self.J.device).float()
        state = h0[0].to(self.J.device).float()
        states = []
        rates = []
        for i in range(input.shape[1]):
            for n in range(steps_per_bin):
                current = state @ self.J + self.I_0 + input[:,i,:]
                rate = current_to_frequency(current)
                state = state + (self.dt / steps_per_bin) * (
                    -state / self.tau_ndma +
                    self.gamma * (1 - state) * rate
                )
            states.append(state)
            rates.append(rate)
        return (
            torch.stack(states, dim=1),
            torch.stack(rates, dim=1)
        ), states[-1].unsqueeze(0)