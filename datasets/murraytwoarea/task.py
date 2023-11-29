import numpy as np
import neurogym
from neurogym.utils import spaces
from typing import Union, Optional, Literal


def ornstein_uhlenbeck( # TODO: diagnose!
    shape: tuple,
    std: float = 0.009,
    tau: float = 0.002,
    dt: float = 0.001,
    steps_per_bin: int = 1,
    rng: Union[int, np.random.Generator] = 0,
) -> np.ndarray:
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    t = shape[0]
    state = np.zeros(shape[-1])
    noise = np.zeros(shape)
    true_dt = (dt / steps_per_bin)
    for i in range(t):
        for n in range(steps_per_bin):
            state = (state + 
                true_dt * (-state / tau) + 
                std * np.sqrt(true_dt * 2 / tau) * # following wikipedia
                rng.normal(loc=0., scale=1., size=state.shape)
            )
        noise[i,:] = state
    return noise


class PulseDiscriminationTask(neurogym.TrialEnv):
    def __init__(
        self,
        dt: float = 1.,
        frequencies: Optional[list[float]] = None,
        rewards: Optional[dict] = None,
        timing: Optional[dict] = None,
        noise_type: Literal["stationary", "ornstein-uhlenbeck"] = "ornstein-uhlenbeck",
        noise_scale: float = 0.005,
        noise_tau: float = 2.,
        pulse_height: float = 0.0118,
        pulse_width: float = 50., # in ms
        pulse_kernel: Literal["box", "decay", "gaussian"] = "box",
        extra_channels: int = 2,
    ):
        super().__init__(dt=dt)

        default_frequencies = [
            4, 6, 8, 10, 12, 14, 16,
            18, 20, 22, 24, 26, 28, 30,
        ]
        self.frequencies = frequencies or default_frequencies

        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 0,
            'stimulus': 1000,
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(2)

        name = {'stimulus': range(2), 'extra': range(2, 2+extra_channels)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2 + extra_channels,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.Discrete(3, name=name)
    
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noise_tau = noise_tau

        self.pulse_height = pulse_height
        self.pulse_width = pulse_width
        self.extra_channels = extra_channels
        if pulse_kernel == "box":
            self.pulse_kernel = np.ones(self.pulse_width // self.dt) * self.pulse_height
        elif pulse_kernel == "decay":
            self.pulse_kernel = np.exp(-np.linspace(0, 5, self.pulse_width // self.dt)) * self.pulse_height
        elif pulse_kernel == "gaussian":
            self.pulse_kernel = np.exp(-0.5 * np.linspace(-3, 3, self.pulse_width // self.dt) ** 2) * self.pulse_height

    def _new_trial(self, **kwargs):
        # Trial info
        stimset = self.rng.choice(len(self.frequencies))
        stim1_hz = self.frequencies[stimset]
        stim2_hz = self.frequencies[-(1 + stimset)]
        stim_total = stim1_hz + stim2_hz
        trial = {'stim1_hz': stim1_hz, 'stim2_hz': stim2_hz}
        trial.update(kwargs)
        stim1_hz = trial['stim1_hz']
        stim2_hz = trial['stim2_hz']
        assert (stim1_hz + stim2_hz) == stim_total, \
            f"total stimulus not equal to {stim_total}, got: {stim1_hz + stim2_hz}"

        trial_dur = int(sum(self.timing.values()) / self.dt)
        stim_dur = int(self.timing['stimulus'] / self.dt)
        stim1_arr = self.rng.poisson(stim1_hz * self.dt / 1000, size=stim_dur)
        stim1_arr = np.convolve(stim1_arr, self.pulse_kernel, mode='same')
        stim2_arr = self.rng.poisson(stim2_hz * self.dt / 1000, size=stim_dur)
        stim2_arr = np.convolve(stim2_arr, self.pulse_kernel, mode='same')

        if self.noise_type == "stationary":
            noise = self.rng.normal(
                loc=0, 
                scale=self.noise_scale, 
                size=(trial_dur, 2 + self.extra_channels),
            )
        else:
            noise = ornstein_uhlenbeck(
                shape=(trial_dur, 2 + self.extra_channels),
                std=self.noise_scale,
                tau=self.noise_tau / 1000,
                dt=self.dt / 1000,
                steps_per_bin=(int(self.dt*5/self.noise_tau) + 1),
                rng=self.rng,
            )
        ob = np.stack([stim1_arr, stim2_arr, np.zeros_like(stim1_arr), np.zeros_like(stim1_arr)], axis=1)

        ground_truth = int(stim1_hz < stim2_hz)
        trial['ground_truth'] = ground_truth

        # Periods
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        # Observations
        self.set_ob(noise)
        self.add_ob(ob, 'stimulus')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        # store true stimulus
        self.true_stimulus = self.view_ob() - noise

        return trial
        
    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
        else:
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class PerceptualDiscriminationTask(neurogym.TrialEnv):
    def __init__(
        self,
        dt: float = 1.,
        cohs: Optional[list[float]] = None,
        rewards: Optional[dict] = None,
        timing: Optional[dict] = None,
        noise_type: Literal["stationary", "ornstein-uhlenbeck"] = "ornstein-uhlenbeck",
        noise_scale: float = 0.005,
        max_input: float = 0.0118,
        extra_channels: int = 2,
    ):
        return
