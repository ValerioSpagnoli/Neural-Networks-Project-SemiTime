import numpy as np

# Important: this functions have been taken from original code (https://github.com/haoyfan/SemiTime)

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        time_warp = CubicSpline(warp_steps[:, dim],
                                warp_steps[:, dim] * random_warps[0, :, dim])(orig_steps)
        scale = (x.shape[0] - 1) / time_warp[-1]
        ret[:, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1),
                                   x[:, dim]).T
    
    return ret


class TimeWarp:
    def __init__(self, sigma, knot):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        return time_warp(data, sigma=self.sigma, knot=self.knot)