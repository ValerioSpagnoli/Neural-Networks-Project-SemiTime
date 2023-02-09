import numpy as np

def magnitude_warp(x, sigma=0.2, knot=4):
    
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    li = []
    for dim in range(x.shape[1]):
        li.append(CubicSpline(warp_steps[:, dim], random_warps[0, :, dim])(orig_steps))
    warper = np.array(li).T

    x_ = x * warper

    return x_


class MagnitudeWrap:
    def __init__(self, sigma, knot):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, data):
        return self.forward(data)
        
    def forward(self, data):
        return magnitude_warp(data, sigma=self.sigma, knot=self.knot)

