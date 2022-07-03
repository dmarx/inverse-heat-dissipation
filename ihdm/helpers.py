import numpy as np
from scipy.fftpack import dct, idct

def heat_eq_forward(u, t):
    """
    Calculates the forward heat-dissipation process.
    
    Code was lifted directly from the paper
    Appendix A.1, Algorithm 3 - https://arxiv.org/pdf/2206.13397.pdf
    """
    # Assuming the image u is an (KxK ) numpy array
    frequencies = np.pi * linspace(0, K-1, K) / K
    frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
    u_proj = dct(u, axis=0)
    u_proj = dct(u_proj , axis =1)
    u_proj = exp(-frequencies_squared*t) * u_proj
    u_reconstucted = idct(u_proj, axis=0)
    u_reconstucted = idct(u_reconstucted, axis=1)
    return u_reconstucted
