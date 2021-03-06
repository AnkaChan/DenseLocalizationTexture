import imageio
import numpy as np

if __name__ == '__main__':
    N = 2000
    X = np.linspace(0, 1, N)
    Y = np.linspace(1, 0, N)
    u, v = np.meshgrid(X, Y)

    b = np.zeros((N, N))

    uvTexture = np.stack([u, v, b], axis=2)
    arr = uvTexture.astype("float32")

    # Write to disk
    imageio.imwrite('uv_texture.exr', arr)
