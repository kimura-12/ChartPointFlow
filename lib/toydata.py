import numpy as np
from torch.distributions.normal import Normal
from sklearn.utils import shuffle as util_shuffle
import sys

def get_toydata(data, batch_size):
    rng = np.random.RandomState()

    if data == "circle":

        linspace = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)

        circ_x = np.cos(linspace)
        circ_y = np.sin(linspace)

        X = np.vstack([circ_x, circ_y]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        return X.astype("float32")

    elif data == "2sines":

        x = (rng.rand(batch_size) - 0.5) *2 * np.pi
        u = (rng.binomial(1, 0.5, batch_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5
        X = np.vstack([x, y]).T
    
        return X.astype("float32")

    elif data == "double_moon":
        
        x1_1 = np.random.normal(4, 3.5, (batch_size // 2,))
        x2_1 = np.random.normal(0.25*(x1_1-4)**2-20, np.zeros_like(x1_1)) -10

        x1_2 = np.random.normal(-4, 3.5, (batch_size//2,))
        x2_2 = np.random.normal(-0.25*(x1_2+4)**2+20, np.zeros_like(x1_2)) + 10

        x1 = np.concatenate([x1_1, x1_2])
        x2 = np.concatenate([x2_1, x2_2])
        X = np.zeros((batch_size, 2))

        X[:,0] = x1*0.2
        X[:,1] = x2*0.1
        X = util_shuffle(X, random_state=rng)

        return X.astype("float32")

    elif data == "chekerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "four_circle":
        def create_circle(num_per_circle, std=0):
            u = np.random.rand(num_per_circle)
            x1 = np.cos(2 * np.pi * u)
            x2 = np.sin(2 * np.pi * u)
            data = 2 * np.vstack([x1, x2]).T
            return data
        
        num_per_circle = batch_size // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        x = np.concatenate(
            [create_circle(num_per_circle) - np.array(center)
             for center in centers]
        )

        x = util_shuffle(x, random_state=rng)
        return x
    
    else :
        print("Error : Input correct dataset name")
        sys.exit()