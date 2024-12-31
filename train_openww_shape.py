import numpy as np

data = np.load('./spectrograms/data.npy')
labels = np.load('./spectrograms/labels.npy')

print(data.shape)  # Should be (num_samples, 64, 100)
print(labels.shape)
