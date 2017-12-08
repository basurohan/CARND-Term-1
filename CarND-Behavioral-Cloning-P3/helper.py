import cv2
import numpy as np
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = int(len(samples) / batch_size)
    num_samples = num_samples * batch_size
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            path = 'data/IMG/'
            images, measurements = [], []
            correction = 0.20

            for batch_sample in batch_samples:
                image = cv2.imread(path + batch_sample[0].split('/')[-1])
                images.append(image)
                measurements.append(float(batch_sample[3]))
                image = cv2.imread(path + batch_sample[1].split('/')[-1])
                images.append(image)
                measurements.append(float(batch_sample[3]) + correction)
                image = cv2.imread(path + batch_sample[2].split('/')[-1])
                images.append(image)
                measurements.append(float(batch_sample[3]) - correction)

            augmented_images, augmented_measurements = [], []

            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)