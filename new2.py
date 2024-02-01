import numpy as np
from new import MesoInception4_3D  # Assuming the MesoInception4_3D class is defined in new.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming IMG_WIDTH, IMG_HEIGHT, learning_rate, batch_size, target_size are defined

class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, frame_skip_prob=0.2, add_random_frames_prob=0.2, frame_rate_change_prob=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_skip_prob = frame_skip_prob
        self.add_random_frames_prob = add_random_frames_prob
        self.frame_rate_change_prob = frame_rate_change_prob

    def random_transform(self, x, seed=None):
        x = super().random_transform(x, seed)

        # Frame reversal with a 50% chance
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2)  # Assuming channel is the last axis (axis=2)

        # Frame skipping
        if np.random.rand() < self.frame_skip_prob:
            x = np.delete(x, np.arange(0, x.shape[0], 2), axis=0)

        # Adding random frames
        if np.random.rand() < self.add_random_frames_prob:
            num_random_frames = np.random.randint(1, 5)
            random_frames = np.random.uniform(0, 1, size=(num_random_frames,) + x.shape[1:])
            x = np.concatenate([x, random_frames], axis=0)

        # Frame rate change
        if np.random.rand() < self.frame_rate_change_prob:
            frame_rate_change_factor = np.random.uniform(0.5, 2.0)
            num_frames = int(x.shape[0] * frame_rate_change_factor)
            x = np.resize(x, (num_frames,) + x.shape[1:])

        return x

# Load the model
import numpy as np
from new import MesoInception4_3D  # Assuming the MesoInception4_3D class is defined in new.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming IMG_WIDTH, IMG_HEIGHT, learning_rate, batch_size, target_size are defined

# Load the model
num_frames = 16  # You can adjust this based on the number of frames you want to consider
classifier = MesoInception4_3D(learning_rate=0.001, num_frames=num_frames)

# Minimal image generator
batch_size = 10

# Use the custom data generator with additional augmentations
dataGenerator = CustomImageDataGenerator(
    rescale=1. / 255,
    frame_skip_prob=0.2,
    add_random_frames_prob=0.2,
    frame_rate_change_prob=0.2
)

# Get directory iterators
train_data_iterator = dataGenerator.flow_from_directory(
    'deepfake_database/train_test',
    target_size=(num_frames, 256),
    batch_size=batch_size,
    class_mode='binary'
)

validation_data_iterator = dataGenerator.flow_from_directory(
    'deepfake_database/validation',
    target_size=(num_frames, 256),
    batch_size=batch_size,
    class_mode='binary'
)

print(train_data_iterator.image_shape)
# Manually generate x and y from iterators
x_train, y_train = next(iter(train_data_iterator))
x_val, y_val = next(iter(validation_data_iterator))

# Model Training
epochs = 1

# Assuming your Classifier class has been updated to accept validation_data
classifier.fit(
    x=x_train,
    y=y_train,
    epochs=epochs,
    validation_data=(x_val, y_val)
)

# Save trained model
classifier.model.save('trained_model_3D.h5')

# Evaluate on Validation Set
validation_loss, validation_accuracy = classifier.model.evaluate(x_val, y_val)
print(f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')
