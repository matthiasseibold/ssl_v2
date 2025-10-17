from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import resnet
import numpy as np
from LoadSpectrograms import load_data_single_folder
import matplotlib.pyplot as plt

# parameters
img_id = 15
layer_id = 4

# load data
x, y = load_data_single_folder(["F:/spectrograms/spectrograms_implant_loosening/loose/P210475_1"])

# load model and weights
original_model = resnet.ResNet18(input_shape=x.shape[1:], classes=10)
base_model = Model(original_model.input, original_model.layers[-2].output)
input_tensor = base_model.input
xx = base_model.output
xx = Dense(1, activation='sigmoid')(xx)
model = Model(inputs=input_tensor, outputs=xx)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("../checkpoints/saved_resnet_18")

# visualize activations
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(np.reshape(x[img_id, :, :, :], (1, 256, 210, 1)))
first_layer_activation = activations[layer_id]
print(first_layer_activation.shape)
for i in range(first_layer_activation.shape[-1]):
    plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')
    plt.show()
