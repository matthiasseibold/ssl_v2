import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.resnet import ResNet50
import utils.resnet as resnet

print("loading data")
train_x = np.load("data/train_x.npy")
train_y = np.load("data/train_y.npy")
test_x = np.load("data/test_x.npy")
test_y = np.load("data/test_y.npy")

print("Training data shape: " + str(train_x.shape))
print("Training labels shape: " + str(train_y.shape))
print("Test data shape: " + str(test_x.shape))
print("Test labels shape: " + str(test_y.shape))
print("")

# load model
model = resnet.ResNet18(input_shape=train_x.shape[1:], classes=train_y.shape[1], use_se=True)

optimizer = Adam(learning_rate=5e-6)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              # loss=[focal_loss],
              metrics=['accuracy'])

callbacks = [

    # reduce LR
    # ReduceLROnPlateau(monitor='val_loss',
    #                   factor=0.1,
    #                   patience=5,
    #                   cooldown=1),

    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
]

history = model.fit(train_x, train_y, epochs=100, callbacks=callbacks, batch_size=32,
                    validation_data=(test_x, test_y))

# model.save_weights("checkpoints/saved_resnet_18")

results = model.evaluate(test_x, test_y)
print('Final test accuracy: ' + str(results[1] * 100) + '%')

# print performance metrics
y_pred = model.predict(test_x)
print('Confusion Matrix')
cm = confusion_matrix(test_y.argmax(axis=1), (y_pred > 0.5).argmax(axis=1))
print(cm)

print('Classification Report')
cr = classification_report(test_y.argmax(axis=1), (y_pred > 0.5).argmax(axis=1), target_names=['nopeak', 'peak'])
print(cr)

model.save("./checkpoints/trained_model.hdf5")

print('Per class accuracies')
accuracies = cm.diagonal() / cm.sum(axis=1)
print(accuracies)


