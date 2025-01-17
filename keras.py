from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Generate dummy data
train_data = np.random.rand(1000, 784)  # 1000 samples, 784 features each
train_labels = np.random.randint(0, 10, 1000)  # 1000 labels (classes 0-9)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(train_data, train_labels)
print(f"Keras Model - Loss: {loss}, Accuracy: {accuracy}")