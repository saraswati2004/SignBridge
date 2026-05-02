from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import os

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Load data
data = []
labels = []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    for file in os.listdir(action_path):
        if file.endswith('.npy'):
            npy_path = os.path.join(action_path, file)
            res = np.load(npy_path)
            data.append(res)
            labels.append(label_map[action])

# Convert to arrays
X = np.array(data).astype('float32')
y = to_categorical(labels).astype(int)

print('X shape:', X.shape)
print('y shape:', y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# Logs
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model 
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, early_stop]
)

# Save model
with open('model_new.json', 'w') as json_file:
    json_file.write(model.to_json())

model.save('model_new.h5')
print('Model saved successfully.')
