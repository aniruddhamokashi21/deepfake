#for neural network
from tensorflow.keras.applications import InceptionResNetV2                # type: ignore
from tensorflow.keras.layers import Conv2D                                 # type: ignore
from tensorflow.keras.layers import MaxPooling2D                           # type: ignore
from tensorflow.keras.layers import Flatten                                # type: ignore
from tensorflow.keras.layers import Dense                                  # type: ignore
from tensorflow.keras.layers import Dropout                                # type: ignore
from tensorflow.keras.layers import InputLayer                             # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D                 # type: ignore
from tensorflow.keras.models import Sequential                             # type: ignore
from tensorflow.keras.models import Model                                  # type: ignore
from tensorflow.keras import optimizers                                    # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping    # type: ignore


X_train = '.\dataset\train_sample_videos\real'
Y_train = '.\dataset\train_sample_videos\fake'
X_val = '.\dataset\test_videos\real'
Y_val = '.\dataset\test_videos\fake'

googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
googleNet_model.trainable = True
model = Sequential()
model.add(googleNet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(units=2, activation='softmax'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=2,
                               verbose=0, mode='auto')
EPOCHS = 20
BATCH_SIZE = 100
history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_val, Y_val), verbose = 1)


model.save('dd-model.keras')