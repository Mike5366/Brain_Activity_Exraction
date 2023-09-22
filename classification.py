import shutil
import os
import pandas as pd
from keras import Sequential
import tensorflow as tf
from tensorflow.keras import layers, models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import MaxPool2D
# from keras import backend as K
# from keras.optimizer_v2.rmsprop import RMSprop


def remove_directory(path):
    # clear result directory if exist
    if os.path.exists(path):
        shutil.rmtree(path)


def make_directory(path):
    # clear result directory if exist
    if os.path.exists(path):
        shutil.rmtree(path)

    # make directory
    os.makedirs(path)


# make folders for training and testing
def make_folder(folder_path, start_patient_no, end_patient_no):
    noise_path = os.path.join(folder_path, "Noise")
    rsn_path = os.path.join(folder_path, "RSN")

    make_directory(folder_path)
    make_directory(noise_path)
    make_directory(rsn_path)

    # initialize and read data
    input_path = os.path.join(".", "PatientData-2", "PatientData")

    ith_patient = start_patient_no

    while ith_patient <= end_patient_no:
        patient_path = os.path.join(input_path, "Patient_%s" % ith_patient)

        label_path = os.path.join(input_path, "Patient_%s_Labels.csv" % ith_patient)
        if os.path.exists(label_path):
            label_per_patient = pd.read_csv(label_path)
        else:
            break

        for index, row in label_per_patient.iterrows():
            image_name = "IC_%s_thresh.png" % row['IC']
            image_path = os.path.join(patient_path, image_name)
            if os.path.exists(image_path):
                patient = "P%s_" % ith_patient
                if row['Label'] == 0:
                    shutil.copy(image_path, os.path.join(noise_path, patient + image_name))
                else:
                    shutil.copy(image_path, os.path.join(rsn_path, patient + image_name))

        ith_patient = ith_patient + 1


train_folder_path = os.path.join(".", "Train")
validate_folder_path = os.path.join(".", "Validate")
make_folder(train_folder_path, 1, 5)
make_folder(validate_folder_path, 5, 5)

batch_size = 16
img_height = 256
img_width = 256

train = ImageDataGenerator()
validate = ImageDataGenerator()

train_dataset = train.flow_from_directory(train_folder_path,
                                          target_size=(img_height, img_width),
                                          batch_size=batch_size,
                                          class_mode='binary')

validate_dataset = validate.flow_from_directory(validate_folder_path,
                                                target_size=(img_height, img_width),
                                                batch_size=batch_size,
                                                class_mode='binary')


# def specificity(y_true, y_pred):
#     tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
#     fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
#     return tn / (tn + fp + K.epsilon())


def create_model():
    model = Sequential([layers.Rescaling(1. / 255),
                        layers.Conv2D(16, (3, 3), padding='same', activation='relu',
                                      input_shape=(img_height, img_width, 3)),
                        layers.MaxPool2D(2, 2),
                        #
                        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                        layers.MaxPool2D(2, 2),
                        #
                        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                        layers.MaxPool2D(2, 2),
                        ##
                        layers.Flatten(),
                        ##
                        layers.Dense(256, activation='relu'),
                        # layers.Dropout(0.1, seed = 2019),
                        # layers.Dense(400, activation ="relu"),
                        # layers.Dropout(0.3, seed = 2019),
                        ##
                        layers.Dense(1, activation='sigmoid')
                        ])

    model.compile(loss='binary_crossentropy',
                  # optimizer=RMSprop(learning_rate=0.001),
                  optimizer='adam',
                  # metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), specificity],
                  metrics=['accuracy'],
                  )

    return model


checkpoint_path = "Trained model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)
model = create_model()

history = model.fit(train_dataset,
                    steps_per_epoch=8,
                    epochs=30,
                    validation_data=validate_dataset,
                    callbacks=[cp_callback])

remove_directory(train_folder_path)
remove_directory(validate_folder_path)
