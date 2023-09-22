import os
import pandas as pd
from keras_preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf


# Load model
print("load model")
checkpoint_path = "Trained model/cp.ckpt"
model = load_model(checkpoint_path)
print(model.summary())


# initialize and read data
input_path = os.path.join(".", "testPatient")
patient_path = os.path.join(input_path, "test_Data")
label_path = os.path.join(input_path, "test_Labels.csv")

img_height = 256
img_width = 256

pred = []
file_name = []
label = []

if os.path.exists(label_path):
    label_per_patient = pd.read_csv(label_path)

    for index, row in label_per_patient.iterrows():
        image_name = "IC_%s_thresh.png" % row['IC']
        image_path = os.path.join(patient_path, image_name)
        if os.path.exists(image_path):
            # load a single image
            # print("Load image %s" % row['IC'])
            new_image = load_img(image_path, target_size=(img_height, img_width))
            input_arr = tf.keras.utils.img_to_array(new_image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            # check prediction
            prob = model.predict(input_arr)
            # print(prob)
            file_name.append(row['IC'])

            if row['Label'] > 0:
                label.append(1)
            else:
                label.append(0)

            if prob[0] > 0.5:
                pred.append(1)
            else:
                pred.append(0)

# print(file_name)
# print(label)
# print(pred)

tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
# print(tn)
# print(fp)
# print(fn)
# print(tp)

accuracy = (tp + tn) / (tn + fp + fn + tp)
precision = tp / (tp + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

accuracy = "%.0f%%" % (accuracy * 100 + 0.5)
precision = "%.0f%%" % (precision * 100 + 0.5)
sensitivity = "%.0f%%" % (sensitivity * 100 + 0.5)
specificity = "%.0f%%" % (specificity * 100 + 0.5)
print("%s: %s" % ('accuracy', accuracy))
print("%s: %s" % ('precision', precision))
print("%s: %s" % ('sensitivity', sensitivity))
print("%s: %s" % ('specificity', specificity))

metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity']
score = [accuracy, precision, sensitivity, specificity]

metrics_df = pd.DataFrame(score, columns = ['Score'], index = metrics)
metrics_df.index.name = "Metrics"
metrics_df.to_csv('./Metrics.csv')


result_df = pd.DataFrame()
result_df['IC_Number'] = file_name
result_df['Label'] = pred
result_df['True_Label'] = label
result_df.to_csv('./Results.csv', index = False)


