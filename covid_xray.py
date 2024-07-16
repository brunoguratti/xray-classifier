import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.initializers import HeNormal

import cv2
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# Example usage
data_dir = "/kaggle/input/covid-data/Val"

def load_images_for_visualization(data_dir, target_size=(224, 224)):
    categories = ['COVID-19', 'Non-COVID', 'Normal']
    original_images = []
    masked_images = []
    labels = []

    for category in categories:
        images_path = os.path.join(data_dir, category, 'images')
        masks_path = os.path.join(data_dir, category, 'lung masks')

        for filename in sorted(os.listdir(images_path)):
            img_path = os.path.join(images_path, filename)
            mask_path = os.path.join(masks_path, filename)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image
            img = cv2.resize(img, target_size)
            mask = cv2.resize(mask, target_size)
            img_masked = cv2.bitwise_and(img, img, mask=mask)

            original_images.append(img)
            masked_images.append(img_masked)
            labels.append(category)
            
    # Convert lists to NumPy arrays for element-wise operations
    original_images = np.array(original_images)
    masked_images = np.array(masked_images)
    
    # One-hot encode the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    return original_images, masked_images, labels

def preprocess_pipeline(image, ksize, threshold=False):
    image = image.astype(np.uint8)
        
    if threshold:
        # Apply adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image

def preprocess_images(image_array, ksize=5, threshold=False):
    processed_images = []
    for image in image_array:
        processed_image = preprocess_pipeline(image, ksize, threshold) / 255.0  # Normalize the image
        if len(processed_image.shape) == 2:  # Check if the image is 2D
            processed_image = np.expand_dims(processed_image, axis=-1)  # Expand dimensions if necessary
        processed_images.append(processed_image)
        
    return np.array(processed_images)

# Pipeline 1 (No preprocessing)
original_images, masked_images, labels = load_images_for_visualization(data_dir)
# Expand one dimension so the array can be used in the CNN
original_images = np.expand_dims(original_images, axis=-1)

# Pipeline 2 (Histogram equalization and Gaussian blur)
images_pipeline2 = preprocess_images(original_images, ksize=5, threshold=False)

# Print the shapes
print("Pipeline images shape:", images_pipeline2.shape)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def plot_confusion_matrix_and_roc_auc(cm, y_true, y_pred, classes = ['COVID-19', 'Non-COVID', 'Normal'], n_classes=3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confusion matrix
    ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion matrix')
    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, rotation=45)
    ax1.set_yticklabels(classes)
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax1.text(j, i, format(cm[i, j], fmt),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    # Plot ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        ax2.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def model_performance(model, history, X_val, y_val, classes = ['COVID-19', 'Non-COVID', 'Normal']):
    
    # Evaluating the model
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # AUC Score
    roc_auc = roc_auc_score(y_val, y_pred_prob, multi_class='ovr')

    # Print classification report
    print(classification_report(y_true, y_pred))
    print(f"Training accuracy: {round(history.history['accuracy'][-1],3)}")
    print(f"Validation accuracy: {round(history.history['val_accuracy'][-1],3)}")
    print(f"ROC AUC: {round(roc_auc,3)}")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix and ROC AUC side by side
    plot_confusion_matrix_and_roc_auc(conf_matrix, y_val, y_pred_prob, classes=classes, n_classes=len(classes))

# Function to extract features from images in batches
def extract_features(model, preprocessed_data, batch_size=32):
    n_samples = preprocessed_data.shape[0]
    features = []
    for i in range(0, n_samples, batch_size):
        batch = preprocessed_data[i:i+batch_size]
        batch_features = model.predict(batch, verbose=0)
        features.append(batch_features)
    features = np.vstack(features)
    return features

# Function to calculate Fisher's Discriminant Ratio (FDR)
def fisher_discriminant_ratio(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    fdr = np.zeros(n_features)
    
    for i in range(n_features):
        feature = X[:, i]
        means = []
        variances = []
        for cls in classes:
            cls_feature = feature[y == cls]
            means.append(np.mean(cls_feature))
            variances.append(np.var(cls_feature))
        
        overall_mean = np.mean(feature)
        between_class_variance = sum([(mean - overall_mean) ** 2 for mean in means])
        within_class_variance = sum(variances)
        
        fdr[i] = between_class_variance / within_class_variance
    
    return fdr

# Function to evaluate classification performance using cross-validation
def evaluate_feature_set(X, y):
    clf = SVC(kernel='linear')
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)

# Pipeline to train CNN model 1
def train_cnn_model_aug(model_fn, input_shape, X, y, optimizer='adam', loss='categorical_crossentropy', epochs=10, batch_size=32):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = model_fn(input_shape, optimizer=optimizer, loss=loss)
       
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
       
    datagen = ImageDataGenerator(
    rotation_range = 10,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode='nearest',
    )
        
    # Initialize the data generator with the training data
    datagen.fit(X_train)
    
    # Ensure y_train is in the correct format
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_flat = np.argmax(y_train, axis=1)
    else:
        y_train_flat = y_train
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_flat), y=y_train_flat)

    # Convert to a dictionary format required by Keras
    class_weights = dict(enumerate(class_weights))

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[es, reduce_lr], class_weight=class_weights)
    model_performance(model, history, X_val, y_val)
    
    return model, history, X_val, y_val, X_test, y_test

# Define the CNN model 3
def create_cnn_model3(input_shape, optimizer, loss):
    input_shape = input_shape
    initializer = HeNormal()


    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', kernel_initializer=initializer),      
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
        MaxPooling2D((2, 2)),
                
        Flatten(),
        Dense(256, activation='relu', kernel_initializer=initializer),
        Dropout(0.6),
        Dense(128, activation='relu', kernel_initializer=initializer),
        Dropout(0.6),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', 'f1_score'])
    return model

# List of datasets
datasets = [images_pipeline2]
input_shape = original_images[0].shape

# Training CNN model on all datasets. Record the accuracy and ROC AUC scores and show the table at the end
for i, data in enumerate(datasets):
    print(f"Training on dataset {i+1}/{len(datasets)}")
    model, history, X_val, y_val, X_test, y_test = train_cnn_model_aug(create_cnn_model3, input_shape, data, labels, epochs=50, batch_size=32)
    print(f"Finished training on dataset {i+1}/{len(datasets)}\n")
    accuracy_score_record = history.history['val_accuracy'][-1]
    roc_auc_score_record = roc_auc_score(y_val, model.predict(X_val), multi_class='ovr')
    if i == 0:
        results = {'Dataset': [f'Dataset {i+1}'], 'Accuracy': [accuracy_score_record], 'ROC AUC': [roc_auc_score_record]}
    else:
        results['Dataset'].append(f'Dataset {i+1}')
        results['Accuracy'].append(accuracy_score_record)
        results['ROC AUC'].append(roc_auc_score_record)

results_model_3df = pd.DataFrame(results)
results_model_3df