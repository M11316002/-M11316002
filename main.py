import argparse
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import os
import matplotlib.pyplot as plt

# 構建參數解析器並解析參數
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset directory")
ap.add_argument("-i", "--image", required=True, help="Path to the test image")
args = vars(ap.parse_args())

# 取得數據集路徑和影像路徑
src_dir = args["dataset"]
predict_img = args["image"]

# 驗證影像的有效性
def validate_images(directory):
    from PIL import UnidentifiedImageError
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, IOError):
                print(f"Invalid image: {img_path}")

validate_images(src_dir)

# 數據增強
datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 建立數據生成器
train_generator = datagen.flow_from_directory(
    src_dir, target_size=(244, 244), batch_size=20, subset='training'
)
valid_generator = datagen.flow_from_directory(
    src_dir, target_size=(244, 244), batch_size=20, subset='validation'
)

# 建立模型
mobilenetV2 = MobileNetV2(include_top=False, pooling='avg')
for mlayer in mobilenetV2.layers:
    mlayer.trainable = False

mobilenetV2output = mobilenetV2.layers[-1].output
dropout = layers.Dropout(0.5)(mobilenetV2output)
fc = layers.Dense(
    units=train_generator.num_classes,
    activation='softmax',
    kernel_regularizer=regularizers.l2(0.01),
    name='custom_fc'
)(dropout)

classification_model = Model(inputs=mobilenetV2.inputs, outputs=fc)
classification_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# 訓練模型
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = classification_model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator,
    callbacks=[early_stopping]
)

# 預測測試影像
true_labels_dict = {v: k for k, v in train_generator.class_indices.items()}

def pred(img_path):
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((244, 244))
    img_array = preprocess_input(np.array(img))
    img_array = np.expand_dims(img_array, axis=0)
    result_prob = classification_model.predict(img_array).tolist()[0]
    max_index = result_prob.index(max(result_prob))
    predicted_class = true_labels_dict[max_index]
    confidence = result_prob[max_index]
    print(f"Image: {img_name}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    plt.imshow(img)
    plt.axis('off')
    plt.text(10, 10, f"{predicted_class}: {confidence:.2f}", color='white', fontsize=12, weight='bold',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))
    plt.show()

if os.path.exists(predict_img):
    pred(predict_img)
else:
    print(f"Image {predict_img} does not exist. Please check the path.")
