import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Menentukan jumlah data minimum per label
min_data_per_label = 1

# Mendefinisikan path untuk data training
train_data_dir = './training/'

# Pra-pemrosesan dan normalisasi gambar
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Menggunakan 80% data untuk training, 20% untuk validasi
)

# Memuat data training dan validasi
train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Mengubah ukuran gambar menjadi 224x224
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Mengecek jumlah data per label
label_counts = np.bincount(train_generator.classes)
num_classes = len(label_counts)
assert np.min(label_counts) >= min_data_per_label, "Jumlah data per label tidak mencukupi."

# Memuat model VGG16 tanpa lapisan terakhir (top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Membuat model baru dengan lapisan tambahan di atas VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Melatih model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Mengujikan model dengan gambar dari kamera
test_image = load_image_from_camera()  # Fungsi untuk memuat gambar dari kamera
normalized_image = test_image / 255.0  # Normalisasi gambar
reshaped_image = np.reshape(normalized_image, (1, 224, 224, 3))  # Mengubah bentuk gambar sesuai dengan input model
predictions = model.predict(reshaped_image)
predicted_label = np.argmax(predictions)

# Menampilkan hasil prediksi
if predicted_label == 0:
    print("mobil")
else:
    print("motor")