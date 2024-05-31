# Импорт библиотеки TensorFlow под псевдонимом tf
import tensorflow as tf
# Импорт модуля ImageDataGenerator для аугментации изображений
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Импорт класса Model для создания модели нейронной сети
from tensorflow.keras.models import Model
# Импорт слоев для построения нейронной сети
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
# Импорт предварительно обученной модели MobileNet и функции для предварительной
# обработки входных данных
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
# Импорт модуля для предварительной обработки изображений
from tensorflow.keras.preprocessing import image
# Импорт модуля для математических операций
import math
# Импорт библиотеки Matplotlib для визуализации данных
import matplotlib.pyplot as plt
# Импорт библиотеки NumPy для работы с массивами
import numpy as np
# Импорт модуля Image из библиотеки Pillow для работы с изображениями
from PIL import Image

TRAIN_DATA_DIR = 'pizza-not_pizza/train' # Путь к каталогу с обучающими данными
VALIDATION_DATA_DIR = 'pizza-not_pizza/test' # Путь к каталогу с данными для валидации
TRAIN_SAMPLES = 1800 # Общее количество обучающих примеров
VALIDATION_SAMPLES = 200 # Общее количество примеров для валидации
NUM_CLASSES = 2 # Количество классов (в данном случае два класса: 'кошки' и 'собаки')
IMG_WIDTH, IMG_HEIGHT = 224, 224 # Размеры изображений (ширина и высота)
BATCH_SIZE = 10 # Размер пакета для обучения модели (количество образцов, обрабатываемых моделью за один шаг обучения)


train_datagen = ImageDataGenerator( # Создание генератора данных для обучения
    preprocessing_function=preprocess_input, # Предварительная обработка изображений (в данном случае с использованием функции preprocess_input)
    rotation_range=20, # Диапазон вращения изображений (от -20 до 20 градусов)
    width_shift_range=0.2, # Диапазон сдвига изображений по горизонтали(относительно ширины изображения)
    height_shift_range=0.2, # Диапазон сдвига изображений по вертикали(относительно высоты изображения)
    zoom_range=0.2 # Диапазон масштабирования изображений (от 0.8 до 1.2)
)
val_datagen = ImageDataGenerator( # Создание генератора данных для валидации
    preprocessing_function=preprocess_input # Предварительная обработка изображений (в данном случае с использованием функции preprocess_input)
)

train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR, # Путь к каталогу с обучающими данными
    target_size=(IMG_WIDTH, IMG_HEIGHT), # Желаемый размер изображений (ширина, высота)
    batch_size=BATCH_SIZE, # Размер пакета для обучения модели
    shuffle=True, # Перемешивание данных после каждой эпохи
    seed=12345, # Задание начального состояния для генератора случайных чисел (для воспроизводимости)
    class_mode='categorical' # Режим классификации (в данном случае многоклассовая классификация)
)
validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR, # Путь к каталогу с данными для валидации
    target_size=(IMG_WIDTH, IMG_HEIGHT), # Желаемый размер изображений (ширина, высота)
    batch_size=BATCH_SIZE, # Размер пакета для валидации модели
    shuffle=False, # Не перемешивать данные (для сохранения порядка)
    class_mode='categorical' # Режим классификации (в данном случае многоклассовая классификация)
)

def model_maker():
    # Создание базовой модели MobileNet без верхнего слоя классификации,
    # указывается форма входных данных (ширина, высота, количество каналов)
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    # Замораживаем веса всех слоев базовой модели, чтобы они не обучались
    for layer in base_model.layers[:]:
        layer.trainable = False# Определение входного тензора модели
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))#Пропуск данных через базовую модель
    custom_model = base_model(input)
    # Глобальное пулингование для уменьшения размерности признаков
    custom_model = GlobalAveragePooling2D()(custom_model)
    # Полносвязный слой с 16 нейронами и функцией активации ReLU
    custom_model = Dense(128, activation='relu')(custom_model)
    # Слой регуляризации для предотвращения переобучения
    custom_model = Dropout(0.25)(custom_model)
    # Выходной слой с NUM_CLASSES нейронами и функцией активации softmax для многоклассовой классификации
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    # Создание модели с заданными входом и выходом
    return Model(inputs=input, outputs=predictions)

model = model_maker() # Создание модели нейронной сети с помощью функции model_maker()
model.compile(loss='categorical_crossentropy', # Функция потерь - категориальная перекрестная энтропия
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # Оптимизатор - Adam с коэффициентом скорости обучения 0.001
    metrics=['acc']) # Метрика для оценки производительности модели - точность классификации
num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE) # Вычисление количества шагов обучения
history = model.fit(train_generator, # Обучение модели на обучающих данных
    steps_per_epoch=num_steps, # Количество шагов обучения
    epochs=10, # Количество эпох обучения
    validation_data=validation_generator, # Данные для валидации модели
    validation_steps=num_steps) # Количество шагов валидации

loss_function = history.history['loss'] # Получение значений функции потерь на обучающем наборе данных из истории обучения
val_loss_function = history.history['val_loss'] # Получение значений функции потерь на валидационном наборе данных из истории обучения
epochs = range(1, len(loss_function) + 1) # Создание списка эпох для оси x

plt.title('Losses') # Заголовок графика
plt.plot(epochs, loss_function, color='blue', label='Потери в обучающей выборке') # График функции потерь на обучающем наборе
plt.plot(epochs, val_loss_function, color='red', label='Потери в валидационной выборке') # График функции потерь на валидационном наборе
plt.xlabel('Epoch') # Подпись оси x
plt.ylabel('Loss value') # Подпись оси y
plt.legend() # Добавление легенды
plt.show() # Отображение графика


import os # Импорт модуля os для работы с файловой системой
import random # Импорт модуля random для работы с случайными числами

categ = ['not_pizza', 'pizza'] # Список категорий: 'cat' - кошка, 'dog' - собака
files = [] # Пустой список для хранения путей к изображениям
# Перебор всех файлов в каталоге с кошками и добавление путей в список files
for root, dirs, filenames in os.walk('pizza-not_pizza/test/pizza'):
    for filename in filenames:
        files.append(os.path.join(root, filename))
# Перебор всех файлов в каталоге с собаками и добавление путей в список files
for root, dirs, filenames in os.walk('pizza-not_pizza/test/not_pizza'):
    for filename in filenames:
        files.append(os.path.join(root, filename))
# Выбор случайных 10 изображений из списка files и их отображение с предсказанием модели
for f in random.sample(files, 15):
    img_path = f # Путь к изображению
    img = image.load_img(img_path, target_size=(224,224)) # Загрузка изображения с изменением размера до 224x224 пикселя
    img_array = image.img_to_array(img) # Преобразование изображения в массив numpy
    expanded_img_array = np.expand_dims(img_array, axis=0) # Расширение массива изображения
    preprocessed_img = preprocess_input(expanded_img_array) # Предварительная обработка изображения
    prediction = model.predict(preprocessed_img) # Получение предсказания от модели
    plt.title(categ[np.argmax(prediction)]) # Установка заголовка графика в соответствии с предсказанием модели
    plt.imshow(img) # Отображение изображения
    plt.show() # Показ графика
