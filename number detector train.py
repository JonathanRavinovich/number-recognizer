import numpy as np
import tensorflow as tf
from mnist import MNIST


mndata = MNIST('mnist')
images, labels = mndata.load_training()  # train data
images2, labels2 = mndata.load_testing()  # test data

#60000
for i in range(60000):
    images[i] = np.asarray(images[i])
    images[i] = images[i]/255

    for n in range(len(images[i])):
        if images[i][n]<0.7:
            images[i][n] = 0
        else:
            images[i][n] = 1


#10000
for i in range(10000):
    images2[i] = np.asarray(images2[i])
    images2[i] = images2[i]/255

    for n in range(len(images2[i])):
        if images2[i][n]<0.7:
            images2[i][n] = 0
        else:
            images2[i][n] = 1



images = np.asarray(images)
images2 = np.asarray(images2)
labels = np.asarray(labels)
labels2 = np.asarray(labels2)



model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28,28,1), input_dim=28*28),
    tf.keras.layers.Conv2D(60,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(30,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model._name = 'number_detector'

# choose optimiser
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# compile model
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

print("Train")
model.fit(images[:60000],labels[:60000],epochs=20)
print('evaluate')
loss, acc = model.evaluate(images2,labels2)

model.save('models/num_detector')
