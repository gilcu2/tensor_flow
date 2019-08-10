import os
import random

import image_processing
import numpy as np
import tensorflow as tf


def build_model():
    x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    y_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
    images_flat = tf.contrib.layers.flatten(x_placeholder)
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_placeholder, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_pred = tf.argmax(logits, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)
    return train_op, loss, correct_pred, accuracy, x_placeholder, y_placeholder


def train(train_op, accuracy, loss, images, labels, x_placeholder, y_placeholder):
    session = tf.Session()

    session.run(tf.global_variables_initializer())

    for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = session.run([train_op, accuracy], feed_dict={x_placeholder: images, y_placeholder: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

    return session


def predict(session, correct_pred,images,x_placeholder ):

    predicted = session.run([correct_pred], feed_dict={x_placeholder: images})[0]

    return predicted


def main():
    initial_dir = "../data/"
    train_data_dir = os.path.join(initial_dir, "Training")
    test_data_dir = os.path.join(initial_dir, "Testing")

    images, labels = image_processing.load_data(train_data_dir)

    images_array = np.array(images)
    labels_array = np.array(labels)
    unique_labels = set(labels_array)

    image_processing.print_images_description(images_array)
    image_processing.print_label_description(labels_array, unique_labels)

    # plot_labels_histogram(labels, len(unique_labels))

    images_gray = image_processing.prepare_images(images_array)

    # image_processing.show_images_per_class(images_gray, labels, unique_labels)

    train_op, loss, correct_pred, accuracy, x_placeholder, y_placeholder = build_model()
    session = train(train_op, accuracy, loss, images_gray, labels, x_placeholder, y_placeholder)

    test_images, test_labels = image_processing.load_data(test_data_dir)
    test_images_array=np.array(test_images)
    test_images_gray = image_processing.prepare_images(test_images_array)

    predicted=predict(session, correct_pred, test_images_gray, x_placeholder)
    ok_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = ok_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))

if __name__ == '__main__':
    main()
