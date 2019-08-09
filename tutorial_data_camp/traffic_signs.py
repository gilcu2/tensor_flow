import os

import matplotlib.pyplot as plt
import numpy as np
import skimage


def load_data(data_directory: str):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


def print_images_description(images_array: np.array):
    print("iamges dim:", images_array.ndim)
    print("images size:", images_array.size)
    print("first image:\n", images_array[0])


def print_label_description(labels_array: np.array, unique_labels):
    print("label dimensions:", labels_array.ndim)
    # Print the number of `labels`'s elements
    print("label size:", labels_array.size)
    # Count the number of labels
    print("number of unique label :", len(unique_labels))


def plot_labels_histogram(labels: np.array, size: int):
    plt.hist(labels, size)
    plt.show()


def show_some_images(images_array: np.array):
    traffic_signs = [300, 2250, 3650, 4000]

    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images_array[traffic_signs[i]])
        plt.subplots_adjust(wspace=0.5)

    plt.show()


def show_some_images_with_features(images_array: np.array):
    traffic_signs = [300, 2250, 3650, 4000]

    for i in range(len(traffic_signs)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images[traffic_signs[i]])
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                      images[traffic_signs[i]].min(),
                                                      images[traffic_signs[i]].max()))


if __name__ == '__main__':
    ROOT_PATH = "../data/"
    train_data_dir = os.path.join(ROOT_PATH, "Training")
    test_data_dir = os.path.join(ROOT_PATH, "Testing")

    images, labels = load_data(train_data_dir)

    images_array = np.array(images)
    labels_array = np.array(labels)
    unique_labels = set(labels_array)

    print_images_description(images_array)
    print_label_description(labels_array, unique_labels)

    # plot_labels_histogram(labels, len(unique_labels))

    show_some_images(images_array)
