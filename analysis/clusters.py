import argparse
import csv
import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import misc
from scipy.misc import imresize
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
from tqdm import tqdm

import models
import random

DATA_FOLDER = '/home/alex/datasets/1miohands/clean'


def load_files():
    image_files, labels = [], []
    for video_dir in os.listdir(DATA_FOLDER):
        cropped_path = os.path.join(DATA_FOLDER, video_dir, 'cropped')
        if os.path.isdir(cropped_path):
            for file in os.listdir(cropped_path):
                if file.endswith('_r.png'):
                    image_files.append(os.path.join(cropped_path, file))
                    label = file.split('_l_')[1].split('_')[0]
                    labels.append(label)
    return image_files, labels


def pad_img(img):
    dif_x = 64 - img.shape[0]
    dif_y = 64 - img.shape[1]

    if dif_x > 0 or dif_y > 0:
        if dif_y > 10 or dif_x > 10:
            print(dif_x, dif_y)
        img = np.pad(img, ((0, dif_x), (0, dif_y), (0, 0)), mode='constant')
    if img.shape[0] > 64 or img.shape[1] > 64:
        img = imresize(img, (64, 64, 3))

    return img


def load_image(image_path):
    image = misc.imread(image_path).astype(np.float32)
    image = pad_img(image)
    image = image / 127.5 - 1.

    return np.array(image)


def plot_clusters(coordinates, labels):
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, colors=labels, s=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='clusters')
    parser.add_argument('--model', type=str, default='ALI')
    parser.add_argument('--weights', type=str,
                        default='/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_32/ali/weights/epoch_00020')
    parser.add_argument('--z_dims', type=int, default=32)
    args = parser.parse_args()

    image_files, labels = load_files()
    labels_counter = Counter(labels)
    print(len(image_files))
    print(labels_counter)

    min_occurences = 500
    filtered_image_files, filtered_labels = [], []
    random.shuffle(image_files)
    for idx, image_file in enumerate(image_files):
        label = labels[idx]
        if labels_counter[label] >= min_occurences:
            filtered_image_files.append(image_file)
            filtered_labels.append(label)
    print(len(filtered_image_files))

    max_samples = 50000
    images = [load_image(image_file) for image_file in tqdm(image_files)][:max_samples]
    images = np.array(images)
    labels = filtered_labels[:max_samples]
    print(images.shape)

    model = getattr(models, args.model)(
        input_shape=(64, 64, 3),
        z_dims=args.z_dims,
        output=''
    )
    model.load_model(args.weights)
    encodings = model.f_Gz.predict(images)
    encodings = encodings[0]

    # clusters = defaultdict(list)
    # intras, inters = [], []
    # for idx, label in enumerate(labels):
    #     clusters[label].append(encodings[idx])
    # for k, cluster in clusters.items():
    #     cluster_np = np.array(cluster)
    #     clusters[k] = cluster_np
    # for cluster in clusters.values():
    #     # https://stackoverflow.com/questions/35062371/python-intra-similarity
    #     dist_matrix = pairwise.pairwise_distances(cluster, metric='euclidean')
    #     cluster_intra = dist_matrix.sum() / (cluster.shape[0] ** 2 - cluster.shape[0])
    #     intras.append(cluster_intra)
    #     print(cluster_intra)
    # for other_cluster in clusters.values():
    #     dist_matrix = pairwise.pairwise_distances(cluster, other_cluster, metric='euclidean')
    #     cluster_inter = dist_matrix.mean()
    #     inters.append(cluster_inter)
    #     print(cluster_inter)
    #
    # intras = np.array(intras)
    # inters = np.array(inters)
    #
    # # write results to csv
    # results_path = 'clusters.csv'
    # csv_exists = os.path.isfile(results_path)
    # with open(results_path, 'a+') as csv_file:
    #     writer = csv.writer(csv_file)
    #     if not csv_exists:
    #         writer.writerow(['Z Dims', 'Mean Intra Similarity', 'Mean Inter Similarity'])
    #     writer.writerow([args.z_dims, intras.mean(), inters.mean()])

    from sklearn.manifold import TSNE
    print('tsne')
    tsne_data = TSNE(n_components=2).fit_transform(encodings)
    plot_clusters(tsne_data, labels)

    pca = PCA(n_components=2)
    pca.fit(encodings)
    pca_data = pca.transform(encodings)
    plot_clusters(pca_data, labels)


if __name__ == '__main__':
    main()
