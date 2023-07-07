import csv
import os
import numpy as np


def open_tab_separated_file(file_path):
    data = dict()
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            data[row[0]] = row[1]
    return data


def load_data(probes_path, gallery_path, penetration_rate):
    probes_list = []
    gallery_list = []
    probe_embeddings = []
    gallery_embeddings = []

    gallery_files = os.listdir(gallery_path)
    np.random.seed(42)
    np.random.shuffle(gallery_files)
    gallery_files = gallery_files[:int(len(gallery_files) * penetration_rate)]
    for probe in os.listdir(probes_path):
        if probe.endswith('.npy'):
            probes_list.append(probe)
            probe_embeddings.append(np.load(os.path.join(probes_path, probe)))
    for gallery in gallery_files:
        if gallery.endswith('.npy'):
            gallery_list.append(gallery)
            gallery_embeddings.append(np.load(os.path.join(gallery_path, gallery)))
    print("Data loaded")
    return probes_list, np.array(probe_embeddings), gallery_list, np.array(gallery_embeddings)