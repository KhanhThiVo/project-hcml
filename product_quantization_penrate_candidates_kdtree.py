import csv
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, vq
from scipy.spatial.distance import cdist
import utils
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree


class ProductQuantization:

    def __init__(self, num_subspaces, num_centroids, identities_path, probes_path, gallery_path):
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.identities_path = identities_path
        # self.probes_path = probes_path
        # self.gallery_path = gallery_path

        self.probes_list, self.probe_embeddings, self.gallery_list, self.gallery_embeddings = utils.load_data_all(probes_path,
                                                                                              gallery_path)
        self.identities = utils.open_tab_separated_file(self.identities_path)

        self.codebook = self.PQ_train(self.gallery_embeddings, self.num_subspaces, self.num_centroids)
        self.PQ_code, self.kdtree = self.PQ_encode(self.gallery_embeddings, self.codebook)

    def PQ_train(self, vectors, num_subspaces, num_centroids):
        segment_length = int(vectors.shape[1] / num_subspaces)  # Dimension (or length) of a segment.
        codebook = np.empty((num_subspaces, num_centroids, segment_length), np.float32)

        for m in range(num_subspaces):
            sub_vectors = vectors[:, m * segment_length:(m + 1) * segment_length]  # Sub-vectors for segment m.
            codebook[m], label = kmeans2(sub_vectors, num_centroids)  # Run k-means clustering for each segment.
        print("Done train")
        return codebook

    def PQ_encode(self, vectors, codebook):
        num_subspaces, num_centroids, s = codebook.shape
        PQ_code = np.empty((vectors.shape[0], num_subspaces), np.uint8)

        for m in range(num_subspaces):
            sub_vectors = vectors[:, m * s:(m + 1) * s]  # Sub-vectors for segment m.
            centroid_ids, _ = vq(sub_vectors, codebook[m])  # vq returns the nearest centroid Ids.
            PQ_code[:, m] = centroid_ids  # Assign centroid Ids to PQ_code.
        print("Done encoding")
        # KD tree stuffs
        tree = cKDTree(PQ_code, leafsize=2, balanced_tree=True)
        return PQ_code, tree

    def PQ_search(self, query_vector, PQ_code, k=5):
        num_subspaces, num_centroids, s = self.codebook.shape
        # =====================================================================
        # Build the distance table.
        # =====================================================================

        distance_table = np.empty((num_subspaces, num_centroids), np.float32)  # Shape is (M, k)
        query_code = np.empty((1, num_subspaces), np.uint8)

        for m in range(num_subspaces):
            query_segment = query_vector[m * s:(m + 1) * s]  # Query vector for segment m.
            # KDtree stuffs
            centroid_ids, _ = vq(np.expand_dims(query_segment, axis=0),
                                 self.codebook[m])  # vq returns the nearest centroid Ids.
            query_code[:, m] = centroid_ids
            # distance_table[m] = cdist([query_segment], codebook[m], "sqeuclidean")[0]

        # KD tree stuffs
        distances, data_indices = self.kdtree.query(query_code, k=k)
        # PQ_code = PQ_code[data_indices][0]

        # Look up the partial distances from the distance table.
        # N, M = PQ_code.shape
        # distance_table = distance_table.T  # Transpose the distance table to shape (k, M)
        # distances = np.zeros((N,)).astype(np.float32)

        # for n in range(N):  # For each PQ Code, lookup the partial distances.
        #     for m in range(M):
        #         distances[n] += distance_table[PQ_code[n][m]][m]  # Sum the partial distances from all the segments.
        # print("Done search")
        # return distance_table, distances
        return data_indices, distances

    def run_search(self, penetration_rate: float):
        start_time = time.time()

        identifications = []
        k = int(len(self.gallery_list) * penetration_rate)
        print(f"knn: {k}")
        for probe_idx, probe_embedding in enumerate(tqdm(self.probe_embeddings)):
            # candidates = []

            # Do PQ search
            # distance_table, distances = self.PQ_search(probe_embedding, self.codebook, self.PQ_code)
            data_indices, distances = self.PQ_search(probe_embedding, self.PQ_code, k=k)

            # Sort distances
            distances_order = np.argsort(distances).tolist()

            # Get candidates according to penetration rate
            candidates = distances_order[:int(len(distances_order) * penetration_rate)]
            candidates = data_indices.tolist()[0]
            identifications.append((probe_idx, candidates))

        # Evaluation
        correct_preds = 0
        total_preds = 0
        for probe_idx, candidates in identifications:
            probe = self.probes_list[probe_idx]
            probe_identity = self.identities[probe]
            for gallery_idx in candidates:
                gallery = self.gallery_list[gallery_idx]
                gallery_identity = self.identities[gallery]
                if probe_identity == gallery_identity:
                    correct_preds += 1
                    break
            total_preds += 1
        hit_rate = correct_preds / total_preds
        end_time = time.time()

        return hit_rate, end_time - start_time


if __name__ == '__main__':
    seed = 143
    total_time_ran = 0
    fpath = os.path.dirname(__file__)
    data_path = os.path.join(fpath, 'IJBC_Split')
    identities_path = os.path.join(data_path, 'identities.txt')
    probes_path = os.path.join(data_path, 'Probe')
    gallery_path = os.path.join(data_path, 'Gallery')

    # probes_list, probe_embeddings, gallery_list, gallery_embeddings = load_data(probes_path, gallery_path, penetration_rate=1.0)

    M = 8  # Number of subspaces
    k = 256  # Number of centroids per subspace

    pq_model = ProductQuantization(num_subspaces=M, num_centroids=k, identities_path=identities_path, probes_path=probes_path, gallery_path=gallery_path)

    hit_rates = []
    penetration_rates = []
    total_start_time = time.time()
    for penetration_rate in np.arange(0.001, 0.35, 0.005):  # 0.001, 0.005, 0.001
        penetration_rate = np.around(penetration_rate, 3)
        print('penetration_rate= ' + str(penetration_rate))
        hit_rate, time_ran = pq_model.run_search(penetration_rate)
        print('hit_rate= ' + str(hit_rate))
        print(f'time ran: {time_ran}')
        hit_rates.append(hit_rate)
        penetration_rates.append(penetration_rate)
        total_time_ran += time_ran
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total time ran: {total_time}")
    root_name = 'product_quantization' + str(seed)
    np.save(root_name + '_penetration_rates.npy', np.array(penetration_rates))
    np.save(root_name + '_hit_rates.npy', np.array(hit_rates))
    plt.plot(penetration_rates, hit_rates)
    plt.xlabel('Penetration Rate')
    plt.ylabel('Hit Rate')
    plt.savefig('eval-pq_kdtree.png')
    plt.show()

