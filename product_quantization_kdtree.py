import csv
import numpy as np
#import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, vq
from scipy.spatial.distance import cdist
import utils
from sklearn.neighbors import KDTree

class ProductQuantization:

    def __init__(self, num_subspaces, num_centroids, identities_path, probes_path, gallery_path):
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.identities_path = identities_path
        self.probes_path = probes_path
        self.gallery_path = gallery_path

    """def product_quantization(self, data, num_subspaces, num_centroids):
        num_samples, dim = data.shape
        # subspace_dim = dim // num_subspaces

        # Split the data into subspaces
        subspaces = np.split(data, num_subspaces, axis=1)

        # Perform quantization for each subspace
        quantized_codes = []
        codebooks = []

        for subspace_data in subspaces:
            # Apply K-means clustering to quantize the subspace data
            kmeans = KMeans(n_clusters=num_centroids)
            kmeans.fit(subspace_data)
            quantized_subspace = kmeans.predict(subspace_data)

            quantized_codes.append(quantized_subspace)
            codebooks.append(kmeans.cluster_centers_)

        # Combine the quantized codes of all subspaces
        quantized_codes = np.concatenate(quantized_codes, axis=1)
        codebooks = np.concatenate(codebooks)

        return quantized_codes, codebooks"""

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
        index = {}
        for m in range(num_subspaces):
            
            sub_vectors = vectors[:, m * s:(m + 1) * s]  # Sub-vectors for segment m.
            centroid_ids, _ = vq(sub_vectors, codebook[m])  # vq returns the nearest centroid Ids.
            PQ_code[:, m] = centroid_ids  # Assign centroid Ids to PQ_code.
            index[m] = centroid_ids
        print("Done encoding")
        tree = KDTree(PQ_code)
        return PQ_code,tree

    def PQ_search(self, query_vector, codebook, PQ_code,tree,k):
        num_subspaces, num_centroids, s = codebook.shape
        # =====================================================================
        # Build the distance table.
        # =====================================================================

        distance_table = np.empty((num_subspaces, num_centroids), np.float32)  # Shape is (M, k)
        query_code = np.empty((1, num_subspaces), np.uint8)
        for m in range(num_subspaces):
            query_segment = query_vector[m * s:(m + 1) * s]  # Query vector for segment m.
            centroid_ids, _ = vq(np.expand_dims(query_segment,axis=0), codebook[m])  # vq returns the nearest centroid Ids.
            query_code[:,m]=centroid_ids
            distance_table[m] = cdist([query_segment], codebook[m], "sqeuclidean")[0]

        # =====================================================================
        # Look up the partial distances from the distance table.
        # =====================================================================
        

            
        distances, data_indices = tree.query(query_code, k=k)
        #extract nearest_pq_codes
        deneme_indices=np.unique(data_indices)
        PQ_code=PQ_code[data_indices][0]
        N, M = PQ_code.shape
        distance_table = distance_table.T  # Transpose the distance table to shape (k, M)
        distances = np.zeros((N,)).astype(np.float32)

        for n in range(N):  # For each PQ Code, lookup the partial distances.
            for m in range(M):
                distances[n] += distance_table[PQ_code[n][m]][m]  # Sum the partial distances from all the segments.
        #print("Done search")
        
        return distance_table, distances,data_indices

    def run_search(self, penetration_rate: float):
        start_time = time.time()
        probes_list, probe_embeddings, gallery_list, gallery_embeddings = utils.load_data(self.probes_path, self.gallery_path,
                                                                                    penetration_rate=penetration_rate)
        identities = utils.open_tab_separated_file(self.identities_path)

        codebook = self.PQ_train(gallery_embeddings, self.num_subspaces, self.num_centroids)
        PQ_code,tree = pq_model.PQ_encode(gallery_embeddings, codebook)

        identifications = []

        for probe_idx, probe_embedding in enumerate(tqdm(probe_embeddings)):
            # best_match_idx = None
            # min_distance = np.inf

            distance_table, distances,data_indices = pq_model.PQ_search(probe_embedding, codebook, PQ_code,tree,k=50)
            min_distance = np.min(distances)
            best_match_idx = data_indices[0][np.argmin(distances)]
            identifications.append((probe_idx, best_match_idx))

        # Evaluation
        print(tree.get_tree_stats())
        correct_preds = 0
        total_preds = 0
        for probe_idx, gallery_idx in identifications:
            probe = probes_list[probe_idx]
            gallery = gallery_list[gallery_idx]
            probe_identity = identities[probe]
            gallery_identity = identities[gallery]
            if probe_identity == gallery_identity:
                correct_preds += 1
            total_preds += 1
        hit_rate = correct_preds / total_preds
        end_time = time.time()

        return hit_rate, end_time - start_time

    """    
    M = 8  # Number of segments
    k = 256  # Number of centroids per segment
    vector_dim = 128  # Dimension (length) of a vector
    total_vectors = 1000000  # Number of database vectors
    
    # Generate random vectors
    np.random.seed(42)
    vectors = np.random.random((total_vectors, vector_dim)).astype(np.float32)  # Database vectors
    q = np.random.random((vector_dim,)).astype(np.float32)  # Query vector
    
    # Train, encode and search with Product Quantization
    codebook = PQ_train(vectors, M, k)
    PQ_code = PQ_encode(vectors, codebook)
    distance_table, distances = PQ_search(q, codebook, PQ_code)
    # All the distances are returned, you may sort them to get the shortest distance.
    """


if __name__ == '__main__':
    seed = 42

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
    for penetration_rate in np.arange(1.0, 1.01, 0.02):  # step=0.01
        penetration_rate = np.around(penetration_rate, 2)
        print('penetration_rate= ' + str(penetration_rate))
        hit_rate, time_ran = pq_model.run_search(penetration_rate)
        print('hit_rate= ' + str(hit_rate))
        hit_rates.append(hit_rate)
        penetration_rates.append(penetration_rate)
    root_name = 'product_quantization' + str(seed)
    np.save(root_name + '_penetration_rates.npy', np.array(penetration_rates))
    np.save(root_name + '_hit_rates.npy', np.array(hit_rates))
    plt.plot(penetration_rates, hit_rates)
    plt.xlabel('Penetration Rate')
    plt.ylabel('Hit Rate')
    plt.savefig('Random_Indexing.png')
    plt.show()

