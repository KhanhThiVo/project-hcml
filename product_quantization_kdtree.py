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

# References:
# product quantization code from https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd
class ProductQuantization:

    def __init__(self, num_subspaces, num_centroids, identities_path, probes_path, gallery_path):
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.identities_path = identities_path
        self.probes_path = probes_path
        self.gallery_path = gallery_path

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
        tree=KDTree(PQ_code,leaf_size=1)
        return PQ_code,tree

    def PQ_search(self, query_vector, codebook, PQ_code,tree,k,max_penetration_rate):
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
        n_calls=tree.get_n_calls()
        arrays=tree.get_arrays()
        # scipy kdtree has the attrşbute of size. Tried different leaf sizes and got the same size when m multiplying the length of these 2 arrays.
        size=len(arrays[2])*len(arrays[3]) 
        if n_calls/size>max_penetration_rate:
            max_penetration_rate=n_calls/size
        #extract nearest_pq_codes
        tree.reset_n_calls()
        PQ_code=PQ_code[data_indices][0]
        N, M = PQ_code.shape
        distance_table = distance_table.T  # Transpose the distance table to shape (k, M)
        distances = np.zeros((N,)).astype(np.float32)

        for n in range(N):  # For each PQ Code, lookup the partial distances.
            for m in range(M):
                distances[n] += distance_table[PQ_code[n][m]][m]  # Sum the partial distances from all the segments.
        
        return distance_table, distances,data_indices,max_penetration_rate
    def run_encoding(self):
        
        probes_list, probe_embeddings, gallery_list, gallery_embeddings = utils.load_data(self.probes_path, self.gallery_path,
                                                                                    penetration_rate=1.0)
        identities = utils.open_tab_separated_file(self.identities_path)

        codebook = self.PQ_train(gallery_embeddings, self.num_subspaces, self.num_centroids)
        PQ_code,tree = pq_model.PQ_encode(gallery_embeddings, codebook)
        return probe_embeddings,codebook,PQ_code,tree,probes_list,gallery_list,identities
    def run_search(self, tree_k,probe_embeddings,codebook,PQ_code,tree,probes_list,gallery_list,identities):
        
        start_time = time.time()
        identifications = []
        max_penetration_rate=0
        for probe_idx, probe_embedding in enumerate(tqdm(probe_embeddings)):
            distance_table, distances,data_indices,max_penetration_rate = pq_model.PQ_search(probe_embedding, codebook, PQ_code,tree,tree_k,max_penetration_rate)
            best_match_idx = data_indices[0][np.argmin(distances)]
            identifications.append((probe_idx, best_match_idx))
        print('max penetration rate',max_penetration_rate)
        #  Evaluation
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
        print('elapsed time in seconds:', end_time-start_time)

        return hit_rate, end_time - start_time, max_penetration_rate


if __name__ == '__main__':
    seed = 42
    total_time_ran = 0
    fpath = os.path.dirname(__file__)
    data_path = os.path.join(fpath, 'IJBC_Split')
    identities_path = os.path.join(data_path, 'identities.txt')
    probes_path = os.path.join(data_path, 'Probe')
    gallery_path = os.path.join(data_path, 'Gallery')

    # probes_list, probe_embeddings, gallery_list, gallery_embeddings = load_data(probes_path, gallery_path, penetration_rate=1.0)

    M = 4  # Number of subspaces
    k = 256  # Number of centroids per subspace
    tree_k=1
    pq_model = ProductQuantization(num_subspaces=M, num_centroids=k, identities_path=identities_path, probes_path=probes_path, gallery_path=gallery_path)
    probe_embeddings,codebook,PQ_code,tree,probes_list,gallery_list,identities=pq_model.run_encoding()
    hit_rates = []
    penetration_rates = []
    max_penetration_rate=0
    while max_penetration_rate<0.8: # Iterate to 80 percent penetration rate. It takes a lot of time until it gets to 80%
        print('nearest neighbour= ' + str(tree_k))
        hit_rate, time_ran,max_penetration_rate = pq_model.run_search(tree_k,probe_embeddings,codebook,PQ_code,tree,probes_list,gallery_list,identities)
        print('hit_rate= ' + str(hit_rate))
        print()
        hit_rates.append(hit_rate)
        penetration_rates.append(max_penetration_rate)
        total_time_ran += time_ran
        if tree_k==1:
            tree_k+=199
        else:
            tree_k+=200
        if tree_k>=len(gallery_list):  # If nearest neighbour is close to the number of lenght of the gallery list do not wait to reach the pre specified pen rate
            break
    print(f"Total time ran: {total_time_ran}")
    root_name = 'product_quantization' + str(seed)
    np.save(root_name + '_leafsize_1_penetration_rates.npy', np.array(penetration_rates))
    np.save(root_name + '_leafsize_1hit_rates.npy', np.array(hit_rates))
    plt.plot(penetration_rates, hit_rates)
    plt.xlabel('Penetration Rate')
    plt.ylabel('Hit Rate')
    plt.savefig('PQ_with_kdtree_leafsize_1.png')
    plt.show()

