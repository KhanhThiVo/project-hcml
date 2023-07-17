import csv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm


class ExhaustiveBaseline:

    def __init__(self, identities_path, probes_path, gallery_path):
        self.identities_path = identities_path
        # self.probes_path = probes_path
        # self.gallery_path = gallery_path

        self.probes_list, self.probe_embeddings, self.gallery_list, self.gallery_embeddings = utils.load_data_all(
            probes_path,
            gallery_path)
        self.identities = utils.open_tab_separated_file(self.identities_path)

    def compute_similarity(self, embedding1, embedding2):
        # Compute cosine similarity between two embeddings.
        # You can use any similarity metric like cosine similarity, Euclidean distance, etc.
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def exhaustive_search(self, probe_embedding, penetration_rate):
        similarities = []
        for gallery_idx, gallery_embedding in enumerate(self.gallery_embeddings):
            similarity = self.compute_similarity(probe_embedding, gallery_embedding)
            similarities.append(similarity)
        similarities_order = np.array(similarities)
        similarities_order = np.argsort(similarities_order).tolist()
        candidates = similarities_order[-int(len(similarities_order) * penetration_rate):]
        # identifications.append((probe_idx, best_match_idx))
        return candidates

    def run_search(self, penetration_rate: float):
        start_time = time.time()

        identifications = []

        for probe_idx, probe_embedding in enumerate(tqdm(self.probe_embeddings)):

            candidates = self.exhaustive_search(probe_embedding, penetration_rate)
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
        hit_rate = correct_preds/total_preds

        end_time = time.time()
        return hit_rate, end_time - start_time       


if __name__ == '__main__':
    seed = 142
    
    fpath = os.path.dirname(__file__)
    data_path = os.path.join(fpath, 'IJBC_Split')
    identities_path = os.path.join(data_path, 'identities.txt')
    probes_path = os.path.join(data_path, 'Probe')
    gallery_path = os.path.join(data_path, 'Gallery')

    baseline_model = ExhaustiveBaseline(identities_path=identities_path, probes_path=probes_path, gallery_path=gallery_path)

    hit_rates = []
    penetration_rates = []
    for penetration_rate in np.arange(0.001, 0.005, 0.001):
        penetration_rate = np.around(penetration_rate, 3)
        print('penetration_rate= '+str(penetration_rate))
        hit_rate, time_ran = baseline_model.run_search(penetration_rate)
        print('hit_rate= '+str(hit_rate))
        hit_rates.append(hit_rate)
        penetration_rates.append(penetration_rate)
    root_name='random_search'+str(seed)
    np.save(root_name+'_penetration_rates.npy',np.array(penetration_rates))
    np.save(root_name+'_hit_rates.npy',np.array(hit_rates))
    plt.plot(penetration_rates,hit_rates)
    plt.xlabel('Penetration Rate')
    plt.ylabel('Hit Rate')
    plt.savefig('Random_Indexing.png')
    plt.show()


