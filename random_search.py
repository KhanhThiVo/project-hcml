import csv
import numpy as np
import os
import time
import matplotlib.pyplot as plt

class ExhaustiveBaseline:

    def __init__(self):
        pass

    def compute_similarity(self, embedding1, embedding2):
        # Compute cosine similarity between two embeddings.
        # You can use any similarity metric like cosine similarity, Euclidean distance, etc.
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def exhaustive_search(self, probe_embeddings, gallery_embeddings):
        identifications = []

        for probe_idx, probe_embedding in enumerate(probe_embeddings):
            best_match_idx = None
            best_similarity = -np.inf

            for gallery_idx, gallery_embedding in enumerate(gallery_embeddings):
                similarity = self.compute_similarity(probe_embedding, gallery_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = gallery_idx
            identifications.append((probe_idx, best_match_idx))
        return identifications
    def open_tab_separated_file(self, file_path):
        data = dict()
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                data[row[0]] = row[1]
        return data

    def run_search(self, identities_path, probes_path, gallery_path,penetration_rate):
        start_time = time.time()
        probes_list = []
        gallery_list = []
        probe_embeddings = []
        gallery_embeddings = []
        identities = self.open_tab_separated_file(identities_path)

        gallery_files=os.listdir(gallery_path)
        np.random.seed(42)
        np.random.shuffle(gallery_files)
        gallery_files=gallery_files[:int(len(gallery_files)*penetration_rate)]
        for probe in os.listdir(probes_path):
            if probe.endswith('.npy'):
                probes_list.append(probe)
                probe_embeddings.append(np.load(os.path.join(probes_path, probe)))
        for gallery in gallery_files:
            if gallery.endswith('.npy'):
                gallery_list.append(gallery)
                gallery_embeddings.append(np.load(os.path.join(gallery_path, gallery)))

        identifications = self.exhaustive_search(probe_embeddings, gallery_embeddings)

        # Evaluation
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
        hit_rate = correct_preds/total_preds

        end_time = time.time()
        return hit_rate, end_time - start_time       
    
if __name__ == '__main__':
    seed=42
    
    fpath = os.path.dirname(__file__)
    data_path = fpath
    identities_path = os.path.join(data_path, 'identities.txt')
    probes_path = os.path.join(data_path, 'Probe')
    gallery_path = os.path.join(data_path, 'Gallery/Gallery')

    baseline = ExhaustiveBaseline()
    hit_rates=[]
    penetration_rates=[]
    for penetration_rate in np.arange(0.01,1.01,0.01):
        penetration_rate=np.around(penetration_rate,2)
        print('penetration_rate= '+str(penetration_rate))
        hit_rate, time_ran = baseline.run_search(identities_path, probes_path, gallery_path,penetration_rate)
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


