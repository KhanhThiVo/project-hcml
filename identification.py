import numpy as np
import os
def compute_similarity(embedding1, embedding2):
    # Compute similarity between two embeddings.
    # You can use any similarity metric like cosine similarity, Euclidean distance, etc.
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def exhaustive_search(probe_embeddings, gallery_embeddings):
    identifications = []

    for probe_idx, probe_embedding in enumerate(probe_embeddings):
        best_match_idx = None
        best_similarity = -np.inf

        for gallery_idx, gallery_embedding in enumerate(gallery_embeddings):
            similarity = compute_similarity(probe_embedding, gallery_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = gallery_idx

        identifications.append((probe_idx, best_match_idx))
        

    return identifications

import csv

def open_tab_separated_file(file_path):
    data = dict()
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            data[row[0]]=row[1]
    return data

def main():
    probes_list=[]
    gallery_list=[]
    probe_embeddings=[]
    gallery_embeddings=[]
    identities=open_tab_separated_file('.\identities.txt')
    for probe in os.listdir('.\Probe'):
        if probe.endswith('.npy'):
            probes_list.append(probe)
            probe_embeddings.append(np.load(os.path.join('.\Probe',probe)))
    for gallery in os.listdir('.\Gallery\Gallery'):
        if gallery.endswith('.npy'):
            gallery_list.append(gallery)
            gallery_embeddings.append(np.load(os.path.join('.\Gallery\Gallery',gallery)))
    identifications=exhaustive_search(probe_embeddings,gallery_embeddings)
    correct_preds=0
    total_preds=0
    for probe_idx,gallery_idx in identifications:
        probe=probes_list[probe_idx]
        gallery=gallery_list[gallery_idx]
        probe_identity=identities[probe]
        gallery_identity=identities[gallery]
        if probe_identity==gallery_identity:
            correct_preds+=1
        total_preds+=1
    hit_rate=correct_preds/total_preds
    print(hit_rate)
        
    
main()