import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.cluster.vq import kmeans2
def product_quantization(data, num_subspaces, num_clusters):
  """Performs product quantization on the given data.

  Args:
    data: The data to be quantized.
    num_subspaces: The number of subspaces.
    num_clusters: The number of clusters per subspace.

  Returns:
    A list of PQ codes, one for each data point.
  """

  codes = []
  for subspace in range(num_subspaces):
    code,label = kmeans2(data[:, subspace], num_clusters)
    codes.append(code)

  return codes

def index_pq_codes(codes, num_subspaces):
  """Builds an index for the given PQ codes.

  Args:
    codes: The PQ codes to be indexed.
    num_subspaces: The number of subspaces.

  Returns:
    A dictionary that maps from a PQ code to a list of data points.
  """

  index = {}
  for i, code in enumerate(codes):
    pq_code = tuple(code)
    if i not in index.keys():
      index[i] = []
    index[i].append(pq_code)

  tree = KDTree(codes)
  return index, tree

def search(index, tree, pq_code, k):
  """Searches the given index for the k nearest neighbors of the given PQ code.

  Args:
    index: The index to be searched.
    tree: The tree to be used for searching.
    pq_code: The PQ code to be searched for.
    k: The number of nearest neighbors to be returned.

  Returns:
    A list of the k nearest neighbors of the given PQ code.
  """

  neighbors = []
  distances, data_indices = tree.query([pq_code], k=k)
  for distance, data_index in zip(distances.reshape(-1), data_indices.reshape(-1)):
    neighbors.append((distance,data_index))

  neighbors.sort(key=lambda x: x[0])
  return neighbors[:k]

if __name__ == "__main__":
  data = np.random.rand(100, 3)
  codes = product_quantization(data, 3, 16)
  index, tree = index_pq_codes(codes, 3)

  neighbors = search(index, tree, codes[0], 3)
  for distance, data_index in neighbors:
    print(distance, data[data_index])