#Code modified from https://github.com/octopize/avatar-paper

import pandas as pd
import numpy as np
import statistics
from saiph.projection import fit_transform
from saiph.projection import transform
from math import log
from typing import Tuple, Union
import faiss
from numpy.typing import NDArray
import math

#Input: train, test and synthetic data as pd.read_csv() objects
#Produces the DCR and NNDR value
from copy import deepcopy
def calculate_metric(args, _real_data, _synthetic):
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    #Assign categorical values with type "category" for both real and synthetic data
    categorical_val_real, continuous_val_real = get_categorical_continuous(real_data)
    real_data[categorical_val_real] = real_data[categorical_val_real].astype("category")
    
    categorical_val_synth, continous_val_synth = get_categorical_continuous(synthetic)
    synthetic[categorical_val_synth] = synthetic[categorical_val_synth].astype("category")

    #Fit model and get coordinates
    real_coord, model = fit_transform(real_data, nf=2)
    synth_coord = transform(synthetic, model)

    results = statistics.mean(get_dcr(real_coord, synth_coord))
    
    if results == 0:
        return 1
    else:
        return 1 - sigmoid(log(results, 10))

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

#Return column names of the categorical(strings) and continuous(integers) attributes
def get_categorical_continuous(data: pd.DataFrame):
    categorical_val = []
    cont_val = []
    for column in data.columns:
            if data[column].dtype == 'O':
                categorical_val.append(column)
            else:
                cont_val.append(column)
    
    return categorical_val, cont_val


def get_dcr(train_coord, synth_coord):
    """Get distances to the closest records.

    DCR is the distance of each synthetic record to a record in the original dataset.
    """
    indices_distances = get_distances_closest_records(
        train_coord, synth_coord, searching_frame=1
    )
    _, distances = zip(*indices_distances)
    return [distance[0] for distance in distances]

def get_distances_closest_records(
    records: pd.DataFrame, synthetic: pd.DataFrame, searching_frame: int
):
    """Get index and distances of the closest records.

    Arguments
    ---------
        records: Original records
        synthetic: Synthetic data
        searching_frame: number of neighbors to find
    Returns
    -------
        indices_distances: indices, distances nearest neighbor among original records.

    """
    nn = FaissKNeighbors(k=searching_frame)
    nn.fit(np.array(records))

    # index.search returns two arrays (distances, indices)
    # https://github.com/facebookresearch/faiss/wiki/Getting-started
    distances, indices = nn.predict(synthetic.to_numpy().astype(np.float32))

    indices_distances = list(zip(indices, distances))
    return indices_distances


class FaissKNeighbors:
    index: Union[faiss.IndexFlatL2, faiss.IndexIVFFlat]

    def __init__(self, k: int = 5) -> None:
        self.index = faiss.IndexFlatL2()
        self.k = k

    def fit(self, X: NDArray[np.float_]) -> None:
        xb: NDArray[np.float_] = X.astype(np.float32)
        size, dimension = X.shape
        nlist = round(math.sqrt(size))
        threshold_size = 50000

        # Use Index for large size dataframe
        if size > threshold_size:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_L2
            )
            assert self.index is not None  # nosec: B101
            self.index.train(xb)

        # perform exhaustive search otherwise
        else:
            self.index = faiss.IndexFlatL2(dimension)

        assert self.index is not None  # nosec: B101
        self.index.add(xb)

    def predict(
        self, X: NDArray[np.float_]
    ) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
        assert self.index is not None  # nosec: B101
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        distances = np.sqrt(distances)
        return distances, indices
