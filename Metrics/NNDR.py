#Code modified from https://github.com/octopize/avatar-paper

import pandas as pd
import numpy as np
import statistics
from saiph.projection import fit_transform
from saiph.projection import transform
import math
from typing import Tuple, Union
import faiss
from numpy.typing import NDArray

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
    #fit a model and transfor real and synthetic data into coordinates
    real_coord, model = fit_transform(real_data, nf=2)
    synth_coord = transform(synthetic, model)
    results = statistics.mean(get_nndr(real_coord, synth_coord))

    return (1-results)

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


def get_nndr(train_coord, synth_coord):
    """Get nearest neighbors distance ratio.

    Ratio of the distance of each synthetic record to its closest
    to the second closest record in the original dataset.
    """
    indices_distances = get_distances_closest_records(
        train_coord, synth_coord, searching_frame=2
    )
    _, distances = zip(*indices_distances)

    ratio = [
        1 if distance[1] == 0 else distance[0] / distance[1] for distance in distances
    ]
    return ratio


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



