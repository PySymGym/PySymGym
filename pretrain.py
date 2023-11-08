from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.models.TAGSageSimple.model import StateModelEncoder
from ml.predict_state_vector_hetero import PredictStateVectorHetGNN


def get_data_hetero_vector():
    dl = ServerDataloaderHeteroVector(
        "/home/cyfra/PycharmProjects/symbolic_exec_GNN-main/SerializedEpisodes_10.10.2023"
    )
    return dl.dataset


if __name__ == "__main__":
    # get_data_hetero_vector()
    pr = PredictStateVectorHetGNN(StateModelEncoder, 32)
    pr.train_and_save("../dataset", 20, "./ml/models/")
