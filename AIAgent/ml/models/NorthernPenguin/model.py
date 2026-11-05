import torch
from torch.nn import Linear
from torch_geometric.nn import TAGConv, SAGEConv, RGCNConv, ResGatedGraphConv, GCN
from ml.schema import NUM_PC_FEATURES


class StateModelEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_of_state_features: int,
        num_hops_1: int,
        num_hops_2: int,
        normalization: bool,
        num_pc_layers: int,
    ):
        super().__init__()
        # PyG's MessagePassing is not an ABC (plain type metaclass): the
        # @abstractmethod on message_and_aggregate/edge_update is decorative,
        # and the convs instantiate fine at runtime (verified). ty treats the
        # wider-signature overrides as unimplemented and marks the classes
        # abstract.
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, 3)  # ty: ignore[call-non-callable]
        self.conv10 = TAGConv(7, hidden_channels, num_hops_1, normalize=normalization)  # ty: ignore[call-non-callable]
        self.conv2 = TAGConv(  # ty: ignore[call-non-callable]
            hidden_channels, hidden_channels, num_hops_2, normalize=normalization
        )
        self.conv3 = ResGatedGraphConv(  # ty: ignore[call-non-callable]
            (hidden_channels, 6), hidden_channels, edge_dim=2
        )
        self.conv32 = SAGEConv(  # ty: ignore[call-non-callable]
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.conv4 = SAGEConv(  # ty: ignore[call-non-callable]
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.conv42 = SAGEConv(  # ty: ignore[call-non-callable]
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.conv5 = SAGEConv(hidden_channels, hidden_channels, normalize=normalization)  # ty: ignore[call-non-callable]
        self.pc_conv = GCN(
            NUM_PC_FEATURES,
            hidden_channels,
            num_layers=num_pc_layers,
            out_channels=hidden_channels,
        )
        self.pc_to_state = SAGEConv(  # ty: ignore[call-non-callable]
            (hidden_channels, hidden_channels), hidden_channels, normalize=normalization
        )
        self.lin = Linear(hidden_channels, num_of_state_features)
        self.lin_last = Linear(num_of_state_features, 1)

    # torch declares Module.forward as a class attribute (Callable[..., Any]
    # = _forward_unimplemented); defining it as a method is the documented
    # torch pattern.
    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        game_x: torch.Tensor,
        state_x: torch.Tensor,
        pc_x: torch.Tensor,
        edge_index_v_v: torch.Tensor,
        edge_type_v_v: torch.Tensor,
        edge_index_history_v_s: torch.Tensor,
        edge_attr_history_v_s: torch.Tensor,
        edge_index_in_v_s: torch.Tensor,
        edge_index_s_s: torch.Tensor,
        edge_index_pc_pc: torch.Tensor,
        edge_index_pc_s: torch.Tensor,
    ):
        game_x = self.conv10(game_x, edge_index_v_v).relu()

        # RGCNConv on empty edges is not the identity (it returns x@root+bias),
        # and empty v-v edges are realistic (null_if_empty in ml/dataset/tensors.py),
        # so skipping conv1 must stay data-driven. A Python `if` on a tensor
        # property makes the ONNX tracer bake the sample's value in with a
        # TracerWarning; torch.where keeps the skip data-driven (the exporter
        # traces the condition as a Shape-based op) without one. numel() is a
        # Python int, so the condition tensor must be created on the input's
        # device explicitly — torch.where rejects a CPU condition with CUDA
        # operands.
        has_v_v_edges = (edge_type_v_v.numel() > 0) * torch.ones(
            1, dtype=torch.bool, device=edge_type_v_v.device
        )
        game_x = torch.where(
            has_v_v_edges,
            self.conv1(game_x, edge_index_v_v, edge_type_v_v).relu(),
            game_x,
        )
        pc_x = self.pc_conv(pc_x, edge_index_pc_pc).relu()
        state_x = self.conv3(
            (game_x, state_x),
            edge_index_history_v_s,
            edge_attr_history_v_s,
        ).relu()

        state_x = self.conv32(
            (game_x, state_x),
            edge_index_history_v_s,
        ).relu()

        state_x = self.conv4(
            (game_x, state_x),
            edge_index_in_v_s,
        ).relu()

        state_x = self.conv42(
            (game_x, state_x),
            edge_index_in_v_s,
        ).relu()

        state_x = self.pc_to_state((pc_x, state_x), edge_index_pc_s).relu()
        state_x = self.conv2(
            state_x,
            edge_index_s_s,
        ).relu()

        state_x = self.conv5(
            state_x,
            edge_index_s_s,
        ).relu()
        state_x = self.lin(state_x).relu()
        return self.lin_last(state_x)
