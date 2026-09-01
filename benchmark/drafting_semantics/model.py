from __future__ import annotations

from .dataset import EDGE_FEATURE_DIM
from .schema import EDGE_ROLES, PANEL_ROLES


DEFAULT_MODEL_CONFIG = {
    "width": 128,
    "heads": 4,
    "layers": 3,
    "feedforward_multiplier": 3,
    "dropout": 0.1,
    "maximum_edges": 40,
}


def build_model(config: dict):
    import torch

    class DraftingEdgeSemanticNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            self.feature_projection = torch.nn.Linear(EDGE_FEATURE_DIM, width)
            self.panel_role = torch.nn.Embedding(len(PANEL_ROLES), width)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=width,
                nhead=int(config["heads"]),
                dim_feedforward=width * int(config["feedforward_multiplier"]),
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=int(config["layers"]), enable_nested_tensor=False)
            self.edge_head = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, len(EDGE_ROLES)),
            )

        def forward(self, features, valid_mask, panel_role_ids):
            hidden = self.feature_projection(features)
            hidden = hidden + self.panel_role(panel_role_ids)[:, None, :]
            hidden = self.encoder(hidden, src_key_padding_mask=~valid_mask)
            return self.edge_head(hidden)

    return DraftingEdgeSemanticNet()

