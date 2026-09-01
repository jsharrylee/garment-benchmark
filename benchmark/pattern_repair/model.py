from __future__ import annotations


def build_model(config: dict):
    import torch

    class PatternRepairNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["width"])
            self.input_projection = torch.nn.Linear(10, width)
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
            self.output = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, 2),
            )
            self.max_displacement = float(config["max_displacement"])

        def forward(self, features, valid_mask):
            hidden = self.input_projection(features)
            hidden = self.encoder(hidden, src_key_padding_mask=~valid_mask)
            delta = torch.tanh(self.output(hidden)) * self.max_displacement
            return features[..., :2] + delta

    return PatternRepairNet()


DEFAULT_MODEL_CONFIG = {
    "width": 128,
    "heads": 4,
    "layers": 4,
    "feedforward_multiplier": 3,
    "dropout": 0.05,
    "max_displacement": 1.5,
    "maximum_nodes": 256,
}
