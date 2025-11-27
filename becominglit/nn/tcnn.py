from typing import Optional

import tinycudann as tcnn


def make_tcnn_mlp(
    in_ch: int,
    out_ch: int,
    hidden_ch: int,
    n_hidden: int,
    activation: str = "LeakyReLU",
    output_activation: Optional[str] = None,
):
    net_config = {
        "otype": "FullyFusedMLP",
        "activation": activation,
        "output_activation": output_activation,
        "n_neurons": hidden_ch,
        "n_hidden_layers": n_hidden,
    }
    return tcnn.Network(in_ch, out_ch, net_config)
