import ml_collections

def get_3DReg_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (2, 2, 2)})
    config.patches.grid = (8, 8, 8)
    config.hidden_size = 48
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = 8

    config.conv_first_channel = 48
    config.encoder_channels = (24, 24, 48)
    config.down_factor = 2
    config.down_num = 2
  #  config.decoder_channels = (96, 48, 32, 32, 16)
  #  config.skip_channels = (32, 32, 32, 32, 16)
    config.decoder_channels = (96, 24)
    config.skip_channels = (96, 24)
    config.n_dims = 3
    config.n_skip = 3

    return config
