import timm
import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders

from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)), **kwargs):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(strides=strides)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights,
                    name,
                    list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        state_dict = model_zoo.load_url(settings["url"], map_location='cpu')
        try:
            from segmentation_models_pytorch_3d.utils.convert_weights import convert_2d_weights_to_3d
            state_dict = convert_2d_weights_to_3d(state_dict)
        except Exception:
            pass
        encoder.load_state_dict(state_dict)

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):

    all_settings = encoders[encoder_name]["pretrained_settings"]
    if pretrained not in all_settings.keys():
        raise ValueError("Available pretrained options {}".format(all_settings.keys()))
    settings = all_settings[pretrained]

    formatted_settings = {}
    formatted_settings["input_space"] = settings.get("input_space", "RGB")
    formatted_settings["input_range"] = list(settings.get("input_range", [0, 1]))
    formatted_settings["mean"] = list(settings["mean"])
    formatted_settings["std"] = list(settings["std"])

    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
