import pickle, sys
import keras.src.layers.core.dense as _kd_mod

_OrigDenseInit = _kd_mod.Dense.__init__

def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
    _OrigDenseInit(self, *args, **kwargs)

_kd_mod.Dense.__init__ = _patched_dense_init
_OrigDenseFromConfig = _kd_mod.Dense.from_config.__func__

@classmethod
def _patched_from_config(cls, config):
    config.pop('quantization_config', None)
    return _OrigDenseFromConfig(cls, config)

_kd_mod.Dense.from_config = _patched_from_config

model = pickle.load(open('resnet50_model.pkl','rb'))
model.summary()
print("Input shape expected:", model.input_shape)
