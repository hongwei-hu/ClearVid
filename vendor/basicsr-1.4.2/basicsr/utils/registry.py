# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501


class Registry():
    """Simple registry with optional suffix support."""

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix
        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered in '{self._name}' registry!"
        )
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        if obj is None:
            def deco(func_or_class):
                self._do_register(func_or_class.__name__, func_or_class, suffix)
                return func_or_class

            return deco

        self._do_register(obj.__name__, obj, suffix)

    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None and isinstance(suffix, str):
            ret = self._obj_map.get(name + '_' + suffix)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')