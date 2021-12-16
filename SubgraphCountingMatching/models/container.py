import operator
import torch as th
import torch.nn as nn
import warnings

from collections import OrderedDict
from dataclasses import fields
from itertools import islice, chain
from torch._jit_internal import _copy_to_script_wrapper
from torch._six import container_abcs
from torch.nn.modules.container import Container, Sequential, ModuleDict, ModuleList, ParameterDict, ParameterList


class OutputDict(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, th.Tensor):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            # inner_dict = {k: v for (k, v) in self.items()}
            # return inner_dict[k]
            return super().__getitem__(k)
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class BufferDict(nn.Module):
    def __init__(self, buffers=None):
        super(BufferDict, self).__init__()
        if buffers is not None:
            self.update(buffers)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer)

    def __delitem__(self, key):
        del self._buffers[key]

    def __setattr__(self, key, value):
        if not isinstance(value, th.Tensor):
            warnings.warn("Setting attributes on BufferDict is not supported.")
        super(BufferDict, self).__setattr__(key, value)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        self._buffers.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def keys(self):
        return self._buffers.keys()

    def items(self):
        return self._buffers.items()

    def values(self):
        return self._buffers.values()

    def update(self, buffers):
        if not isinstance(buffers, container_abcs.Iterable):
            raise TypeError(
                "BufferDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, (OrderedDict, BufferDict)):
            for key, buffer in buffers.items():
                self[key] = buffer
        elif isinstance(buffers, container_abcs.Mapping):
            for key, buffer in sorted(buffers.items()):
                self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Buffer containing: [{} of size {}{}]'.format(th.typename(p), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('BufferDict should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn(
            "nn.BufferDict is being used with DataParallel but this is not "
            "supported. This dict will appear empty for the models replicated "
            "on each GPU except the original one."
        )

        return super(BufferDict, self)._replicate_for_data_parallel()


class BufferList(nn.Module):
    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self += buffers

    def _get_abs_string_index(self, idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._buffers.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._buffers[str(idx)]

    def __setitem__(self, idx, buffer):
        idx = self._get_abs_string_index(idx)
        return self.register_buffer(str(idx), buffer)

    def __setattr__(self, key, value):
        if not isinstance(value, th.Tensor):
            warnings.warn("Setting attributes on BufferList is not supported.")
        super(BufferList, self).__setattr__(key, value)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def __iadd__(self, buffers):
        return self.extend(buffers)

    def __dir__(self):
        keys = super(BufferList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, buffer):
        self.register_buffer(str(len(self)), buffer)
        return self

    def extend(self, buffers):
        if not isinstance(buffers, container_abcs.Iterable):
            raise TypeError("BufferList.extend should be called with an " "iterable, but got " + type(buffers).__name__)
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Buffer containing: [{} of size {}{}]'.format(th.typename(p), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('BufferList should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn(
            "nn.BufferList is being used with DataParallel but this is not "
            "supported. This list will appear empty for the models replicated "
            "on each GPU except the original one."
        )

        return super(BufferList, self)._replicate_for_data_parallel()


class MixtureDict(nn.Module):
    def __init__(self, input=None):
        super(MixtureDict, self).__init__()
        self._module_dict = ModuleDict()
        self._param_dict = ParameterDict()
        self._buffer_dict = BufferDict()

        if input is not None:
            self.update(input)

    @_copy_to_script_wrapper
    def __getitem__(self, key):
        if key in self._buffer_dict:
            return self._buffer_dict[key]
        if key in self._param_dict:
            return self._param_dict[key]
        elif key in self._module_dict:
            return self._module_dict[key]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        # nn.Parameter is also an instance of th.Tensor
        if isinstance(value, nn.Parameter):
            self._param_dict.register_parameter(key, value)
        elif isinstance(value, th.Tensor):
            self._buffer_dict.register_buffer(key, value)
        else:
            self._module_dict.add_module(key, value)

    def __delitem__(self, key):
        if key in self._buffer_dict:
            del self._buffer_dict[key]
        if key in self._param_dict:
            del self._param_dict[key]
        elif key in self._module_dict:
            del self._module_dict[key]
        else:
            raise KeyError

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._buffer_dict) + len(self._param_dict) + len(self._module_dict)

    @_copy_to_script_wrapper
    def __iter__(self):
        return chain(iter(self._buffer_dict), iter(self._param_dict), iter(self._module_dict))

    @_copy_to_script_wrapper
    def __contains__(self, key):
        return key in self._buffer_dict or key in self._param_dict or key in self._module_dict

    def clear(self):
        self._buffer_dict.clear()
        self._param_dict.clear()
        self._module_dict.clear()

    def pop(self, key):
        if key in self._buffer_dict:
            return self._buffer_dict.pop(key)
        elif key in self._param_dict:
            return self._param_dict.pop(key)
        elif key in self._module_dict:
            return self._module_dict.pop(key)
        else:
            raise KeyError

    @_copy_to_script_wrapper
    def keys(self):
        return chain(self._buffer_dict.keys(), self._param_dict.keys(), self._module_dict.keys())

    @_copy_to_script_wrapper
    def items(self):
        return chain(self._buffer_dict.items(), self._param_dict.items(), self._module_dict.items())

    @_copy_to_script_wrapper
    def values(self):
        chain(self._buffer_dict.values(), self._param_dict.values(), self._module_dict.values())

    def update(self, values):
        if not isinstance(values, container_abcs.Iterable):
            raise TypeError(
                "MixtureDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(values).__name__
            )

        if isinstance(values, (OrderedDict, container_abcs.Mapping)):
            for k, v in values.items():
                self[k] = v
        elif isinstance(values, MixtureDict):
            for k, v in values._buffer_dict.items():
                self._buffer_dict[k] = v
            for k, v in values._param_dict.items():
                self._param_dict[k] = v
            for k, v in values._module_dict.items():
                self._module_dict[k] = v
        elif isinstance(values, BufferDict):
            for k, v in values.items():
                self._buffer_dict[k] = v
        elif isinstance(values, ParameterDict):
            for k, v in values.items():
                self._param_dict[k] = v
        elif isinstance(values, ModuleDict):
            for k, v in values.items():
                self._module_dict[k] = v
        else:
            for j, m in enumerate(values):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "MixtureDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "MixtureDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                self[m[0]] = m[1]

    def forward(self):
        raise NotImplementedError()


class Parallel(nn.Module):
    def __init__(self, *args):
        super(Parallel, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Parallel, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        output = []
        for module in self:
            output.append(module(input))
        return th.cat(output, dim=-1)
