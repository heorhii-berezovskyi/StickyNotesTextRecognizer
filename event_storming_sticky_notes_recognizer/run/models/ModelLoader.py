import importlib

from torch.nn.modules.module import Module

from event_storming_sticky_notes_recognizer.Exception import ParamNotFoundException


class ModelLoader:
    @staticmethod
    def load(name: str) -> Module:
        try:
            name = '.' + name
            module = importlib.import_module(name=name, package='event_storming_sticky_notes_recognizer.run.models')
            return module.net()
        except:
            raise ParamNotFoundException('Model with name ' + name + ' not found.')
