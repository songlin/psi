from typing import Union, Annotated
from pydantic import BaseModel, Field, model_validator
# from tyro.conf import subcommand as cmd
from psi.config.config import LaunchConfig
# from psi.config import model as pm
from psi.config.model_qwen3vl import Qwen3VL_ModelConfig
from psi.config.data_he import HERawDataConfig
from psi.config import transform as pt
from psi.config.transform import DataTransform

class DynamicDataTransform(DataTransform):
    repack: pt.HEPretrainRepackTransform
    action_state: pt.ActionMaxMinTransform
    model: pt.Qwen3vlModelTransform
    
    def __call__(self, data, **kwargs):
        data = self.repack(data, **kwargs)
        data = self.action_state(data, **kwargs)
        data = self.model(data, **kwargs)
        return data
    
class DynamicEgoDexDataConfig(HERawDataConfig):
    transform: DynamicDataTransform

class DynamicLaunchConfig(LaunchConfig):
    data: DynamicEgoDexDataConfig
    model: Qwen3VL_ModelConfig
