from __future__ import annotations
from pydantic import BaseModel

class ModelConfig(BaseModel): 
    ...

class DummyModelConfig(ModelConfig):
    # boilerplate model config
    ...
