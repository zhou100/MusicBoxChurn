from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42


class DataConfig(BaseModel):
    path: Path = Path("Processed_data/df_model_final.csv")
    split: SplitConfig = Field(default_factory=SplitConfig)


class TrainConfig(BaseModel):
    model_type: str = "lr"
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = 42


class InferenceConfig(BaseModel):
    run_dir: Path
    input_path: Path
    output_path: Path
