import json
import zipfile
from pathlib import Path
from typing import Any, List

from taichi.aot.conventions.gfxruntime140 import dr, sr


class GfxRuntime140:
    def __init__(self, metadata_json: Any, graphs_json: Any) -> None:
        metadata = dr.from_json_metadata(metadata_json)
        graphs = [dr.from_json_graph(x) for x in graphs_json]
        self.metadata = sr.from_dr_metadata(metadata)
        self.graphs = [sr.from_dr_graph(self.metadata, x) for x in graphs]

    @staticmethod
    def from_module(module_path: str) -> "GfxRuntime140":
        if Path(module_path).is_file():
            with zipfile.ZipFile(module_path) as z:
                with z.open("metadata.json") as f:
                    metadata_json = json.load(f)
                with z.open("graphs.json") as f:
                    graphs_json = json.load(f)
        else:
            with open(f"{module_path}/metadata.json") as f:
                metadata_json = json.load(f)
            with open(f"{module_path}/graphs.json") as f:
                graphs_json = json.load(f)

        return GfxRuntime140(metadata_json, graphs_json)

    def to_metadata_json(self) -> Any:
        return dr.to_json_metadata(sr.to_dr_metadata(self.metadata))

    def to_graphs_json(self) -> List[Any]:
        return [dr.to_json_graph(sr.to_dr_graph(x)) for x in self.graphs]
