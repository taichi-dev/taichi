from typing import Any, List

from taichi.aot.conventions.gfxruntime140 import dr, sr


class GfxRuntime140:
    def __init__(self, metadata_json: Any, graphs_json: Any) -> None:
        metadata = dr.from_json_metadata(metadata_json)
        graphs = [dr.from_json_graph(x) for x in graphs_json]
        self.metadata = sr.from_dr_metadata(metadata)
        self.graphs = [sr.from_dr_graph(self.metadata, x) for x in graphs]

    def to_metadata_json(self) -> Any:
        return dr.to_json_metadata(sr.to_dr_metadata(self.metadata))

    def to_graphs_json(self) -> List[Any]:
        return [dr.to_json_graph(sr.to_dr_graph(x)) for x in self.graphs]
