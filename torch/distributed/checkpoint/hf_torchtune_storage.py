# mypy: allow-untyped-defs
import dataclasses
import json
import queue
import threading
from typing import Dict, List, Optional

import torch
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint.filesystem import _item_size, _SerialCpuLoader
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    StorageMeta,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

try:
    from safetensors.torch import load, save
except ImportError:
    pass


__all__ = ["HuggingFaceHubTorchTuneWriter", "HuggingFaceHubTorchTuneReader"]

_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_shards}"
NUM_ITEMS_PER_FILE = 15
SUFFIX = ".safetensors"


class HuggingFaceHubTorchTuneWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with the huggingface hub.

    The checkpoint of a model.safetensors.index.json file with the metadata.

    Args:
        path: hf directory where the checkpoint will be written to. Should begin with hf://.
        token: The token to use to authenticate with huggingface hub.

    """

    def __init__(
        self,
        path: str,
        token: Optional[str] = None,
    ) -> None:
        """
        Initialize the huggingface writer

        Args:
        """
        super().__init__(path=path, token=token)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        storage_data : Dict[str, int] = {}
        fqn_count = 0
        index = 1
        for write_item in plan.items:
            if fqn_count % NUM_ITEMS_PER_FILE == 0:
                index += 1
            storage_data[write_item.index.fqn] = index
            fqn_count += 1

        return dataclasses.replace(plan, storage_data=storage_data)

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        assert len(plans) == 1, "distributed checkpointing is not yet supported"
        return plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
        storage_plan : Dict[str, int] = plan.storage_data

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(buckets.keys())

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = self._gen_file_name(file_index, highest_index)
            file_queue.put((file_name, write_items))

        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=self._write_files_from_queue,
                args=(
                    file_queue,
                    result_queue,
                    planner,
                ),
            )
            t.start()
            threads.append(t)

        self._write_files_from_queue(
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            fut: Future[List[WriteResult]] = Future()
            fut.set_result(res)
            return fut

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        metadata_to_write = {}
        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index.fqn: wr.storage_data for wr in wr_list})
        metadata_to_write["weight_map"] = storage_md

        metadata_path = self.fs.concat_path(self.path, f"{_metadata_fn}")
        with self.fs.create_stream(metadata_path, "w") as metadata_file:
            json.dump(metadata_to_write, metadata_file)

    def _write_files_from_queue(
        self,
        file_queue: queue.Queue,
        result_queue: queue.Queue,
        planner: SavePlanner,
    ) -> None:
        try:
            while True:
                file_name, write_items = file_queue.get_nowait()

                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

                tensor_w = [wi for wi in write_items if wi.type == WriteItemType.TENSOR]
                for write_item in tensor_w:
                    loader.add(_item_size(write_item), write_item)
                loader.start_loading()

                bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
                write_results = []
                if len(bytes_w) > 0:
                    raise NotImplementedError("Byte IO not supported yet")

                full_path = self.fs.concat_path(self.path, file_name)
                with self.fs.create_stream(full_path, "wb") as stream:
                    tensor_dict = {}
                    for tensor, write_item in loader.values():
                        assert tensor.is_cpu
                        write_results.append(
                            self._write_item(tensor, write_item, file_name)
                        )
                        tensor_dict[write_item.index.fqn] = tensor

                    stream.write(save(tensor_dict))

                result_queue.put(write_results)
        except queue.Empty:
            pass

    def _write_item(
        self,
        data: torch.Tensor,
        write_item: WriteItem,
        storage_key: str,
    ) -> WriteResult:
        assert data.device == torch.device("cpu")

        length = data.numel() * data.element_size()

        return WriteResult(
            index=write_item.index,
            size_in_bytes=length,
            storage_data=storage_key,
        )

    def _split_by_storage_plan(self, storage_plan : Dict[str, int], items: List[WriteItem]) -> Dict[int, List[WriteItem]]:
        # storage_plan is a map from key to index
        buckets = {}
        for item in items:
            key = item.index.fqn
            idx = storage_plan[key]
            if idx not in buckets:
                buckets[idx] = [item]
            else:
                buckets[idx].append(item)

        return buckets

    def _gen_file_name(self, index: int, largest_index: int) -> str:
        return FILE_NAME.format(
                    cpt_idx=f"{index}".zfill(5), num_shards=f"{largest_index}".zfill(5)
                )
    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class HuggingFaceHubTorchTuneReader(FsspecReader):

    def __init__(self, path: str, token : Optional[str] = None) -> None:
        """
        A reader that reads from a huggingface repository in the huggingface format.
        Uses in Fsspec back-end to communicate with the huggingface hub.

        Args:
            path: hf directory where the checkpoint will be read from. Should begin with hf://.
            token: The token to use to authenticate with huggingface hub.

        """
        super().__init__(path=path, token=token)


    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: Dict[str, List[ReadItem]] = {}

        for read_item in plan.items:
            file_name =  self.storage_data[read_item.storage_index.fqn]
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, file_name)
            with self.fs.create_stream(new_path, "rb") as stream:
                loaded_tensors = load(stream.read())
                for req in reqs:
                    tensor = loaded_tensors[req.dest_index.fqn]

                    target_tensor = planner.resolve_tensor(req).detach()
                    target_tensor.resize_(tensor.size())
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)


        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        path = self.fs.concat_path(self.path, _metadata_fn)
        with self.fs.create_stream(path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        state_dict_metadata = {}
        for key in metadata["weight_map"].keys():
            state_dict_metadata[key] = BytesStorageMetadata()
        metadata = Metadata(state_dict_metadata = state_dict_metadata, storage_data=metadata["weight_map"])

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
