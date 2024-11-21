import os
from dotenv import load_dotenv
import supervisely as sly
from supervisely.api.annotation_api import ApiField as AF
import supervisely.app.development as development
from typing import List, Dict, Literal

# # * Advanced debug mode
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    development.enable_advanced_debug()

# * Creating an instance of the supervisely API according to the environment variables.
api = sly.Api.from_env()
box_id = 0

# * Directories that will be used for checkpoints & temporary image storage
SLY_APP_DATA = sly.app.get_data_dir()
sly.fs.clean_dir(SLY_APP_DATA)
MODEL_DIR = "./checkpoints"

class Cache:

    def __init__(self):
        # * Attributes that will be cached from events later
        self.project_id: int = None
        self.dataset_id: int = None
        self.image_id: int = None
        self.figure_id: int = None

        # * Other attributes for caching
        self.project_metas: Dict[str, sly.ProjectMeta] = {}
        self.project_meta: sly.ProjectMeta = None
        self.project_settings: Dict = {}
        self.ann_needs_update: bool = False

        # * Attributes needed for processing
        self.group_tag_id: int = None
        self.image_ann: sly.Annotation = None
        self.path_to_id: Dict[str, int] = {}
        self.type_tag_meta = sly.TagMeta(
            "bbox match type", sly.TagValueType.ONEOF_STRING, ["reference", "matched"]
        )
        self.id_tag_meta = sly.TagMeta("matched bbox ID", sly.TagValueType.ANY_NUMBER)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'Cache' object has no attribute '{name}'")

    def cache_project_meta(self, project_id: int) -> None:
        if project_id not in self.project_metas:
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            self.project_metas[project_id] = project_meta

            project_settings = api.project.get_settings(project_id)
            self.project_settings[project_id] = project_settings
            self.group_tag_id = project_settings['groupImagesByTagId']
            self.project_meta = project_meta

    def cache_image_ann(self, image_id: int) -> None:
        ann = sly.Annotation.from_json(
            api.annotation.download(image_id).annotation, self.project_meta
        )
        if ann is None:
            sly.logger.error("Failed to download annotation.")
            return
        self.image_ann = ann

    @sly.timeit
    def cache_event(self, event: sly.Event.ManualSelected.FigureChanged) -> None:
        self.cache_project_meta(event.project_id)
        if self.image_id != event.image_id or self.ann_needs_update:
            self.cache_image_ann(event.image_id)

        attrs_to_cache = ["project_id", "dataset_id", "image_id", "figure_id"]
        for k, v in event.__dict__.items():
            if k in attrs_to_cache:
                self.__dict__[k] = v
        self.log_contents()

    def log_contents(self) -> None:
        attrs_to_log = [
            "project_id",
            "dataset_id",
            "image_id",
            "figure_id",
            "group_tag_id",
        ]
        cache = {k: v for k, v in self.__dict__.items() if k in attrs_to_log}
        sly.logger.debug(f"CACHE", extra=cache)

    def _get_group_imageinfos(self, tag_id, tag_value):
        filters = [
            {
                AF.TYPE: "images_tag",
                AF.DATA: {
                    AF.TAG_ID: tag_id,
                    AF.INCLUDE: True,
                    AF.VALUE: tag_value,
                },
            }
        ]
        return api.image.get_filtered_list(self.dataset_id, filters)

    @sly.timeit
    def download_group_images(self) -> List[str]:
        ref_img_info = api.image.get_info_by_id(self.image_id)
        if self.group_tag_id is None:
            self.cache_project_meta()

        group_tag_value = None
        for tag in ref_img_info.tags:
            if tag[AF.TAG_ID] == self.group_tag_id:
                group_tag_value = tag[AF.VALUE]
                break

        # to make sure reference image info is always first
        image_infos = [ref_img_info] + [
            info
            for info in self._get_group_imageinfos(self.group_tag_id, group_tag_value)
            if info.id != self.image_id
        ]

        path_to_id = {f"{SLY_APP_DATA}/{info.name}": info.id for info in image_infos}
        self.path_to_id = path_to_id

        ids = list(path_to_id.values())
        paths = list(path_to_id.keys())

        api.image.download_paths(self.dataset_id, ids, paths)
        return paths

    def get_reference_bbox_labels(self) -> List[sly.Label]:
        if self.figure_id is not None:
            label = self.image_ann.get_label_by_id(self.figure_id)
            if label is None:
                sly.logger.warning(
                    f"Figure (id: {self.figure_id}) not found in the image annotation"
                )
                return []
            if label.tags.get(self.type_tag_meta.name) is not None:
                return []
            return [label]
        return [
            label
            for label in self.image_ann.labels
            if (isinstance(label.geometry, sly.Rectangle))
            and label.tags.get(self.type_tag_meta.name) is None
        ]

    @sly.timeit
    def download_anns(self, ids: List[int]) -> List[sly.Annotation]:
        return [
            sly.Annotation.from_json(ann.annotation, self.project_meta)
            for ann in api.annotation.download_batch(self.dataset_id, ids)
        ]

    def grouping_is_on(self) -> bool:
        project_settings = self.project_settings[self.project_id]
        return (
            project_settings["groupImages"] and project_settings["groupImagesByTagId"] is not None
        )

    def image_has_unprocessed_bboxes(self) -> bool:
        if len(self.get_reference_bbox_labels()) == 0:
            sly.logger.debug(
                "Selected image has no bbox labels, or all boxes on it are already processed"
            )
            return False
        return True

    def add_tags_to_projmeta(self) -> None:
        project_meta = self.project_meta
        tag_metas = [self.type_tag_meta, self.id_tag_meta]
        meta_needs_update = False
        for tag_meta in tag_metas:
            if project_meta.get_tag_meta(tag_meta.name) is None:
                project_meta = project_meta.add_tag_meta(tag_meta)
                meta_needs_update = True

        if meta_needs_update:
            self.project_meta = api.project.update_meta(self.project_id, project_meta)

    def add_type_tag_to_labels(
        self, labels: List[sly.Label], value: Literal["reference", "matched"]
    ) -> List[sly.Label]:
        type_tag = sly.Tag(self.type_tag_meta, value)
        res_labels = []
        for label in labels:
            if label.tags.get(self.type_tag_meta.name) is None:
                label = label.add_tag(type_tag)
            res_labels.append(label)
        return res_labels


CACHE = Cache()
