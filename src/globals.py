import os
from dotenv import load_dotenv
import supervisely as sly
from supervisely.api.annotation_api import ApiField as AF
import supervisely.app.development as development
from typing import List, Dict, Literal
from collections import defaultdict

# # * Advanced debug mode
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    development.enable_advanced_debug()

# * Creating an instance of the supervisely API according to the environment variables.
api = sly.Api.from_env()

# * Directories that will be used for checkpoints & temporary image storage
SLY_APP_DATA = sly.app.get_data_dir()
sly.fs.clean_dir(SLY_APP_DATA)
MODEL_DIR = "./checkpoints"

class Cache:
    def __init__(self):
        self.project_metas: Dict[str, sly.ProjectMeta] = {}
        self.project_meta: sly.ProjectMeta = None
        self.project_settings: Dict = {}
        self.group_tag_id: int = None
        self.image_ann: sly.Annotation = None
        self.path_to_id: Dict[str, int] = {}
        self.reference_boxes: Dict[int, List[sly.Label]] = {}
        self.tag_meta = sly.TagMeta(
            "bbox type", sly.TagValueType.ONEOF_STRING, ["reference", "matched"]
        )
        self.figure_id = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'Cache' object has no attribute '{name}'")

    def cache_project_meta(self) -> None:
        project_id = self.project_id
        if project_id not in self.project_metas:
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            self.project_metas[project_id] = project_meta

            project_settings = api.project.get_settings(self.project_id)
            self.project_settings[project_id] = project_settings
            self.group_tag_id = project_settings['groupImagesByTagId']
            self.project_meta = project_meta

    @sly.timeit
    def cache_event(self, event: sly.Event.ManualSelected.FigureChanged, cache_figure=False):
        attrs_to_cache = [
            "project_id",
            "dataset_id",
            "image_id",
        ]
        if cache_figure is True:
            attrs_to_cache.append("figure_id")
        else:
            self.figure_id = None
        for k, v in event.__dict__.items():
            if k in attrs_to_cache:
                self.__dict__[k] = v
        self.cache_project_meta()

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
        group_tag_id = self.group_tag_id

        # get group tag's value
        group_tag_value = None
        for tag in ref_img_info.tags:
            if tag[AF.TAG_ID] == group_tag_id:
                group_tag_value = tag[AF.VALUE]
                break

        # to make sure reference image info is always first
        image_infos = [ref_img_info] + [info for info in self._get_group_imageinfos(group_tag_id, group_tag_value) if info.id != self.image_id]

        path_to_id = {f"{SLY_APP_DATA}/{info.name}": info.id for info in image_infos}
        self.path_to_id = path_to_id

        ids = list(path_to_id.values())
        paths = list(path_to_id.keys())

        api.image.download_paths(self.dataset_id, ids, paths)
        return paths

    @sly.timeit
    def get_reference_bbox_labels(self) -> List[sly.Label]:
        if self.figure_id is not None:
            selected_label = api.annotation.get_label_by_id(self.figure_id, self.project_meta)
            if selected_label is None:
                sly.logger.warn("Selected label is not found")
                return []
            self.image_ann = sly.Annotation.from_json(
                api.annotation.download(self.image_id).annotation, self.project_meta
            )
            return [selected_label]

        ann = sly.Annotation.from_json(
            api.annotation.download(self.image_id).annotation, self.project_meta
        )
        if ann is None:
            sly.logger.warn("Failed to download annotation.")
            return []
        self.image_ann = ann
        return [
            label
            for label in ann.labels
            if (
                isinstance(label.geometry, sly.Rectangle)
                and label.tags.get(self.tag_meta.name) is None
            )
        ]

    @sly.timeit
    def download_anns(self, ids) -> List[sly.Annotation]:
        return [
            sly.Annotation.from_json(ann.annotation, self.project_meta)
            for ann in api.annotation.download_batch(self.dataset_id, ids)
        ]

    def grouping_is_on(self) -> bool:
        project_settings = self.project_settings[self.project_id]
        return (
            project_settings["groupImages"] and project_settings["groupImagesByTagId"] is not None
        )

    @sly.timeit
    def image_has_bboxes(self) -> bool:
        bbox_labels = self.get_reference_bbox_labels()
        if len(bbox_labels) == 0:
            sly.logger.debug(
                "Selected image has no bbox labels, or all boxes are already processed"
            )
            return False
        return True

    def add_tag_to_projmeta(self) -> None:
        project_meta = self.project_meta
        if project_meta.get_tag_meta(self.tag_meta.name) is None:
            project_meta = project_meta.add_tag_meta(self.tag_meta)
            self.project_meta = api.project.update_meta(self.project_id, project_meta)

    def add_tag_to_labels(self, labels: List[sly.Label], value: Literal["reference", "matched"]):
        tag = sly.Tag(self.tag_meta, value)
        res_labels = []
        for label in labels:
            if label.tags.get(self.tag_meta.name) is None:
                label = label.add_tag(tag)
            res_labels.append(label)
        return res_labels

    def log_contents(self) -> None:
        cache = {
            k: v for k, v in self.__dict__.items() if k not in ["project_metas", "project_meta"]
        }
        sly.logger.debug(f"CACHE", extra=cache)


CACHE = Cache()
