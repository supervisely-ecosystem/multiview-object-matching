import os
from dotenv import load_dotenv
import supervisely as sly
from supervisely.api.annotation_api import ApiField as AF
import supervisely.app.development as development
from typing import List

# * Advanced debug mode
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

    development.supervisely_vpn_network(action="up")
    development.create_debug_task(sly.env.team_id(), port="8000")

# * Creating an instance of the supervisely API according to the environment variables.
api = sly.Api.from_env()

SLY_APP_DATA = sly.app.get_data_dir()
sly.fs.clean_dir(SLY_APP_DATA)
MODEL_DIR = "./checkpoints"

class Cache:
    def __init__(self):
        self.project_metas = {}
        self.project_meta = None
        self.project_settings = {}
        self.group_tag_id = None
        self.path_to_id = {}

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f"'Cache' object has no attribute '{name}'")

    def cache_project_meta(self):
        project_id = self.project_id
        if project_id not in self.project_metas:
            project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
            self.project_metas[project_id] = project_meta

            project_settings = api.project.get_settings(self.project_id)
            self.project_settings[project_id] = project_settings
            if project_settings["groupImages"] is False:
                raise RuntimeError("App only supports multiview projects.")

            self.group_tag_id = project_settings['groupImagesByTagId']
            self.project_meta = project_meta

    def cache_event(self, event: sly.Event.ManualSelected.FigureChanged):
        attrs_to_cache = [
            "project_id",
            "dataset_id",
            "image_id",
            "figure_id",
        ]
        for k, v in event.__dict__.items():
            if k in attrs_to_cache:
                self.__dict__[k] = v
        self.cache_project_meta()
        self.log_contents()

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

    def download_images(self) -> List[str]:
        ref_img_info = api.image.get_info_by_id(self.image_id)
        group_tag_id = self.group_tag_id
        if group_tag_id is None:
            self.cache_project_meta()
            group_tag_id = self.group_tag_id

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

    def get_bbox(self):
        bbox = api.annotation.get_label_by_id(self.figure_id, self.project_meta)
        return bbox.geometry, bbox.obj_class

    def log_contents(self):
        cache = {
            k: v for k, v in self.__dict__.items() if k not in ["project_metas", "project_meta"]
        }
        sly.logger.debug(f"CACHE", extra=cache)

CACHE = Cache()
