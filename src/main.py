import supervisely as sly
import src.ui.layout as layout
import src.globals as g
from src.globals import CACHE
import src.process_funcs as process

app = sly.Application(layout=layout.layout_card)


@app.event(sly.Event.FigureCreated)
def figure_created_cb(api: sly.Api, event: sly.Event.FigureCreated):
    if event.tool == "rectangle":
        CACHE.ann_needs_update = True

    CACHE.cache_event(event)
    if CACHE.project_meta.get_obj_class_by_id(event.figure_class_id) is None:
        CACHE.cache_project_meta(event.project_id)

    if CACHE.image_has_unprocessed_bboxes() and CACHE.grouping_is_on():
        layout.match_bbox_button.enable()
    else:
        if not CACHE.grouping_is_on():
            layout.grouping_warning.show()
        else:
            layout.grouping_warning.hide()
        layout.match_bbox_button.disable()


@app.event(sly.Event.ManualSelected.FigureChanged)
def figure_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.FigureChanged):
    CACHE.cache_event(event)
    if (
        CACHE.image_has_unprocessed_bboxes()
        and CACHE.grouping_is_on()
        and event.tool == "rectangle"
    ):
        layout.match_bbox_button.enable()
    else:
        if not CACHE.grouping_is_on():
            layout.grouping_warning.show()
        else:
            layout.grouping_warning.hide()
        layout.match_bbox_button.disable()


@app.event(sly.Event.ManualSelected.ImageChanged)
def image_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.ImageChanged):
    CACHE.cache_event(event)

    if CACHE.image_has_unprocessed_bboxes() and CACHE.grouping_is_on():
        layout.match_bbox_button.enable()
    else:
        if not CACHE.grouping_is_on():
            layout.grouping_warning.show()
        else:
            layout.grouping_warning.hide()
        layout.match_bbox_button.disable()


@layout.match_bbox_button.click
@sly.timeit
def match_click_cb():
    if not CACHE.grouping_is_on():
        sly.logger.error("Multiview mode must be enabled in order for application to work.")
        return

    image_id = CACHE.image_id
    id_tag_meta = CACHE.id_tag_meta

    # * Add tag metas to project meta to later add tags to matched boxes
    CACHE.add_tags_to_projmeta()
    # * Get latest reference image annotation, as it could have been updated since it was last retrieved
    CACHE.cache_image_ann(image_id)
    image_ann = CACHE.image_ann

    ref_ann_labels = image_ann.labels
    ref_boxes_labels = CACHE.get_reference_bbox_labels()

    device = layout.device_selector.get_device()
    if device is None:
        sly.logger.error("No processing device found, trying to run on CPU...")
        device = "cpu"

    # to handle the case when there were boxes, but got deleted before processing
    if len(ref_boxes_labels) == 0:
        sly.logger.error(
            "Selected image has no bbox labels, or all bboxes on it are already processed."
        )
        return
    try:
        # * Download images from grouping
        image_paths = CACHE.download_group_images()
        sly.logger.info(
            f"Appying lightglue for {len(image_paths)} images on '{device.upper()}' device",
            extra={"reference bboxes count": len(ref_boxes_labels)},
        )

        # * Apply lightglue to group images, log a warning if some images fail matching
        process_image_cnt = len(image_paths)
        points_list = [pts for pts in process.apply_lightglue(image_paths, 1024, 512, device)]

        failed_imgs_cnt = (process_image_cnt - 1) - len(points_list)
        if failed_imgs_cnt > 0:
            sly.logger.warning(
                f"Matching failed for {failed_imgs_cnt} image(s). They will be skipped."
            )
    except Exception as e:
        sly.logger.error(f"An error occured while processing bboxes: {e}")
        sly.fs.clean_dir(g.SLY_APP_DATA)
        return
    finally:
        # * Clear directory from downloaded image paths
        sly.logger.debug(f"Cleaning {g.SLY_APP_DATA} directory from paths")
        sly.fs.clean_dir(g.SLY_APP_DATA)

    # * Transform reference boxes for group images using matching keypoints
    # new_bbox_labels = [
    #     process.apply_transform_to_bboxes(ref_boxes_labels, ref_pts, img_pts)
    #     for ref_pts, img_pts in points_list
    # ]

    new_bbox_labels = [
        process.transpose_bbox_with_keypoints(ref_boxes_labels, ref_pts, img_pts)
        for ref_pts, img_pts in points_list
    ]

    # * Download original annotations and add new labels to them
    group_ids = [CACHE.path_to_id[path] for path in image_paths]
    orig_anns = CACHE.download_anns(group_ids)
    res_bbox_labels = [
        CACHE.add_type_tag_to_labels((box_list), "matched") for box_list in new_bbox_labels
    ]

    # * Add type tag to reference boxes
    new_ref_ann_labels = []
    for label in ref_ann_labels:
        if label in ref_boxes_labels:
            label = CACHE.add_type_tag_to_labels([label], "reference")[0]
        new_ref_ann_labels.append(label)

    # * Add ID tag to boxes
    for idx, ref_box in enumerate(new_ref_ann_labels):
        tag = sly.Tag(id_tag_meta, g.box_id)
        new_ref_ann_labels[idx] = ref_box.add_tag(tag)
        for image_boxes in res_bbox_labels:
            if idx < len(image_boxes):
                image_boxes[idx] = image_boxes[idx].add_tag(tag)
        g.box_id += 1

    # * Merge new and original annotations
    anns = [ann.add_labels(box_labels) for ann, box_labels in zip(orig_anns, res_bbox_labels)]
    new_ref_ann = image_ann.clone(labels=new_ref_ann_labels)

    # * Upload annotations
    ids_to_upload = [image_id] + group_ids
    anns_to_upload = [new_ref_ann] + anns
    sly.logger.info(f"Uploading {len(anns_to_upload)} annotations...")
    g.api.annotation.upload_anns(ids_to_upload, anns_to_upload)


# @TODO: modal win, docker, poster & icon
# to think through: meta is getting cached, and if updated by user, will be overwritten with older version of it when processing code runs, leading to errors.
# to fix: re-cache meta when settings updated
# to check: bug when an exception rises while processing, and labeling toolbox defaults
