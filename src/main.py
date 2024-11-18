import supervisely as sly
import src.ui.layout as layout
import src.globals as g
from src.globals import CACHE
import src.process_funcs as process

app = sly.Application(layout=layout.layout_card)


@app.event(sly.Event.FigureCreated)
def figure_created_cb(api: sly.Api, event: sly.Event.FigureCreated):
    if event.tool == "rectangle":
        g.CACHE.ann_needs_update = True

    g.CACHE.cache_event(event)
    if event.tool == "rectangle" and g.CACHE.image_has_unprocessed_bboxes():
        layout.match_bbox_button.enable()
    else:
        layout.match_bbox_button.disable()


@app.event(sly.Event.ManualSelected.FigureChanged)
def figure_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.FigureChanged):
    g.CACHE.cache_event(event)
    if event.tool == "rectangle" and g.CACHE.image_has_unprocessed_bboxes():
        layout.match_bbox_button.enable()
    else:
        layout.match_bbox_button.disable()


@app.event(sly.Event.ManualSelected.ImageChanged)
def image_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.ImageChanged):
    CACHE.cache_event(event)
    if CACHE.image_has_unprocessed_bboxes() and CACHE.grouping_is_on():
        layout.match_bbox_button.enable()
    else:
        layout.match_bbox_button.disable()


@layout.match_bbox_button.click
@sly.timeit
def match_click_cb():
    # CACHE.log_contents()
    CACHE.add_tag_to_projmeta()
    device = layout.device_selector.get_device()
    if device is None:
        sly.logger.error("No processing device found, trying to run on CPU...")
        device = "cpu"
    try:
        # * Get latest image annotation, as it could have been updated since it was retrieved
        CACHE.cache_image_ann(CACHE.image_id)
        ref_bbox_labels = CACHE.get_reference_bbox_labels()
        if len(ref_bbox_labels) == 0:
            sly.logger.error(
                "Selected image has no bbox labels, or all boxes on it are already processed."
            )
            return

        # * Download images from grouping and apply lightglue to them
        image_paths = CACHE.download_group_images()
        sly.logger.info(
            f"Appying lightglue for {len(image_paths)} images on '{device.upper()}' device",
            extra={"reference bboxes count": len(ref_bbox_labels)},
        )
        id_to_labels = process.apply_lightglue_bounding_boxes(
            image_paths, ref_bbox_labels, 1024, device
        )

        # * Download original annotations and add new labels to them, adding tag to new anns
        ids = list(id_to_labels.keys())
        orig_anns = CACHE.download_anns(ids)
        boxes_labels = [
            CACHE.add_tag_to_labels((box_list), "matched") for box_list in id_to_labels.values()
        ]
        anns = [ann.add_labels(box_labels) for ann, box_labels in zip(orig_anns, boxes_labels)]

        # * Add tag to reference boxes aswell
        new_ref_ann_labels = []
        for label in CACHE.image_ann.labels:
            if label in ref_bbox_labels:
                label = CACHE.add_tag_to_labels([label], "reference")[0]
            new_ref_ann_labels.append(label)

        # * Make sure reference image's annotation gets updated
        ids_to_upload = [CACHE.image_id] + ids
        anns_to_upload = [CACHE.image_ann.clone(labels=new_ref_ann_labels)] + anns

        # * Upload all the annotations
        sly.logger.info(f"Uploading {len(anns_to_upload)} annotations...")
        g.api.annotation.upload_anns(ids_to_upload, anns_to_upload)
    except Exception as e:
        sly.logger.error(f"An error occured while processing bboxes: {e}")
    finally:
        # * Clear directory from downloaded image paths
        sly.logger.debug(f"Cleaning {g.SLY_APP_DATA} directory from paths: {image_paths}")
        sly.fs.clean_dir(g.SLY_APP_DATA)


# @TODO: docker image, modal window (resize option), check lightglue confidence,
# to think through: meta is getting cached, and if updated by user, will be overwritten with older version of it when processing code runs, leading to errors.
