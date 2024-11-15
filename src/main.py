import supervisely as sly
import src.ui.layout as layout
import src.globals as g
from src.globals import CACHE
import src.process_funcs as process

app = sly.Application(layout=layout.layout_card)

# @app.event(sly.Event.ManualSelected.FigureChanged)
# def figure_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.FigureChanged):
#     if event.tool != 'rectangle' or event.figure_id is None:
#         layout.match_bbox_button.disable()
#         return
#     else:
#         layout.match_bbox_button.enable()
#     g.CACHE.cache_event(event)

# @TODO: Add single bbox mode aswell v ^
@app.event(sly.Event.ManualSelected.FigureChanged)
def figure_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.FigureChanged):
    pass


@app.event(sly.Event.ManualSelected.ImageChanged)
def image_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.ImageChanged):
    CACHE.cache_event(event)
    if CACHE.image_has_bboxes() and CACHE.grouping_is_on():
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
        sly.logger.error("No device selected")
        return
    try:
        image_paths = CACHE.download_group_images()
        ref_bbox_labels = CACHE.get_reference_bbox_labels()
        if len(ref_bbox_labels) == 0:
            sly.logger.warning("All bboxes are already processed.")

        sly.logger.info(
            f"Appying lightglue for {len(image_paths)} images on '{device.upper()}' device",
            extra={"reference bboxes count": len(ref_bbox_labels)},
        )
        id_to_labels = process.apply_lightglue_bounding_boxes(
            image_paths, ref_bbox_labels, 1024, device
        )

        ids = list(id_to_labels.keys())
        boxes_labels = [
            CACHE.add_tag_to_labels((box_list), "matched") for box_list in id_to_labels.values()
        ]
        orig_anns = CACHE.download_anns(ids)
        anns = [ann.add_labels(box_labels) for ann, box_labels in zip(orig_anns, boxes_labels)]

        ids_to_upload = [CACHE.image_id] + ids
        ref_ann = CACHE.image_ann
        new_ref_ann_labels = []
        for label in ref_ann.labels:
            if label in ref_bbox_labels:
                label = CACHE.add_tag_to_labels([label], "reference")[0]
            new_ref_ann_labels.append(label)
        anns_to_upload = [ref_ann.clone(labels=new_ref_ann_labels)] + anns
        sly.logger.info(f"Uploading {len(anns_to_upload)} annotations...")
        g.api.annotation.upload_anns(ids_to_upload, anns_to_upload)
    except Exception as e:
        sly.logger.error(f"An error occured while processing bboxes: {e}")
    finally:
        sly.logger.debug(f"Cleaning {g.SLY_APP_DATA} directory from paths: {image_paths}")
        sly.fs.clean_dir(g.SLY_APP_DATA)


# @TODO: don't process images with tag, docker image, modal window, check lightglue confidence,
# add modal window option for resizing

# to think through: meta is getting cached, and if updated, will be overwritten when processing code runs, leading to errors.
