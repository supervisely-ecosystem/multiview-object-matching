import supervisely as sly
import src.ui.layout as layout
import src.globals as g
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


@app.event(sly.Event.ManualSelected.FigureChanged)
def figure_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.FigureChanged):
    pass


@app.event(sly.Event.ManualSelected.ImageChanged)
def image_changed_cb(api: sly.Api, event: sly.Event.ManualSelected.ImageChanged):
    g.CACHE.cache_event(event)
    if g.CACHE.image_has_bboxes() is True:
        layout.match_bbox_button.enable()
    else:
        layout.match_bbox_button.disable()


@layout.match_bbox_button.click
@sly.timeit
def match_click_cb():
    g.CACHE.log_contents()
    device = layout.device_selector.get_device()
    if device is None:
        sly.logger.error("No device selected")
        return
    if not g.CACHE.grouping_is_on():
        sly.logger.warning("Grouping is disabled, or no grouping tag found.")
        return
    image_paths = g.CACHE.download_images()
    ref_bbox_labels = g.CACHE.reference_boxes

    sly.logger.info(
        f"Appying lightglue for {len(image_paths)} images on '{device.upper()}' device",
        extra={"reference bboxes count": len(ref_bbox_labels)},
    )
    try:
        id_to_labels = process.apply_lightglue_bounding_boxes(
            image_paths, ref_bbox_labels, 1024, device
        )

        ids = list(id_to_labels.keys())
        boxes_labels = list(id_to_labels.values())

        orig_anns = g.CACHE.download_anns(ids)
        anns = [ann.add_labels(box_labels) for ann, box_labels in zip(orig_anns, boxes_labels)]

        sly.logger.info(f"Uploading {len(anns)} annotations...")
        g.api.annotation.upload_anns(ids, anns)
    except Exception as e:
        sly.logger.error(f"An error occured while processing bboxes: {e}")
    finally:
        sly.logger.debug(f"Cleaning {g.SLY_APP_DATA} directory from paths: {image_paths}")
        sly.fs.clean_dir(g.SLY_APP_DATA)


# @TODO: uuid tag, docker image, batches, modal window, check params like resize, homography thresholds, lightglue confidence,
# think about resizing and cuda: add modal window option
