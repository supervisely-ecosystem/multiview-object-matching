import supervisely as sly
import src.ui.layout as layout
import src.globals as g
from src.apply_lightglue import apply_lightglue_bounding_box

app = sly.Application(layout=layout.layout_card)


@app.event(sly.Event.ManualSelected.FigureChanged)
def manualselected_cb(api: sly.Api, event: sly.Event.ManualSelected.FigureChanged):
    # check if project supports multiview
    if event.tool != 'rectangle' or event.figure_id is None:
        layout.match_bbox_button.disable()
        return
    else:
        layout.match_bbox_button.enable()

    g.CACHE.cache_event(event)
    g.CACHE.project_settings


@layout.match_bbox_button.click
@sly.handle_exceptions
def match_click_cb():
    device = layout.device_selector.get_device()
    if device is None:
        sly.logger.error("No device selected")
        return
    image_paths = g.CACHE.download_images()
    ref_bbox, obj_class = g.CACHE.get_bbox()

    log_info = {
        'device': device, 
        "image paths": image_paths, 
        "reference bbox": ref_bbox.to_json(), 
        "object class": obj_class.name
        }
    sly.logger.debug(f"Appying lightglue", extra=log_info)

    id_to_box = apply_lightglue_bounding_box(image_paths, ref_bbox, device)
    sly.fs.clean_dir(g.SLY_APP_DATA)

    ids = list(id_to_box.keys())
    boxes = list(id_to_box.values())
    dataset_id = g.CACHE.dataset_id
    project_meta = g.CACHE.project_meta
    orig_anns = [sly.Annotation.from_json(ann.annotation, project_meta) for ann in g.api.annotation.download_batch(dataset_id, ids)]

    boxes_labels = [sly.Label(box, obj_class) for box in boxes]
    anns = [ann.add_label(box_label) for ann, box_label in zip(orig_anns, boxes_labels)]
    sly.logger.info(f"Uploading {len(anns)} annotations...")
    g.api.annotation.upload_anns(ids, anns)
