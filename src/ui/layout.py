from supervisely.app.widgets import (
    Card,
    Button,
    SelectCudaDevice,
    Container,
    NotificationBox,
    Field,
    Checkbox,
    Text,
    InputNumber,
    Slider,
)

match_bbox_button = Button("MATCH BBOXES", "success", icon="zmdi zmdi-collection-item")
match_bbox_button.disable()
device_selector = SelectCudaDevice(sort_by_free_ram=True, include_cpu_option=True)
device_selector_card = Card("Select Cuda Device", "Device which will be used for processing", True, device_selector)
device_selector_card.collapse()
resize_check = Checkbox(Text("Resize"), True)
max_keypoints_inputnum = InputNumber(1024, 0, 2048, 256)
max_keypoints_field = Field(
    max_keypoints_inputnum, "Max Keypoints", "Maximum keypoints amount for an image"
)
resize_inputnum = InputNumber(256, 128, 1024, 128)
resize_field = Field(
    Container([resize_check, resize_inputnum], "horizontal"),
    "Resize images for processing",
    "Only used for computation, vastly affects CPU computing times",
)
filter_threshold = Slider(0.3, 0, 0.95, 0.05, False, True)
filter_field = Field(
    filter_threshold, "Filter threshold", "Keypoint filter threshold for inlier keypoints"
)
lightglue_params = Card(
    "LightGlue parameters",
    "Configure LightGlue processing",
    True,
    Container([max_keypoints_field, resize_field, filter_field]),
)
lightglue_params.collapse()
grouping_warning = NotificationBox(
    "Enable Multiview",
    "This application is designed to work with multiview images, "
    "please enable images grouping in the project settings",
    "error",
)
grouping_warning.hide()

layout_card = Card(
    content=Container([grouping_warning, match_bbox_button, device_selector_card, lightglue_params])
)


@resize_check.value_changed
def check_cb(value):
    if value is True:
        resize_inputnum.show()
    else:
        resize_inputnum.hide()
