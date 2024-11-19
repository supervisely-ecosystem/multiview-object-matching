from supervisely.app.widgets import Card, Button, SelectCudaDevice, Container, NotificationBox

match_bbox_button = Button("MATCH BBOXES", "success", icon="zmdi zmdi-collection-item")
match_bbox_button.disable()
device_selector = SelectCudaDevice(sort_by_free_ram=True, include_cpu_option=True)
device_selector_card = Card("Select Cuda Device", "Device which will be used for processing", True, device_selector)
device_selector_card.collapse()
grouping_warning = NotificationBox(
    "Enable Multiview",
    "This application is designed to work with multiview images, "
    "please enable images grouping in the project settings",
    "error",
)
grouping_warning.hide()
layout_card = Card(content=Container([grouping_warning, match_bbox_button, device_selector_card]))
