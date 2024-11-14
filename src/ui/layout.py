from supervisely.app.widgets import Card, Button, SelectCudaDevice, Container

match_bbox_button = Button("MATCH BBOXES", "success", icon="zmdi zmdi-collection-item")
match_bbox_button.disable()
device_selector = SelectCudaDevice(sort_by_free_ram=True, include_cpu_option=True)
device_selector_card = Card("Select Cuda Device", "Device which will be used for processing", True, device_selector)
device_selector_card.collapse()

layout_card = Card(content=Container([match_bbox_button, device_selector_card]))
