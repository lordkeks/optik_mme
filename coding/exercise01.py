import cv2
import sys
from harvesters.core import Harvester

# create harvester object
h = Harvester()
h.add_file('C:/Users/Chris/Desktop/bildgebende_optische_systeme/optik_mme/coding/JaiUSB3vTL.cti', check_validity=True)
h.update()
if len(h.device_info_list) < 1:
    print("no camera found!")
    sys.exit()
print(h.device_info_list[0])

# create image acquirer
ia = h.create_image_acquirer(0)
width = 2560
height = 1960
ia.device.node_map.Width.value, ia.device.node_map.Height.value = width, height
ia.device.node_map.PixelFormat.value = 'Mono8'

# start image aq
ia.start_image_acquisition()
with ia.fetch_buffer() as buffer:
    component = buffer.payload.components[0]
    frame = component.data.reshape(component.height, component.width)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('snap.png', frame)
    cv2.imshow('frame', frame)

cv2.destroyAllWindows()
ia.stop_image_acquisition()
ia.destroy()
h.reset()
