import cv2
from mjpeg_streamer import MjpegServer, Stream

cap = cv2.VideoCapture(0)

stream = Stream("my_camera", size=(640, 480), quality = 50, fps=30)

# 127.0.0.1, 8080 for local, 192.168.5.174 for device
server = MjpegServer("192.168.225.69", 8080)
server.add_stream(stream)
server.start()

while True:
	_, frame = cap.read()
	#cv2.imshow(stream.name, frame)
	if cv2.waitKey(1) == ord("q"):
		break
		
	stream.set_frame(frame)
	
server.stop()
cap.release()
cv2.destroyAllWindows()
