import cv2

source = "http://172.28.171.45:8090/"
cap = cv2.VideoCapture(source)
i = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    i += 1
    cv2.putText(frame, "Blinks: {}".format(i), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


