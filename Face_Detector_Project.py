
import cv2

image = cv2.imread("Images/testimage2.jpeg")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.12,
    minNeighbors=5,
    minSize=(40,40)
)

print("Faces detected:", len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Face Detector", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Faces detected:", len(faces))
