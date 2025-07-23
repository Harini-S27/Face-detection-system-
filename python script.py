import cv2
import face_recognition
from openvino.runtime import Core

# Load known face encodings
known_face_encodings = []
known_face_names = []

# Directory containing known face images
known_faces_dir = 'path/to/known_faces'  # Update with your path

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name

# Initialize OpenVINO Core
ie_core = Core()
model_path = r"C:\Users\27har\Downloads\OpenVino_face-detection_python-master\OpenVino_face-detection_python-master\models\face-detection-adas-0001.xml"
exec_net = ie_core.compile_model(ie_core.read_model(model=model_path), device_name="CPU")

# Open video capture
cap = cv2.VideoCapture(r"C:\Users\27har\Downloads\WhatsApp Video 2024-10-06 at 18.33.12.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS
print(f"Video FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (672, 384))
    input_data = resized_frame.reshape(1, 3, 384, 672).astype('float32')
    
    # Perform inference
    res = exec_net.infer(inputs={next(iter(exec_net.inputs)): input_data})
    
    # Process detection results
    output_blob = next(iter(res.keys()))
    boxes = res[output_blob][0][0]

    # Check if faces are detected
    for box in boxes:
        confidence = box[2]
        if confidence > 0.5:
            # Get coordinates
            xmin = int(box[3] * frame.shape[1])
            ymin = int(box[4] * frame.shape[0])
            xmax = int(box[5] * frame.shape[1])
            ymax = int(box[6] * frame.shape[0])

            # Extract face ROI
            face_roi = frame[ymin:ymax, xmin:xmax]

            # Convert BGR to RGB for face_recognition
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Encode the face
            face_encoding = face_recognition.face_encodings(rgb_face)

            if face_encoding:
                face_encoding = face_encoding[0]
                
                # Compare to known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"
                # Check if a match was found
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]  # Get the name of the matched person

                # Draw rectangle and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate timestamp
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                timestamp = current_frame / fps
                print(f"Face detected: {name} at {timestamp:.2f} seconds (Frame: {current_frame})")

    # Display the frame
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
