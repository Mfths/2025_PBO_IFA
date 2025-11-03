import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Landmark mata kiri (EAR)
L_TOP, L_BOTTOM = 159, 145   # Titik vertikal mata
L_LEFT, L_RIGHT = 33, 133    # Titik horizontal mata

# Fungsi hitung jarak euclidean
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi deteksi FaceMesh
detector = FaceMeshDetector(
    staticMode=False, 
    maxFaces=1, 
    minDetectionCon=0.5, 
    minTrackCon=0.5
)

# Variabel logika kedipan
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3
EYE_AR_THRESHOLD = 0.20
is_closed = False

ear = 1  # default value biar tidak error sebelum deteksi

while True:
    success, img = cap.read()
    if not success:
        break

    # Deteksi titik FaceMesh
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]  # Data 468 titik wajah

        # Hitung EAR untuk mata kiri
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)

        # Tampilkan nilai EAR
        cv2.putText(img, f"EAR(L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Logika kedipan
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

    # Tampilkan jumlah kedipan di layar
    cv2.putText(img, f"Blink: {blink_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame kamera
    cv2.imshow("FaceMesh + EAR", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera & jendela
cap.release()
cv2.destroyAllWindows()
