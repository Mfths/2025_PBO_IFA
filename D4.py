import cv2
from cvzone.HandTrackingModule import HandDetector

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi Hand Detector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]  # Informasi tangan: lmList, bbox, dll.
        fingers = detector.fingersUp(hand)  # List 5 elemen berupa 0/1 (Open/Close)
        count = sum(fingers)  # Hitung jari yang terangkat

        # Tampilkan jumlah jari
        cv2.putText(
            img, f"Fingers: {count} {fingers}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )

    # Tampilkan frame
    cv2.imshow("Hands + Fingers", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
