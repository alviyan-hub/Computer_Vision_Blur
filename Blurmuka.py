import os
import cv2
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x1 + w), min(H, y1 + h)

            img[y1:y2, x1:x2] = cv2.blur(img[y1:y2, x1:x2], (55, 55))

    return img

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='webcam')  # Ubah default ke 'webcam'
parser.add_argument("--filePath", default=None)
args = parser.parse_args()

output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode == "image":
        img = cv2.imread(args.filePath)
        if img is None:
            print("Gambar tidak ditemukan. Periksa path gambar.")
            exit()
        result_img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, 'output.png'), result_img)
        print("Gambar disimpan di folder output sebagai 'output.png'")

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        if not ret:
            print("Video tidak bisa dibuka. Periksa path video.")
            exit()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), fourcc, 25, (frame.shape[1], frame.shape[0]))

        while ret:
            result_frame = process_img(frame, face_detection)
            out.write(result_frame)
            ret, frame = cap.read()

        cap.release()
        out.release()
        print("Video disimpan di folder output sebagai 'output.mp4'")

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)  # Gunakan 0 untuk webcam default
        if not cap.isOpened():
            print("Webcam tidak bisa dibuka.")
            exit()

        while True:
            ret, frame = cap.read()  # Baca frame dari webcam
            if not ret:
                print("Tidak ada frame yang bisa dibaca.")
                break

            result_frame = process_img(frame, face_detection)
            cv2.imshow('Webcam', result_frame)  # Tampilkan frame yang telah diproses

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Keluar jika tombol 'q' ditekan
                break

        cap.release()
        cv2.destroyAllWindows()  # Tutup semua jendela