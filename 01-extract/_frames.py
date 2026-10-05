import os
import cv2


def extract_frames(
    video_path, output_folder, frame_interval=20, prefix="frame"
):
    """สกัดภาพเฟรมจากวิดีโอตามช่วงระยะเฟรมที่กำหนด (frame_interval)"""
    if not os.path.exists(video_path):
        print(f"❌ ไม่พบไฟล์วิดีโอ: {video_path}")
        return

    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ไม่สามารถเปิดวิดีโอได้: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 กำลังประมวลผลไฟล์: {os.path.basename(video_path)}")
    print(
        f"   - FPS: {fps:.1f} | รวมทั้งหมด: {total_frames} เฟรม | สกัดทุกๆ {frame_interval} เฟรม"
    )

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # สกัดภาพเฉพาะเฟรมที่หาร frame_interval ลงตัว
        if frame_count % frame_interval == 0:
            output_filename = f"{prefix}_{saved_count:04d}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(
        f"✅ สกัดภาพเสร็จสิ้น! บันทึกไปทั้งหมด {saved_count} ภาพ ลงใน: {output_folder}\n"
    )


if __name__ == "__main__":
    # --- 1. สกัดคลิปปาท่องโก๋แบรนด์ (1004(1).mp4) บันทึกลง raw_images/brand1 ---
    extract_frames(
        video_path="raw_images/vdo/1004(1).mp4",
        output_folder="raw_images/brand1",
        frame_interval=15,  # ดึง 1 ภาพทุกๆ 15 เฟรม
        prefix="brand",
    )

    # --- 2. สกัดคลิปปาท่องโก๋ร้านทั่วไป (1004.mp4) บันทึกลง raw_images/streed1 ---
    extract_frames(
        video_path="raw_images/vdo/1004.mp4",
        output_folder="raw_images/streed1",
        frame_interval=15,  # ดึง 1 ภาพทุกๆ 15 เฟรม
        prefix="street",
    )