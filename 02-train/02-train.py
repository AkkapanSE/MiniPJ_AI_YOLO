from ultralytics import YOLO

if __name__ == '__main__':
    # 1. โหลด Pre-trained Model (ใช้ yolov8n.pt เป็นฐานเริ่มต้น)
    model = YOLO('yolov8n.pt')

    # 2. สั่งเริ่มเทรนโมเดลด้วย Dataset รวมในโฟลเดอร์ mixed
    print("🚀 เริ่มต้นการเทรนโมเดลปาท่องโก๋ (Mixed Dataset)...")
    results = model.train(
        data='mixed/data_local.yaml',  # path สัมบูรณ์ (data.yaml เดิมชี้ ../train ผิดที่)
        epochs=50,                   # จำนวนรอบการเทรน
        imgsz=640,                   # ขนาดรูปภาพมาตรฐาน
        batch=16,                    # ขนาด Batch size (ถ้า VRAM ไม่พอ ปรับลดเป็น 8 ได้)
        device=0,                    # ใช้ GPU การ์ดใบที่ 0 (ถ้าไม่มี GPU ให้ตัดบรรทัดนี้ออก)
        workers=4,                   # จำนวน worker สำหรับโหลดข้อมูล (ปรับลดเป็น 2 หรือ 0 ได้ถ้าเจอปัญหาบน Windows)
        name='pathongko_mixed_model' # ชื่อโฟลเดอร์ที่จะใช้บันทึกผลลัพธ์ใน runs/detect/
    )

    print("🎉 เทรนโมเดลเสร็จสิ้นเรียบร้อยแล้ว!")