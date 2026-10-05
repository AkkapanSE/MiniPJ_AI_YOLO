from ultralytics import YOLO

if __name__ == '__main__':
    # เทรนรอบ 5 บน mixed2_os (oversample คลาส 3/0/2: train 713→1164, สัดส่วน 6.3:1→2.3:1)
    # เป้าหมาย: ดัน street_burnt_defect (test R 0.56) + street_burnt_good ให้สำเร็จ
    model = YOLO('yolov8s.pt')

    print("🚀 เทรนรอบ 5: yolov8s + oversample + ชดเชย class imbalance (100 epochs)...")
    results = model.train(
        data='mixed2_os/data_local.yaml',  # train oversample, valid/test ชุดเดิม
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,                    # GPU ใบที่ 0
        workers=4,
        patience=20,                 # early stopping ถ้าไม่ดีขึ้น 20 epochs
        cls=1.0,                     # เน้น classification loss ชดเชยคลาสน้อย
        copy_paste=0.3,              # สุ่มแปะวัตถุเพิ่ม
        mixup=0.2,                   # ผสมภาพเพิ่มความหลากหลาย
        name='pathongko_mixed5_os'  # ผลใน runs/detect/ (แยกจากรอบ 4)
    )

    print("🎉 เทรนรอบ 5 เสร็จสิ้น!")
