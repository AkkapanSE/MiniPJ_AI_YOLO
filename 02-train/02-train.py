from ultralytics import YOLO

if __name__ == '__main__':
    # เทรนรอบ 6 บน mixed3 (DTPG v3 + street2 61 ภาพที่ผ่าน review):
    # train 764 / valid 93 / test 56 (test มี street_burnt_good แล้ว)
    # สูตรเดิมรอบ 3: yolov8n 80e + cls=1.0 + copy_paste/mixup ชดเชยคลาสน้อย
    model = YOLO('all_model/yolov8n.pt')

    print("🚀 เทรนรอบ 6: yolov8n 80e บน mixed3 ...")
    results = model.train(
        data='mixed3/data_local.yaml',
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,                    # GPU ใบที่ 0
        workers=4,
        patience=20,                 # early stopping ถ้าไม่ดีขึ้น 20 epochs
        cls=1.0,                     # เน้น classification loss ชดเชยคลาสน้อย
        copy_paste=0.3,              # สุ่มแปะวัตถุเพิ่ม
        mixup=0.2,                   # ผสมภาพเพิ่มความหลากหลาย
        name='pathongko_mixed6'      # อย่าทับรอบ 3 (best_round3.pt ยังเป็นตัวสำรอง)
    )

    print("🎉 เทรนรอบ 6 เสร็จสิ้น!")
