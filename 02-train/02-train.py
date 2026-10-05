from ultralytics import YOLO

if __name__ == '__main__':
    # เทรนรอบ 3 บน mixed2 หลังแก้บั๊ก label (polygon→bbox ครบ 100%, train 713 ภาพ)
    # ชดเชยคลาสน้อย (street_burnt_good 47 / brand_golden_defect 51 กล่อง เทียบ street_golden_defect 298):
    #  - cls=1.0       เพิ่มน้ำหนัก classification loss (default 0.5)
    #  - copy_paste/mixup  สังเคราะห์บริบทให้วัตถุคลาสน้อยเห็นบ่อยขึ้น
    model = YOLO('yolov8n.pt')

    print("🚀 เทรนรอบ 3: mixed2 label แก้แล้ว + ชดเชย class imbalance (80 epochs)...")
    results = model.train(
        data='mixed2/data_local.yaml',  # path สัมบูรณ์ (data.yaml เดิมชี้ ../train ผิดที่)
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,                    # GPU ใบที่ 0
        workers=4,
        patience=20,                 # early stopping ถ้าไม่ดีขึ้น 20 epochs
        cls=1.0,                     # เน้น classification loss ชดเชยคลาสน้อย
        copy_paste=0.3,              # สุ่มแปะวัตถุเพิ่ม
        mixup=0.2,                   # ผสมภาพเพิ่มความหลากหลาย
        name='pathongko_mixed3_fixed'  # ผลใน runs/detect/ (แยกจากรอบ 2)
    )

    print("🎉 เทรนรอบ 3 เสร็จสิ้น!")
