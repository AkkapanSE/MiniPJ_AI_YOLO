from ultralytics import YOLO

if __name__ == '__main__':
    # เทรนรอบ 8 บน mixed5 (DTPG v5: 852 เก่า + 1011 ใหม่ที่ผ่าน review)
    # train 1714 / valid 93 / test 56 (test มี street_burnt_good 8 กล่องแล้ว)
    # ต่อจาก best_round3.pt (nc=6 ตรงกัน) สูตรเดิม: 80e + cls=1.0 + copy_paste/mixup
    model = YOLO('all_model/best_round3.pt')

    print("ROUND8 mixed5: fine-tune from best_round3 ...")
    results = model.train(
        data='mixed5/data_local.yaml',
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        cls=1.0,
        copy_paste=0.3,
        mixup=0.2,
        name='pathongko_mixed5',
    )

    print("DONE round8 mixed5")
