from ultralytics import YOLO

if __name__ == '__main__':
    # เทรนรอบ 9 บน mixed6 (DTPG v6: แก้ label รูพรุน good->defect + เกลี่ย street ใหม่)
    # train 1714 / valid 93 / test 56 (valid/test ชุดเดิม)
    # ต่อจาก best_round8_mixed5.pt (nc=6 ตรงกัน) สูตรเดิม: 80e + cls=1.0 + copy_paste/mixup
    model = YOLO('all_model/best_round8_mixed5.pt')

    print("ROUND9 mixed6: fine-tune from best_round8 ...")
    results = model.train(
        data='mixed6/data_local.yaml',
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        cls=1.0,
        copy_paste=0.3,
        mixup=0.2,
        name='pathongko_v6',
    )

    print("DONE round9 mixed6")
