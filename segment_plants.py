    import marimo

    __generated_with = "0.20.2"
    app = marimo.App(width="medium")


    @app.cell
    def _():
        from ultralytics import YOLO
        # Load a COCO-pretrained YOLO26n model
        model = YOLO("yolo26m-seg.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="yolo_dataset/dataset.yaml", epochs=300,
            imgsz=640,
            batch=4,
            patience=50,      # ← Early stopping
            lr0=0.001,        # ← Меньший LR
            augment=True,     # ← Аугментация (критично!)
            mixup=0.1,        # ← Mixup
            copy_paste=0.1    # ← Copy-paste)
                            )
        # Run inference with the YOLO26n model on the 'bus.jpg' image
        return (YOLO,)


    @app.cell
    def _(YOLO):
        # Результат сохранится в: runs/predict/exp/image.jpg
        from PIL import Image
        import matplotlib.pyplot as plt
        model_test = YOLO("runs/segment/train/weights/best.pt")
        results_test = model_test("hackaton/cv_solv/УФИЦРАН19022026/arugula/arugula_20260219162118326.jpg",conf=0.2,  save=True,
            save_crop=True)
        for r in results_test:
            # r.plot() возвращает numpy array с наложенными масками/bbox
            plt.figure(figsize=(15, 10))
            plt.imshow(r.plot())
            plt.axis('off')
            plt.show()

            # Детали детекций
            print(f"Детекций: {len(r.boxes)}")
            print(f"Масок: {len(r.masks) if r.masks is not None else 0}")
        return


    @app.cell
    def _():
        return


    @app.cell
    def _():
        return


    if __name__ == "__main__":
        app.run()
