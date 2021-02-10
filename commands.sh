python detect.py --source ~/data/project.avi --weights weights/best.pt
python train.py --weights weights/yolo-tiny.pt
python test.py --weights weights/best.pt
