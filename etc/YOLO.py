import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

def nms(boxes, scores, iou_threshold):
    if not boxes:
        return []
    
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        inter = torch.prod(torch.min(boxes[order[1:]], boxes[i]) - torch.max(boxes[order[1:]], boxes[i]) + 1, 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (iou <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return keep

def save_detection_data(sns_output_dir, boxes, scores, labels):
    filename = os.path.join(sns_output_dir, "detections.txt")
    with open(filename, "w") as file:
        for box, score, label in zip(boxes, scores, labels):
            file.write(f"Label: {label}, Score: {score:.2f}, Box: [{', '.join(map(str, map(int, box)))}]\n")

def load_and_process_images(input_dir, output_dir, model, threshold=0.1, iou_thresh=0.5, target_labels=['person']):
    if not os.path.exists(input_dir):
        print(f"입력 디렉토리 {input_dir}가 존재하지 않습니다.")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_detections_dir = os.path.join(output_dir, "All_Detections")
    os.makedirs(all_detections_dir, exist_ok=True)

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path)
            results = model(image)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            boxes = []
            scores = []
            labels = []

            for result in results:
                if result.boxes.shape[0] > 0:
                    confs = result.boxes.conf.cpu().tolist()
                    for i, box in enumerate(result.boxes.xyxy):
                        label = result.names[result.boxes.ids[i]]
                        confidence = round(confs[i], 2)
                        if confidence > threshold and label in target_labels:
                            boxes.append(box.cpu().tolist())
                            scores.append(confidence)
                            labels.append(label)

            keep = nms(boxes, scores, iou_thresh)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
            labels = [labels[i] for i in keep]

            for index in keep:
                x1, y1, x2, y2 = map(int, boxes[index])
                confidence = scores[index]
                label = labels[index]
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x1, y1 - 10), f"{label} {confidence:.2f}", fill="red", font=font)

            original_save_path = os.path.join(all_detections_dir, image_name)
            image.save(original_save_path)
            print(f"{original_save_path}에 객체 인식 원본 이미지를 저장했습니다.")

            base_filename = os.path.splitext(image_name)[0]  # 확장자 제거
            for sns_name in ['facebook', 'instagram', 'twitter']:
                sns_output_dir = os.path.join(output_dir, base_filename, sns_name)
                os.makedirs(sns_output_dir, exist_ok=True)
                image_save_path = os.path.join(sns_output_dir, f"{sns_name}_{image_name}")
                image.save(image_save_path)
                print(f"{image_save_path}에 처리된 이미지를 저장했습니다.")
                save_detection_data(sns_output_dir, boxes, scores, labels)

if __name__ == "__main__":
    model = YOLO('seoul/checkpoint/yolov8/best.pt')

    input_dir = 'data/seoul/test'
    output_dir = 'seoul/output'
    threshold = 0.1
    iou_thresh = 0.1
    # sns = 'instagram' #'facebook' or 'twitter' or 'instagram' or None
    target_dict = {1:'feature', 2:'square'} #1: 1인 // 2:다인
    target_labels = [target_dict[1], target_dict[2]] #원하는 레이블만 입력 #1인: [target_dict[1]] #다인: [target_dict[2]]

    load_and_process_images(input_dir, output_dir, model, threshold, iou_thresh, target_labels)