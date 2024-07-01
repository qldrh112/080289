import comet_ml
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import cv2

# Comet ML 초기화
# comet_ml.config.save(api_key='GUBL6LmpoNNY0QYc4WKz6OORR')
comet_ml.init(project_name='seoulsystem-yolo8v')

# YOLO 모델 초기화
model = YOLO("yolov9e.pt")

# 모델 훈련
model.train(data='data/seoul/for_yolo/tld.yaml',
            # 훈련에 사용할 GPU(0번 GPU 사용)
            device=0,
            epochs=15,
            # 입력 이미지 크기
            imgsz=640,
            project='seoul-system-yolo9v',
            batch=8,
            # 1 에포크마다 저장
            save_period=1,
            save_json=True,
            name='test',
            # True면 동일한 이름의 프로젝트가 있으면 덮어쓴다.
            exist_ok=True)

# 비최대 억제(NMS) 함수
def nms(boxes, scores, iou_threshold):
    """
    객체 탐지의 후처리 단계에서 중복된 박스를 제거하는 함수
    boxes: 탐지된 객체의 경계 상자 리스트
    scores: 탐지된 객체의 신뢰도 점수 리스트
    iou_threshold: NMS를 수행할 때 사용할 IoU 임계값
    """
    if not boxes:
        return []
    
    boxes = torch.tensor(boxes, dtype=torch.float32).to('cpu')
    scores = torch.tensor(scores, dtype=torch.float32).to('cpu')
    # boxes.T = boxes 텐서의 전치 => (N, 4)를 (4, N)를 바꿔 각 경계 상자의 좌표를 쉽게 접근할 수 있게 함
    x1, y1, x2, y2 = boxes.T
    # 경계 상자의 면적 계산, 픽셀 좌표계에서 계산의 정확성을 높이기 위해 1을 더함
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 신뢰도 점수를 기준으로 상자를 정렬 -> 0 차원이 아마도 신뢰도 점수겠죠?
    _, order = scores.sort(0, descending=True)
    keep = []

    # order.numel: order의 요소 수
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break   # while
        # 남은 모든 경계 상자와 현재 경계 상자의 교집합 영역 계산, prod: 요소를 곱함
        inter = torch.prod(torch.min(boxes[order[1:]], boxes[i]) - torch.max(boxes[order[1:]], boxes[i]) + 1, 1)
        # iou = 교집합 / a + b - 교집합
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # IoU가 threshold 이하인 요소의 인덱스를 반환, 반환된 인덱스 배열의 차원을 축소 -> 임계값 이하인 경계 상자의 인덱스를 1차원 배열로 반환
        ids = (iou <= iou_threshold).nonzero().squeeze()
        # 남은 경계 상자가 현재 경계 상자와 모두 너무 많이 겹쳐서 사용할 수 없음
        if ids.numel() == 0:
            break   # while
        # 경계 상자를 제외한 남은 경계 상자의 인덱스 갱신
        order = order[ids + 1]
    return keep


def save_detection_data(sns_output_dir, boxes, scores, labels):
    """
    객체 탐지 결과를 저장하는 함수
    sns_output_dir: 저장할 디렉토리
    boxes: 경계 상자 리스트
    scores: 신뢰도 점수 리스트
    labels: 탐지된 객체의 레이블 리스트
    """
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
    # 탐지 결과 저장 디렉토리
    os.makedirs(all_detections_dir, exist_ok=True)

    # 입력 디렉토리에서 이미지를 호출
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(image_path)
            # YOLO 모델을 사용하여 객체 탐지를 수행
            results = model(image)
            # image에 그릴 수 있게 하는 ImageDraw를 사용
            draw = ImageDraw.Draw(image)
            # 기본 폰트를 선택
            font = ImageFont.load_default()
            boxes = []
            scores = []
            labels = []

            for result in results:
                if result.boxes.shape[0] > 0:
                    confs = result.boxes.conf.cpu().tolist()
                    # 경계 상자의 좌표
                    for i, box in enumerate(result.boxes.xyxy):
                        # 탐지된 객체의 레이블 이름
                        label = result.names[result.boxes.ids[i]]
                        # confidence: 신뢰도, 해당 경계 상자가 실제 객체를 포함하고 있을 확률 계산
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
                # 위치를 지정, 경계 상자 바로 위에 그리기 위해 10을 뺌
                draw.text((x1, y1 - 10), f"{label} {confidence:.2f}", fill="red", font=font)

            original_save_path = os.path.join(all_detections_dir, image_name)
            image.save(original_save_path)
            print(f"{original_save_path}에 객체 인식 원본 이미지를 저장했습니다.")

            # 각 이미지를 원본과 SNS 별로 저장
            base_filename = os.path.splitext(image_name)[0]  # 확장자 제거
            for sns_name in ['facebook', 'instagram', 'twitter']:
                sns_output_dir = os.path.join(output_dir, base_filename, sns_name)
                os.makedirs(sns_output_dir, exist_ok=True)
                image_save_path = os.path.join(sns_output_dir, f"{sns_name}_{image_name}")
                # seoul/output/image1/facebook/facebook_image1.jpg
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