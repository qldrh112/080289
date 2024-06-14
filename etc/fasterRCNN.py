import os
import xml.etree.ElementTree as ET
import torch
import torch.nn.utils as utils
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
import traceback
from collections import defaultdict


def calculate_precision_recall_ap_per_label(predictions, targets, iou_threshold=0.5):
    # 레이블별 성능 계산을 위한 구조 초기화
    label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})

    # 각 타겟에 포함된 레이블별 개수 계산
    for target in targets:
        for gt_box, gt_label in zip(target['boxes'], target['labels']):
            label_metrics[gt_label.item()]['total_gt'] += 1

    # 예측값과 실제값 비교
    for prediction, target in zip(predictions, targets):
        matched_gt = set()
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']
        pred_labels = prediction['labels']

        for i, (p_box, p_score, p_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_gt_idx = -1
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if p_label == gt_label and j not in matched_gt:
                    iou = calculate_iou(gt_box, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > iou_threshold:
                label_metrics[p_label.item()]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                label_metrics[p_label.item()]['fp'] += 1

        for j, gt_label in enumerate(gt_labels):
            if j not in matched_gt:
                label_metrics[gt_label.item()]['fn'] += 1

    # 레이블별 정밀도, 재현율, AP 계산
    results = {}
    for label, counts in label_metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        total_gt = counts['total_gt']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        ap = (precision * recall)  # Simplified AP
        results[label] = {'precision': precision, 'recall': recall, 'AP': ap}

    return results

def calculate_iou(gt_box, pred_box):
    """Calculate intersection over union for the specified ground truth and prediction boxes"""
    x1_t, y1_t, x2_t, y2_t = gt_box['bbox']
    x1_p, y1_p, x2_p, y2_p = pred_box['bbox']

    x1_i = max(x1_t, x1_p)
    y1_i = max(y1_t, y1_p)
    x2_i = min(x2_t, x2_p)
    y2_i = min(y2_t, y2_p)

    area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area_t = (x2_t - x1_t) * (y2_t - y1_t)
    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_u = area_t + area_p - area_i

    return area_i / area_u if area_u != 0 else 0

def calculate_precision_recall_ap(predictions, targets, iou_threshold=0.5):
    """Calculate precision, recall, and AP for the given predictions and targets"""
    tp = 0
    fp = 0
    fn = 0
    n_positives = 0

    # Count total positive targets
    for target in targets:
        n_positives += len(target['boxes'])

    for prediction, target in zip(predictions, targets):
... (366줄 남음)
접기
message.txt
17KB
코드해설 필요하면 주말에 오세요
from ultralytics import YOLO
import cometml
import cv2

#cometapikey = 'r0vxHKbhI9MrufPfWBLSfsTPw'

cometml.init(project_name='seoulsystem-yolo8v')

model = YOLO("yolov9e.pt")

model.train(data = 'data/seoul/for_yolo/tld.yaml',
            device=0,
            epochs=15,
            imgsz=640,
            project='seoul-system-yolo9v',
            batch=8,
            save_period=1,
            save_json=True,
            name='test',
            exist_ok=True,)


if __name
 == "__main":
epoch = 15
imgsz = 640
output_path = 'llamaparse/yolotest'
main(epoch, imgsz, output_path)
이상하게 보내지는군
from ultralytics import YOLO
import comet_ml
import cv2

comet_ml.init(project_name='seoulsystem-yolo8v')

model = YOLO("yolov9e.pt")

model.train(data = 'data/seoul/for_yolo/tld.yaml',
            device=0,
            epochs=15,
            imgsz=640,
            project='seoul-system-yolo9v',
            batch=8,
            save_period=1,
            save_json=True,
            name='test',
            exist_ok=True,)
위에건 fasterrcnn
아래건 yolo
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
... (4줄 남음)
접기
message.txt
5KB
이거는 yolo를 이용한 개체 finder
6월 말부터 한 2주정도 학송선생이 내 룸메가 되기로 했거든요
종종 찾아오도록 하세요
이규석 — 오늘 오후 1:14
으하하하
마지막 줄이 가장 재미난 코드네요
아니네 [-2:]라고 해야하나
전에 yolo 코드 맨 처음 여기서 봤을 때
감도 안 잡혔는데 이제는 그래도 입질은 오네요
슈넬치킨맨 — 오늘 오후 1:40
욜로학습은
코멧에 계정파서
cometkey 주고 하세요
이규석 — 오늘 오후 1:42
얏호
﻿
import os
import xml.etree.ElementTree as ET
import torch
import torch.nn.utils as utils
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
import traceback
from collections import defaultdict


def calculate_precision_recall_ap_per_label(predictions, targets, iou_threshold=0.5):
    # 레이블별 성능 계산을 위한 구조 초기화
    label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})

    # 각 타겟에 포함된 레이블별 개수 계산
    for target in targets:
        for gt_box, gt_label in zip(target['boxes'], target['labels']):
            label_metrics[gt_label.item()]['total_gt'] += 1

    # 예측값과 실제값 비교
    for prediction, target in zip(predictions, targets):
        matched_gt = set()
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']
        pred_labels = prediction['labels']

        for i, (p_box, p_score, p_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_gt_idx = -1
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if p_label == gt_label and j not in matched_gt:
                    iou = calculate_iou(gt_box, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > iou_threshold:
                label_metrics[p_label.item()]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                label_metrics[p_label.item()]['fp'] += 1

        for j, gt_label in enumerate(gt_labels):
            if j not in matched_gt:
                label_metrics[gt_label.item()]['fn'] += 1

    # 레이블별 정밀도, 재현율, AP 계산
    results = {}
    for label, counts in label_metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        total_gt = counts['total_gt']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        ap = (precision * recall)  # Simplified AP
        results[label] = {'precision': precision, 'recall': recall, 'AP': ap}

    return results

def calculate_iou(gt_box, pred_box):
    """Calculate intersection over union for the specified ground truth and prediction boxes"""
    x1_t, y1_t, x2_t, y2_t = gt_box['bbox']
    x1_p, y1_p, x2_p, y2_p = pred_box['bbox']

    x1_i = max(x1_t, x1_p)
    y1_i = max(y1_t, y1_p)
    x2_i = min(x2_t, x2_p)
    y2_i = min(y2_t, y2_p)

    area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area_t = (x2_t - x1_t) * (y2_t - y1_t)
    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_u = area_t + area_p - area_i

    return area_i / area_u if area_u != 0 else 0

def calculate_precision_recall_ap(predictions, targets, iou_threshold=0.5):
    """Calculate precision, recall, and AP for the given predictions and targets"""
    tp = 0
    fp = 0
    fn = 0
    n_positives = 0

    # Count total positive targets
    for target in targets:
        n_positives += len(target['boxes'])

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction['boxes']
        gt_boxes = target['boxes']

        # Match detections to ground truth
        matched = set()
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if j in matched:
                    continue
                if calculate_iou(gt_box, pred_box) > iou_threshold:
                    tp += 1
                    matched.add(j)
                    break
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (n_positives) if n_positives > 0 else 0
    ap = precision * recall  # Simplified AP calculation for example purposes

    return precision, recall, ap






def parse_annotation(xml_file, label_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        labels.append(label_map[label])  # 레이블 매핑 사용
        xmlbox = obj.find('bndbox')
        bbox = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        boxes.append(bbox)

    return boxes, labels

class CustomDataset(Dataset):
    def __init__(self, root, label_map, transforms=None, target_size=(3676, 2715)):
        self.root = root
        self.transforms = transforms
        self.target_size = target_size        
        self.label_map = label_map      
        self.file_names = []  
        
        # 이미지와 어노테이션 파일 리스트
        imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

        # 이미지와 어노테이션의 쌍이 일치하는지 확인
        self.imgs = []
        self.annotations = []
        for img in imgs:
            # annot = img.replace('.jpeg', '.xml')  # 이미지 파일 이름을 어노테이션 파일 이름으로 변경
            # annot = img.split('.')[0]+'.xml'
            if img[-4] == '.':
                annot = img[:-4]+'.xml'
            elif img[-5] == '.':
                annot = img[:-5]+'.xml'
            if annot in annotations:  # 어노테이션이 있는 이미지만 사용
                self.file_names.append(img)
                self.imgs.append(img)
                self.annotations.append(annot)
        
    def __getitem__(self, idx):
        # 이미지와 어노테이션 파일의 이름 가져오기
        img_filename = self.imgs[idx]
        annot_filename = self.annotations[idx]

        # 이미지 파일과 어노테이션 파일의 경로 생성
        img_path = os.path.join(self.root, "img", img_filename)
        annot_path = os.path.join(self.root, "annotations", annot_filename)

        # 이미지 열기
        img = Image.open(img_path).convert("RGB")

        # 이미지 크기 조절
        img = img.resize(self.target_size, Image.BILINEAR)

        # 어노테이션 파싱
        boxes, labels = parse_annotation(annot_path, self.label_map)

        # 바운딩 박스 조정
        width_ratio = self.target_size[0] / img.width
        height_ratio = self.target_size[1] / img.height

        new_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            x_min *= width_ratio
            y_min *= height_ratio
            x_max *= width_ratio
            y_max *= height_ratio
            new_boxes.append([x_min, y_min, x_max, y_max])

        num_objs = len(new_boxes)
        new_boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        # print(img_filename)
        area = (new_boxes[:, 3] - new_boxes[:, 1]) * (new_boxes[:, 2] - new_boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = new_boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, [annot_filename]

    def __len__(self):
        return len(self.imgs)
    
    def split_dataset(self, train_ratio=0.9):
        total_length = len(self)
        train_length = int(total_length * train_ratio)
        test_length = total_length - train_length
        
        train_dataset = CustomDataset.__new__(CustomDataset)
        train_dataset.root = self.root
        train_dataset.transforms = self.transforms
        train_dataset.target_size = self.target_size
        train_dataset.label_map = self.label_map
        train_dataset.imgs = self.imgs[:train_length]
        train_dataset.annotations = self.annotations[:train_length]

        test_dataset = CustomDataset.__new__(CustomDataset)
        test_dataset.root = self.root
        test_dataset.transforms = self.transforms
        test_dataset.target_size = self.target_size
        test_dataset.label_map = self.label_map
        test_dataset.imgs = self.imgs[train_length:]
        test_dataset.annotations = self.annotations[train_length:]

        return train_dataset, test_dataset

# def get_model(num_classes):
#     # Load a pre-trained ResNet-101 model
#     backbone = backbone_utils.resnet_fpn_backbone(backbone_name="resnet101", pretrained=True)
#     num_classes = num_classes

#     model = FasterRCNN(
#         backbone=backbone,
#         num_classes=num_classes
#     )

#     return model

def get_model(num_classes):
    # Load a pre-trained vgg16 model
    get_vgg16 = torchvision.models.vgg16(pretrained=True)
    backbone = torch.nn.Sequential(*list(get_vgg16.features.children()))

    backbone.out_channels = 512
    
    rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator
    )

    return model

# def get_model(num_classes):
#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    
    
#     return model

    

def collate_fn(batch):
    images = [transforms.ToTensor()(item[0]) for item in batch]  # Convert PIL.Image to tensor
    targets = [item[1] for item in batch]
    file_names = [item[2] for item in batch]
    
    images = torch.stack(images, dim=0)

    return images, targets, file_names

def calculate_iou(gt_box, pred_box):
    """Calculate intersection over union for the specified ground truth and prediction boxes"""
    # 좌표 추출
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    # 교차 영역의 좌표 계산
    x1_i = max(x1_t, x1_p)
    y1_i = max(y1_t, y1_p)
    x2_i = min(x2_t, x2_p)
    y2_i = min(y2_t, y2_p)

    # 교차 영역의 넓이
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    if inter_area == 0:
        return 0.0

    # 각 박스의 넓이
    area_t = (x2_t - x1_t) * (y2_t - y1_t)
    area_p = (x2_p - x1_p) * (y2_p - y1_p)

    # 합집합의 넓이
    union_area = area_t + area_p - inter_area

    # IoU 계산
    return inter_area / union_area

def calculate_accuracy(predictions, targets, iou_threshold=0.5):
  
    total_correct = 0
    total_samples = 0

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction['boxes']
        pred_labels = prediction['labels']

        target_boxes = target['boxes']
        target_labels = target['labels']

        iou_matrix = calculate_iou(pred_boxes, target_boxes)

        for i in range(len(pred_boxes)):
            max_iou = np.max(iou_matrix[i, :])
            if max_iou >= iou_threshold and pred_labels[i] == target_labels[np.argmax(iou_matrix[i, :])]:
                total_correct += 1

        total_samples += len(pred_boxes)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device} device")
    root = "data/seoul/extension_data"
    iou_threshold = 0.5
    
    #레이블 변동에 따라 수정 필요
    label_map = {'feature': 1, 'square':2}
    # label_map = {'text': 1, 'image':2, 'table':3}
    

    # Use our dataset and defined transformations
    target_size = (640, 640) #전체 데이터의 이미지 너비 / 높이 평균
    dataset = CustomDataset(root, label_map, target_size=target_size)

    train_dataset, test_dataset = dataset.split_dataset(train_ratio=0.9)

    # print(len(train_dataset))  # 데이터셋의 크기 출력
    # if len(train_dataset) == 0:
    #     print("Training dataset is empty. Check the dataset loading logic.")
    
    # 데이터 로더 생성
    batch_size = 4
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print(train_data_loader)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
       
    # Define the model
    model = get_model(num_classes=len(label_map)+1)  # For four classes + background
    model.to(device)

    num_epochs = 10

    # Define optimizer and loss function
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
   
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets, file_names in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                images = list(image.to(device) for image in images)
                # images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # print(images[0].shape)
                # print(len(images[0].shape))
                # print(model)
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backpropagation
                optimizer.zero_grad()
                losses.backward()

                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if not torch.isnan(losses).any():
                    running_loss += losses.item()
                else:
                    print("Warning: NaN detected in losses.")
            except Exception as ex:
                print(file_names)
                err_msg = traceback.format_exc()                
                print(err_msg)
                with open('seoul/error_list.txt', 'a', encoding='utf-8')as f:
                    f.write(err_msg)
                    f.write(file_names[0][0])
                    f.write('\n')
                    
                
            
        lr_scheduler.step()

        avg_loss = running_loss / len(train_data_loader) if running_loss != 0 else float('nan')
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss}")

        # 모델 저장        
            
        torch.save(model.state_dict(), f"seoul/checkpoint/vgg16/model_epoch_{epoch+1}.pth")
        print(f"Model saved as model_epoch_{epoch+1}.pth")

        # 테스트 반복문
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets, file_names in tqdm(test_data_loader, desc=f"Testing Epoch {epoch+1}/{num_epochs}"):
                try:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # 모델 예측
                    predictions = model(images)

                    all_predictions.extend(predictions)
                    all_targets.extend(targets)
                except Exception as e:
                    print(f"Error processing {file_names}: {e}")

            # 정밀도, 재현율 및 AP 계산
            results = calculate_precision_recall_ap_per_label(all_predictions, all_targets, iou_threshold=0.5)
            for label, metrics in results.items():
                with open('seoul/result.txt', 'a', encoding='utf-8') as f:
                    f.write(f"epoch:{epoch}|| Label {label}: Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, AP: {metrics['AP']:.2f}\n")
                print(f"Label {label}: Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, AP: {metrics['AP']:.2f}")

if __name__ == "__main__":
    main()
message.txt
17KB