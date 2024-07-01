# 라이브러리 가져오기
import os
# XML 파일을 파싱하고 생성할 수 있는 모듈, XML 문서를 트리 구조로 다룸
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
# 다양한 백본 모델을 생성하고, 설정하는 데 사용, 객체 탐지 모델에서 특징 추출기로 사용
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
# 예외 발생 시 트레이스백 정보를 포맷팅하고 출력할 수 있는 모듈, 디버깅을 도와 줌
import traceback
from collections import defaultdict


def calculate_precision_recall_ap_per_label(predictions, targets, iou_threshold=0.5):
    # 레이블별 성능 계산을 위한 구조 초기화
    # label_metrics[레이블]['tp'] 등으로 접근
    # 레이블에 접근할 때 자동으로 아래 기본값이 설정
    label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0})

    # 각 타겟에 포함된 레이블(사람, 자동차 등)별 개수 계산
    for target in targets:
        # gt(ground truth):  실제 값이나 정답을 의미
        for gt_box, gt_label in zip(target['boxes'], target['labels']):
            label_metrics[gt_label.item()]['total_gt'] += 1

    # 예측값과 실제값 비교
    for prediction, target in zip(predictions, targets):
        # 매칭된 박스
        matched_gt = set()
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']
        pred_labels = prediction['labels']

        # 최고의 iou와 그 인덱스를 갱신
        for i, (p_box, p_score, p_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_gt_idx = -1
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                # 예측과 실제가 같으면
                if p_label == gt_label and j not in matched_gt:
                    iou = calculate_iou(gt_box, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # iou 문지방을 넘어라
            if best_iou > iou_threshold:
                label_metrics[p_label.item()]['tp'] += 1
                matched_gt.add(best_gt_idx)
            else:
                label_metrics[p_label.item()]['fp'] += 1

        # 매칭되지 않은 것은 모두 false negative로 간주
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
        # AP: Precision-Recall 곡선 아래 면적을 계산
        ap = (precision * recall)  # Simplified AP
        results[label] = {'precision': precision, 'recall': recall, 'AP': ap}

    return results

def calculate_iou(gt_box, pred_box):
    """Calculate intersection over union for the specified ground truth and prediction boxes"""
    """지정된 실제 박스와 예측 박스의 교차 비율(Intersection over Union)을 계산합니다."""
    x1_t, y1_t, x2_t, y2_t = gt_box['bbox']
    x1_p, y1_p, x2_p, y2_p = pred_box['bbox']

    x1_i = max(x1_t, x1_p)
    y1_i = max(y1_t, y1_p)
    x2_i = min(x2_t, x2_p)
    y2_i = min(y2_t, y2_p)

    # 교집합 면적
    area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area_t = (x2_t - x1_t) * (y2_t - y1_t)
    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    # 합집합 면적
    area_u = area_t + area_p - area_i

    # 교집합 면적을 합집합 면적으로 나눈 비율(IoU)를 반환
    return area_i / area_u if area_u != 0 else 0

def calculate_precision_recall_ap(predictions, targets, iou_threshold=0.5):
    """예측과 실제 간 정밀도, 정확도, AP 계산"""
    tp = 0
    fp = 0
    fn = 0
    n_positives = 0

    # 박스 수 세기
    for target in targets:
        n_positives += len(target['boxes'])

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction['boxes']
        gt_boxes = target['boxes']

        # 실제와 같은 것 찾기
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
    # XML 요소에 접근할 수 있게 트리 구조로 변환
    # XML 문서는 계층적, 이미지에 대한 메타데이터를 담음
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []
    # 모든  <object> 요소를 반복
    for obj in root.iter('object'):
        label = obj.find('name').text
        labels.append(label_map[label])  # 레이블 매핑 사용
        xmlbox = obj.find('bndbox')
        bbox = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        boxes.append(bbox)

    return boxes, labels

class CustomDataset(Dataset):
    # 어노테이션이란 이미지에 대한 메타데이터
    def __init__(self, root, label_map, transforms=None, target_size=(3676, 2715)):
        self.root = root
        self.transforms = transforms 
        self.target_size = target_size        
        self.label_map = label_map      
        self.file_names = []  
        
        # 이미지와 어노테이션 파일 리스트(img, annotations 디렉토리에서 모든 파일을 가져 옴)
        imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

        # 이미지와 어노테이션의 쌍이 일치하는지 확인
        self.imgs = []
        self.annotations = []
        for img in imgs:
            # annot = img.replace('.jpeg', '.xml')  # 이미지 파일 이름을 어노테이션 파일 이름으로 변경
            # annot = img.split('.')[0]+'.xml'
            # 확장자의 차이가 있으므로
            if img[-4] == '.':
                annot = img[:-4]+'.xml'
            elif img[-5] == '.':
                annot = img[:-5]+'.xml'
            # 어노테이션이 있는 이미지만 사용
            # 이거 중복해서 값이 들어가는 거 아니야?
            if annot in annotations:  
                self.file_names.append(img)
                self.imgs.append(img)
                self.annotations.append(annot)
        
    def __getitem__(self, idx):
        # dataset[i] -> dataset.__getitem__(i)이 실행
        # 이미지와 어노테이션 파일의 이름 가져오기
        img_filename = self.imgs[idx]
        annot_filename = self.annotations[idx]

        # 이미지 파일과 어노테이션 파일의 경로 생성
        # C:/data/img/image1.jpg
        img_path = os.path.join(self.root, "img", img_filename)
        annot_path = os.path.join(self.root, "annotations", annot_filename)

        # 이미지 열기
        img = Image.open(img_path).convert("RGB")

        # 이미지 크기 조절, pillow 라이브러리에서 제공하는 이미지 리샘플링 필터  
        img = img.resize(self.target_size, Image.BILINEAR)

        # 어노테이션 파싱
        boxes, labels = parse_annotation(annot_path, self.label_map)

        # 원본 이미지가 다양한 크기를 가질 수 있기에 바운딩 박스 조정
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

        # 정수를 텐서로 변환
        image_id = torch.tensor([idx])
        # print(img_filename)

        # 각 바운딩 박스의 면적 계산
        area = (new_boxes[:, 3] - new_boxes[:, 1]) * (new_boxes[:, 2] - new_boxes[:, 0])
        # 이미지 내 객체의 수를 0으로 채워진 1차원 텐서를 만듦
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = new_boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 데이터 전처리나 augmentation이 필요할 때
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, [annot_filename]

    def __len__(self):
        return len(self.imgs)
    
    def split_dataset(self, train_ratio=0.9):
        """리스트 형태의 이미지와 어노테이션 파일을 다루므로 커스텀 메서드로 구현"""
        total_length = len(self)
        train_length = int(total_length * train_ratio)
        # test_length = total_length - train_length
        
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
    # 특징을 추출하는 데 사용하는 기본 네트워크
    backbone = torch.nn.Sequential(*list(get_vgg16.features.children()))

    backbone.out_channels = 512
    
    # 객체 탐지를 위한 기본적인 관심 영역(anchors)를 생성하여 객체를 예측
    # 크기와 비율을 설정하여 다양한 형태의 앵커 생성
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
    
    # 여러 텐서를 새로운 차원에 연결 행이 늘어나는 방식이다.
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
            # i번째 예측 박스와 각 실제 박스 사이의 ioU 중 가장 높은 값 선택
            max_iou = np.max(iou_matrix[i, :])
            # 문지방보다 ioU 값이 크거나 ioU 행렬의 i 번째 행에서 최대 ioU 값의 인덱스
            if max_iou >= iou_threshold and pred_labels[i] == target_labels[np.argmax(iou_matrix[i, :])]:
                total_correct += 1

        total_samples += len(pred_boxes)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


def main():
    # 사용 가능한 첫 번째 CUDA GPU를 지정함
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
    
    model = get_model(num_classes=len(label_map)+1)  # Faster R-CNN의 클래스는 정의한 클래스 + 배경
    model.to(device)

    num_epochs = 10

    # 모델의 학습 가능한 파라미터 리스트
    params = [p for p in model.parameters() if p.requires_grad]
    # momentum: 이전 기울기의 몇 %를 추가할 지, weight_decay: 가중치에 0.0005의 작은 페널티 부여
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # 매 3 번째 에포크마다 학습률을 10% 감소시킴
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
   
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # desc: 진행 바에 표시되는 설명
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

                # Backpropagation(역전파)
                # 역전파 실행 전 기울기 초기화
                optimizer.zero_grad()
                losses.backward()

                # 기울기의 크기를 제한, 기울기 폭주를 방지하여 학습 안정성 향상, max_norm = 1.0 기울기의 최대 L2 norm 값을 1로 제한
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
        # 모델의 학습 가능한 파라미터와 버퍼를 딕셔너리 형태로 반환, 모델의 상태를 저장하고 나중에 불러올 수 있다.
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