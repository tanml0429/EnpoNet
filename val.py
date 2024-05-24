

import os, sys
from pathlib import Path
here = Path(__file__).parent
p = f'{here.parent}'
if p not in sys.path:
    sys.path.append(p)
from polyp.apis import YOLO, LYMO
from dataclasses import dataclass, field
import hepai


def run(args):
    # Create a new YOLO model from scratch
    # model_name = args.pop('model')
    kwargs = args.__dict__
    model_name_or_cfg = kwargs.pop('model')
    model_weights = kwargs.pop('weights', None)
    LYMO.apply_improvements()
    model = LYMO(model_name_or_cfg)
    # model = YOLO(model_name_or_cfg)

    if model_weights:
        model = model.load(model_weights)
    
    # model = YOLO(model_name).load(model_weights)

    # results = model.train(**kwargs)

    # Evaluate the model's performance on the validation set
    results = model.val(data=args.data, fine_cls=args.fine_cls)  # results是validator.metrics
    print(results)

    # Perform object detection on an image using the model
    # results = model(f'{here}/lymonet/data/scripts/image.png')
    # print(results)

    # Export the model to ONNX format
    # success = model.export(format='onnx')

@dataclass
class Args:
    model: str =  '/home/tml/VSProjects/polyp_mixed/runs/detect/train7/weights/best.pt'
    mode: str = 'val'
    val: bool = True
    # model: str =  f'{here}/lymonet/configs/yolov8s_1MHSA_CA.yaml'
    # model: str = "yolov8x.yaml"
    # weights: str = 'yolov8n.pt'
    data: str = f'{here}/polyp/configs/polypsset.yaml'
    split: str = 'val'
    # epochs: int = 300
    batch: int = 16
    imgsz: int = 640
    workers: int = 80
    device: str = '0'  # GPU id 
    project: str = 'runs/val'
    name: str = 'polyp'
    # patience: int = 0
    # dropout: float = 0.51
    fine_cls: str = False  # 是否使用精细分类模型
    
 
if __name__ == '__main__':
    args = hepai.parse_args_into_dataclasses(Args)
    run(args)