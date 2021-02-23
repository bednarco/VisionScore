import argparse
import time
import os
from pathlib import Path
import pickle
import cv2
import imageio
import sklearn
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import io

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from sportsfield_release import field
import player

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


template_np, template_torch = field.read_template()
template = cv2.imread('../data/template.png') 

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h
    
players = {}

def draw_points(img, bbox, pitch_template, h_frame, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        if id in players.keys():
            current_player = players.get(id)
            if len(players) >= 3:              
                player.detectMainColors(players)
                current_player.assignTeam(players)
            current_player.updatePosition(int(x2-((x2-x1)/2)), int(y2))
        else:
            color = player.detectPlayerColor(img,x1,x2,y1,y2)
            current_player = player.Player(id, isVisible=True, color=color, x=int(x2-((x2-x1)/2)), y=int(y2))
            players[id] = current_player

        plot_one_box(box, img, label=str(current_player.id), color=(int(current_player.color[0]), int(current_player.color[1]), int(current_player.color[2])), line_thickness=1)

    player.transformAllPositions(players, img.shape[0:2], h_frame, template_np.shape[0:2])

    
    player.drawAllPlayers(players, pitch_template)

    return img, pitch_template

def detect(save_img=False):
    save_img, source, weights, view_img, save_txt, imgsz, homography = opt.save_img, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.homography_file
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    homograpfy_file = open(homography, 'rb')
    # if cpu is used
    homography_list = CPU_Unpickler(homograpfy_file).load()
    # uncomment if gpu is used
    # homography_list = pickle.load(homograpfy_file)

    # Directories
    if save_img:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # DeepSort Initialize
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        # save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap, frame in dataset:
        # if frame > 50:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            pitch = template.copy()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = Path(path[i]), '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

                if save_img:
                    save_path = str(save_dir / p.name)
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det) >= 3:
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        if cls == 0:
                            x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                            if player.checkIfOnPitch(x_c, y_c, homography_list[frame-1], im0.shape[0:2]):
                                obj = [x_c, y_c, bbox_w, bbox_h]
                                bbox_xywh.append(obj)
                                confs.append([conf.item()])


                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)
                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_points(im0, bbox_xyxy, pitch, homography_list[frame-1], identities)
                        # draw_boxes(im0, bbox_xyxy, identities)

                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if  cls == 32 and (save_img or view_img):  # Add bbox to ball
                            #label = f'{names[int(cls)]}'
                            label = 'ball'
                            plot_one_box(xyxy, im0, label=label, color=[0,0,0], line_thickness=2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            x_norm = player.normalize(xywh[0], im0.shape[0:2][1])
                            y_norm = player.normalize(xywh[1], im0.shape[0:2][0])
                            normalizedPosition = np.array([[x_norm/2, y_norm/2]], dtype=np.float32)
                            warpedPosition = player.perspectiveTransform(normalizedPosition, homography_list[frame-1])
                            x_dst = player.denormalize(warpedPosition[0], template_np.shape[0:2][1])
                            y_dst = player.denormalize(warpedPosition[1], template_np.shape[0:2][0])
                            positionOnTemplate = (x_dst, y_dst)
                            cv2.circle(pitch, (int(positionOnTemplate[0]),int(positionOnTemplate[1])), 5, (255,255,255), -1)


                    # Write MOT compliant results to file
                    """ if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (j, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format """
                    
                    
                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.imshow('pitch', pitch)

                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                            vid_writer2 = cv2.VideoWriter(str(save_dir / 'pitch.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), vid_cap.get(cv2.CAP_PROP_FPS), (int(pitch.shape[1]), int(pitch.shape[0])))
                        vid_writer.write(im0)
                        vid_writer2.write(pitch)



    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 32], help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--homography-file", type=str, default="../data")
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
