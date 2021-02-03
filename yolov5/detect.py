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
homograpfy_file = open('../data/homography_list.txt', 'rb')
homography_list = pickle.load(homograpfy_file)

template_np, template_torch = field.read_template()

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

# def draw_boxes(img, bbox, identities=None, offset=(0, 0)):

#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         id = int(identities[i]) if identities is not None else 0
#         if id in players.keys():
#             current_player = players.get(id)
#             # only if checking colors automatically:
#             # current_player.assignTeam(players)
#         else:
#             # check color manually
#             team, color = player.check_color_manual2(left_clicks,img,x1,x2,y1,y2)

#             # check color automatically
#             # color = player.detectPlayerColor(img,x1,x2,y1,y2)

#             current_player = player.Player(id,"Player"+str(id),color=color, x=x2-(x2-x1),y=y2)

#             players[id] = current_player
            
#         label = current_player.label
#         plot_one_box(box, img, label=label, color=(int(current_player.color[0]), int(current_player.color[1]), int(current_player.color[2])), line_thickness=1)
#         #plot_one_box(box, img, label=label, color=current_player.color, line_thickness=1)

#     return img

def draw_points(img, bbox, pitch_template, frame, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        if id in players.keys():
            current_player = players.get(id)
            # only if checking colors automatically:
            current_player.assignTeam(players)
            current_player.updatePosition(int(x2-((x2-x1)/2)), int(y2-5))
        else:
            # check color manually
            # team, color = player.check_color_manual2(left_clicks,img,x1,x2,y1,y2)

            # check color automatically
            color = player.detectPlayerColor(img,x1,x2,y1,y2)

            current_player = player.Player(id,"Player"+str(id),color=color, x=int(x2-((x2-x1)/2)),y=int(y2-5))

            players[id] = current_player

        # dst_x, dst_y = player.transformPosition(current_player.x, current_player.y, homography_list[frame])
        # print(dst_x, dst_y)
        # print("i: " + str(i) + " ----- frame: " + str(frame) + " ----- player: " + str(current_player.id))
        # cv2.circle(layer, (current_player.x,current_player.y), radius=5, color=(int(current_player.color[0]), int(current_player.color[1]), int(current_player.color[2])), thickness=-1)
        # cv2.circle(pitch_template, (current_player.x,current_player.y), radius=5, color=current_player.color, thickness=-1)
        # cv2.circle(pitch_template, (int(dst_x)*1000, int(dst_y)*1000), radius=5, color=current_player.color, thickness=-1)
    # layer = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # player.transformAllPositions(players, homography_list[frame])
    field.transform(homography_list[frame])
    # print(np.transpose(np.nonzero(layer)))
    # print(homography_list[frame].cpu().detach().numpy())
    # pitch_template = field.show_top_down(pitch_template, img, homography_list[frame])
    # # pitch_template = cv2.cvtColor(pitch_template, cv2.COLOR_RGB2BGR)
    # cv2.imshow('pitch', pitch_template)
    # # print(img[frame])
    # if cv2.waitKey(1) == ord('q'):  # q to quit
    #     raise StopIteration
    return img

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # colorsB = first_frame[y,x,0]
        # colorsG = first_frame[y,x,1]
        # colorsR = first_frame[y,x,2]
        global left_clicks
        colors = first_frame[y,x]
        left_clicks.append(colors)
        # print("Red: ",colorsR)
        # print("Green: ",colorsG)
        # print("Blue: ",colorsB)
        # print("BRG Format: ",colors)
        # print("Coordinates of pixel: X: ",x,"Y: ",y)

def detect(save_img=False):
    save_img, source, weights, view_img, save_txt, imgsz = opt.save_img, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

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

    # global first_frame
    # video = cv2.VideoCapture(source)
    # status, first_frame = video.read()
    # cv2.namedWindow('pick_color')
    # cv2.setMouseCallback('pick_color',pick_color)
    # while(1):
    #     cv2.imshow('pick_color',first_frame)
    #     if cv2.waitKey(20) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()
    # print(left_clicks)

    for path, img, im0s, vid_cap in dataset:
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
            if len(det):
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
                    draw_points(im0, bbox_xyxy, template_torch, frame, identities)
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
                        # dst_x, dst_y = player.transformPosition(xywh[0]*1000, xywh[1]*1000, homography_list)
                        # print(dst_x, dst_y)
                        # cv2.circle(template, (int(dst_x)*5, int(dst_y)*8), radius=3, color=[0,0,0], thickness=-1)
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                    # # print("xy: "+ str(xywh[:, :2]))
                    # # tutaj homografia współrzędnych!!!
                    # x = xywh[:,1]
                    # y = xywh[:,0]
                    
                    # print(x,y)

                    # xy = torch.stack([x, y, torch.ones_like(x)])
                    # xy_warped = torch.matmul(torch.inverse(homography_list[frame]), xy)

                    # xy_warped, z_warped = xy_warped.split(2, dim=1)

                    # xy_warped, z_warped = xy_warped.split(2, dim=1)
                    # xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
                    # x_warped, y_warped = torch.unbind(xy_warped, dim=1)
                    # print(x_warped.squeeze(0).squeeze(0).numpy(), y_warped.squeeze(0).squeeze(0).numpy())
                    # out_shape = template_torch.shape[1:3]
                    # print(x_warped.view(*out_shape[-2:]))
                    # x_np = x_warped.squeeze(0).squeeze(0).numpy()
                    # y_np = y_warped.squeeze(0).squeeze(0).numpy()
                    # print("x: "+str(x_np[0][0]))
                    # print("y: "+str(y_np[0][0]))
                    # dst = np.dot(torch.inverse(homography_list[frame]), xy)
                    # dst = cv2.perspectiveTransform(xy, torch.inverse(homography_list[frame]))
                    # dst_x, dst_y = player.transformPosition(xywh[0], xywh[1], homography_list[frame])
                    # print("dst_xy: "+ str(dst))
                    # cv2.circle(template_np, (int(x_np), int(x_np)), radius=4, color=[0,0,0], thickness=-1)

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

            pitch_template = field.show_top_down(template_torch, im0, homography_list[frame])

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                # pitch = cv2.cvtColor(template_np, cv2.COLOR_RGB2BGR)
                cv2.imshow("pitch", pitch_template)

                # cv2.imshow('template', template_np)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            
            # layer = np.zeros((im0.shape[0], im0.shape[1], 3))

            # for player in players:
            #     x, y = players[player].getPosition()
            #     color = players[player].getColor()
                # cv2.circle(template_np,(int(dst[0].item()),int(dst[1].item())) , 8, (50,255,0), -1)
                # cv2.circle(layer, (x,y), radius=5, color=color, thickness=-1)
                # print ("x: "+str(x)+ " y: "+str(y)+ " color: "+ str(color)+ " layer: "+str(layer))
            
            # top_down = field.show_top_down(template_torch, im0, homography_list[frame])
            # layer = field.show_field(template_torch, im0, homography_list[frame])
            # layer = cv2.cvtColor(layer, cv2.COLOR_RGB2BGR)
            # for player in players:
            #     x, y = players[player].getPosition()
            #     team = players[player].getTeam()
            #     print(team)
            #     color = players[player].getColor()
            #     cv2.circle(layer, (x,y), radius=5, color=color, thickness=-1)

            # layer = layer*0.5+im0
            # cv2.imshow('pitch', layer)
            # cv2.waitKey(1)
            # Save results (image with detections)
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
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
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
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
