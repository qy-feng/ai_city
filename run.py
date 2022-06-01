import argparse
import time
import glob
import pickle
import os
from skimage import io
import cv2
import numpy as np
from tracker import base
from utils.homograph import transform, cal_homograph
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

color_seed = 50
colors = np.random.rand(color_seed, 3)
frame_num = 1251
target_cls = [1, 2, 3, 4, 5, 6, 7, 8, 67]


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--data_path", default='../data/Unleash', help="Path to data.", type=str)
    parser.add_argument("--detr_path", default='../mmdetection/results', help="Path to detection results.", type=str)
    parser.add_argument("--out", default='output', help="Path to tracking results.", type=str)
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold",
                        help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


def select_bbox_by_class(target_cls, bbs, conf_thres=0.25):
    bboxes = []
    for cls in target_cls:
        for bb in bbs[cls]:
            if bb[-1] > conf_thres:
                # bb.append(cls)
                bboxes.append(bb)
    return np.array(bboxes)


def track(args):
    if(args.display):
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    # build a mot tracker
    mot_tracker = base.Sort(max_age=args.max_age,
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold)

    with open(os.path.join(args.out, 'track_result.txt'), 'w') as out_file:
        for frame in range(frame_num):
            dets = pickle.load(open(os.path.join(args.detr_path, 'frame_%d.pkl' % frame), 'rb'))
            dets = select_bbox_by_class(target_cls, dets)

            if(args.display):
                fn = os.path.join(args.data_path, 'frames',
                                  '%06d.jpg' % (frame))
                im = io.imread(fn)
                ax1.imshow(im)
                plt.title('Tracked Targets')

            trackers = mot_tracker.update(dets)

            for d in trackers:
                x0, y0, x1, y1 = d[0], d[1], d[2], d[3]
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                      (frame, d[4], x0, y0, x1, y1), file=out_file)
                if(args.display):
                    d = d.astype(np.int32)
                    ax1.add_patch(patches.Rectangle(
                        (x0, y0), x1-x0, y1-y0, fill=False, lw=2, ec=colors[d[4] % color_seed, :]))

            if(args.display):
                fig.canvas.flush_events()
                plt.draw()
                ax1.cla()

    if(args.display):
        print("Note: to get real runtime results run without the option: --display")


def visualize(track_res):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    h_s, w_s = 1194, 1954
    h_t, w_t = 487, 742
    for frame in range(frame_num):
        im = cv2.imread('final-view.png')
        # rotate image
        # im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax1.imshow(im)
        plt.title('Tracked Targets')
        plt.axis('off')
        for target in track_res[str(frame)]:
            target_id = target['id']
            loc = target['trans']
            # change location
            loc = [h_s - loc[1], loc[0]]
            loc = [loc[0]/h_s*w_t, loc[1]/w_s*h_t]
            # print(loc)
            if loc[1] < h_t and loc[0] < w_t:
                ax1.add_patch(patches.Circle(
                            (loc[0], loc[1]), radius=10, fill=False, lw=2, 
                            ec=colors[target_id % color_seed, :]))
                ax1.text(loc[0], loc[1], 'Id-%d'%target_id, fontsize=8, fontweight='bold',
                        color=colors[target_id % color_seed, :])
            else:
                print('%f, %f, out of view')
        
        # cv2.imwrite(os.path.join(args.out, 'track', '%06d.jpg'%frame), im)
        plt.savefig(os.path.join('output', 'track', '%06d.jpg'%frame), bbox_inches='tight')
        fig.canvas.flush_events()
        plt.draw()
        ax1.cla()


def main(args):
    # detect & track
    results = track(args)

    # perspective transform
    track_res = dict()
    for frame in range(frame_num):
        track_res[str(frame)] = []

    modes = ['near', 'mid', 'far']
    Hs = dict()
    for m in modes:
        Hs[m] = cal_homograph(args.data_path, m)

    with open(os.path.join(args.out, 'track_result.txt'), 'r') as f:
        for line in f.readlines():
            frame, d, x0, y0, x1, y1, _,_,_,_ = line.split(',')
            bbox = [float(x0), float(y0), float(x1), float(y1)]
            target = transform(bbox, Hs)
            track_res[str(frame)].append({'id': int(d),
                                        'bbox': bbox,
                                        'trans': target})

    visualize(track_res)
        

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    main(args)