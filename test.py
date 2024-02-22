import os
import glob
import argparse

from mmdet.apis import DetInferencer
from mmdet.apis import inference_detector, init_detector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='교통CCTV 돌발상황 분석 프로그램')
    parser.add_argument('--data_dir', required=True, help='이미지 폴더 경로를 입력하세요')
    parser.add_argument('--result_dir', required=True, help='분석 결과가 저장될 경로를 입력하세요')

    args = parser.parse_args()

    # 이미지 파일 경로 호출
    file_paths = glob.glob(os.path.join(args.data_dir, '*.jpg'))

    os.makedirs(args.result_dir, exist_ok=True)

    # Inference Model 준비
    # Choose to use a config
    configs = './custom_pedestrian.py'
    # Setup a checkpoint file to load
    checkpoint = 'D:/Side/2024_Sejoong_Jaywalking/backup/detection/rtmdet_l_pedestrian/label-2-ep300/best_coco_bbox_mAP_epoch_233.pth'
    # Set the device to be used for evaluation
    device = 'cuda:0'

    # Use the detector to do inference
    inferencer = DetInferencer(configs, checkpoint, device)

    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        inference_result = inferencer(file_path, no_save_vis=True, pred_score_thr=0.3)

        predict_result = inference_result['predictions'][0]

        bboxes = predict_result['bboxes']
        labels = predict_result['labels']
        scores = predict_result['scores']

        thr = 0.3

        bbox_label_score = zip(bboxes, labels, scores)

        result_txt = open(os.path.join(args.result_dir, f'{file_name}.txt'), 'w')
        if len(labels) == 0:
            result_txt.close()
        else:
            print(f'[ {file_name} ]')
            for idx, (bbox, label, score) in enumerate(bbox_label_score):
                if score < 0.3:
                    continue
                result_txt.write(f'{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')

            result_txt.close()
