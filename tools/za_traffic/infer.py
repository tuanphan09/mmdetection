from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv


config_file = 'configs/retinanet/traffic_sign/retinanet_r50_fpn_1x_traffic_sign.py'

checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x_traffic_sign/latest.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '/data2/zalo-ai-2020/za_traffic_2020/data/traffic_public_test/images/9921.png'
result = inference_detector(model, img)

print(len(result))
model.show_result(img, result, score_thr=0.3, show=False, out_file="tmp/result.png")


