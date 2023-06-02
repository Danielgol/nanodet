import os
import cv2
import torch
import argparse
from demo.demo import Predictor
from nanodet.util import cfg, load_config, Logger


def convert_to_yolo_v7(x1, y1, x2, y2, img_width, img_height):
    # Calcula as coordenadas do centro da bounding box
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    # Calcula a largura e a altura da bounding box em relação ao tamanho da imagem
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    # Normaliza as coordenadas do centro, largura e altura em relação ao tamanho da imagem
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    bbox_width_norm = bbox_width / img_width
    bbox_height_norm = bbox_height / img_height
    # Retorna as coordenadas no formato YOLO v7
    return x_center_norm, y_center_norm, bbox_width_norm, bbox_height_norm


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', type=str, help='config file path')
  parser.add_argument('--model', type=str, help='model file path')
  parser.add_argument('--source', type=str, default='png', help='images source')
  opt = parser.parse_args()
  
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="0"
  device = torch.device('cuda')
    
  os.mkdir("output")

  config_path = opt.cfg
  model_path = opt.model
  image_path = opt.source

  load_config(cfg, config_path)
  logger = Logger(-1, use_tensorboard=False)

  for name in os.listdir(image_path):

      if not name.endswith(".png"):
          continue

      path = image_path+name
      img = cv2.imread(path)

      predictor = Predictor(cfg, model_path, logger, device=device)
      meta, res = predictor.inference(path)

      label_file = "output/"+str(name.split(".")[0])+".txt"
      open(label_file, "w").close()

      for label in res[0]:
          for bbox in res[0][label]:

              x1 = int(bbox[0])
              y1 = int(bbox[1])
              x2 = int(bbox[2])
              y2 = int(bbox[3])
              conf = float(bbox[4])

              x_center, y_center, bb_width, bb_height = convert_to_yolo_v7(x1, y1, x2, y2, img.shape[1], img.shape[0])
              line = f"{label} {x_center:.6f} {y_center:.6f} {bb_width:.6f} {bb_height:.6f} {conf:.6f}\n"
              with open(label_file, "+a") as f:
                  f.write(line)
                  f.close()

              # if conf < 0.5:
              #     continue

              #img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)
              #cv2.imwrite("./output/"+name, img)
              #print(label, conf)
