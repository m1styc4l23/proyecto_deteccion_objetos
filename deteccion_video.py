from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
import time
import paho.mqtt.client
from datetime import datetime

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def on_connect(client, userdata, flags, rc):
    print('connected (%s)' % client._client_id)
    client.subscribe(topic='yolo/conteo', qos=2)
    
def on_message(client, userdata, message):
    print('_______________________')
    print('topic: %s' % message.topic)
    print('payload: %s' % message.payload)
    print('qos: %d' % message.qos)

def Client_MQTT():
    client = paho.mqtt.client.Client(client_id='yolo-subs', clean_session=False)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host='127.0.0.1', port=1884)
    client.loop_forever()
    
def mqtt_connect():
    client = paho.mqtt.client.Client(client_id='yolo-subs', clean_session=False)
    client.connect(host='127.0.0.1', port=1884)
    client.loop_start()
    

if __name__ == "__main__":
    
    client = paho.mqtt.client.Client(client_id='yolo-pubs', clean_session=False)
    client.connect(host='127.0.0.1', port=1884)
    client.loop_start()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=1,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,960))
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,960))
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]
    
    
    #new_frame_time = time.time()
    #x = 1 # displays the frame rate every 1 second
    #counter = 0
    
    #variable contador con la cual se valida que si cambia el conteo realiza una publicación en mqtt
    person_counter_mqtt = 0
    
    while cap:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)
        #LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))


        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        #fraps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        #counter+=1
        #if (time.time() - new_frame_time) > 1 :
        #    print("FPS: ", counter / (time.time() - new_frame_time))
        #    counter = 0
        #    new_frame_time = time.time()

        #print("FPS: ", 1.0 / (time.time() - new_frame_time)) 

        texto_estado = "Conteo"
        person_counter = 0

        for detection in detections:
            if detection is not None:
                puntos = []
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    #print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                    cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                    cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase
                    person_counter += 1
                    puntos.append([x1,y1,box_w,box_h])
                    
                if len(puntos) == 2:
                    x1, y1, w1, h1 = puntos[0]
                    x2, y2, w2, h2 = puntos[1]
                    
                    if x1 < x2:
                        distancia_pixeles = abs(x2 - (x1+w1))
                        distancia_cm = (distancia_pixeles*240)/1280
                        cv2.putText(frame, "{:.2f} cm".format(distancia_cm), (x1+w1+distancia_pixeles//2, y1-30), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
                        cv2.line(frame,(x1+w1,y1-20), (x2,y1-20), (0,0,255), 2)
                        cv2.line(frame,(x1+w1,y1-30), (x1+w1,y1-10), (0,0,255), 2)
                        cv2.line(frame,(x2,y1-30), (x2,y1-10), (0,0,255), 2)
                    else:
                        distancia_pixeles = abs(x1 - (x2+w2))
                        distancia_cm = (distancia_pixeles*240)/1280
                        cv2.putText(frame, "{:.2f} cm".format(distancia_cm), (x2+w2+distancia_pixeles//2, y2-30), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
                        cv2.line(frame,(x2+w2,y2-20), (x1,y2-20), (0,0,255), 2)
                        cv2.line(frame,(x2+w2,y2-30), (x2+w2,y2-10), (0,0,255), 2)
                        cv2.line(frame,(x1,y2-30), (x1,y2-10), (0,0,255), 2)
        
        #Imprimir fps
        cv2.putText(frame, 'FPS: ' + fps, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        
        #Imprimir Conteo
        #cv2.rectangle(frame, (0,0), (frame.shape[1],40), (0,0,0), -1)
        color = (0, 255, 0)
        texto_estado = 'Conteo: ' + str(person_counter)
        cv2.putText(frame, texto_estado, (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        #Publicar Conteo Mqtt
        if person_counter_mqtt != person_counter :
            person_counter_mqtt = person_counter
            client.publish("yolo/conteo", '{' + str(person_counter_mqtt) + ', ' + str(datetime.today()) + '}')
        
        #Convertimos de vuelta a BGR para que cv2 pueda desplegarlo en los colores correctos
        if opt.webcam==1:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
            out.write(RGBimg)
        else:
            out.write(Convertir_BGR(RGBimg))
            cv2.imshow('frame', RGBimg)
        #cv2.waitKey(0)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
