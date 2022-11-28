from flask import Flask, render_template, Response
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours


import numpy as np

import imutils

app = Flask(__name__)

#def midpoint(ptA, ptB):
    #return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def gen_frames(): 
    camera = cv2.VideoCapture(0)  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            
            while (camera.read()):
                ref,fr = camera.read()
            
                frame = cv2.resize(fr, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
                orig = frame[:1080,0:1920]
                
                def midpoint(ptA, ptB):
                    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (15, 15), 0)
                thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                kernel = np.ones((3,3),np.uint8)
                closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=3)

                result_img = closing.copy()
                contours,hierachy = cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                hitung_objek = 0

                pixelsPerMetric = None
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 1000 or area > 120000:
                        continue

                    orig = frame.copy()
                    box = cv2.minAreaRect(cnt)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    box = perspective.order_points(box)
                    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)

                    cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
                    cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
                    cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
                    cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

                    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
                    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

                    lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                    if pixelsPerMetric is None:
                        pixelsPerMetric = lebar_pixel
                        pixelsPerMetric = panjang_pixel

                    lebar = lebar_pixel
                    panjang = panjang_pixel

                    cv2.putText(orig, "L: {:.1f}CM".format(lebar_pixel/25.5),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
                    cv2.putText(orig, "B: {:.1f}CM".format(panjang_pixel/25.5),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)

                    hitung_objek+=1
                cv2.putText(orig, "OBJECTS: {}".format(hitung_objek),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  

                #ret, jpeg = cv2.imencode('.jpg', frame)
                #return jpeg.tobytes()        
                ret, buffer = cv2.imencode('.jpg', orig)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result'''

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
