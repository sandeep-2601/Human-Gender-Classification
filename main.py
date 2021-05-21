from flask import render_template,Flask
from flask import request,redirect,url_for
import os
import cv2
import datetime
from app import util
from PIL import Image
app=Flask(__name__)

UPLOAD_FLODER = 'static/uploads'

def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/faceapp')
def faceapp():
    return render_template('faceapp.html')

@app.route('/live-detection')
def livedetect():
    # define a video capture object 
    cam = cv2.VideoCapture(0) 
    #cv2.namedWindow("capture your picture")
    img_counter=0
    while(True): 
         
        rt, frame = cam.read()
        frame=util.pipeline_model(frame,'bgr')
        cv2.imshow('videoframe',frame) 

        # the 'Esc' button is set as quit
        k=cv2.waitKey(1)
        if k%256==27:
            print("escape")
            break
        elif k%256==32:
            #y=datetime.datetime.now()
            #x=y.strftime("%D")+y.strftime("%T")    	
            #scrop_frame=frame[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
            file='C:/project/Image/'+ str(img_counter) +'.jpg'
            cv2.imwrite(file,frame)
            img_counter+=1

    # After the loop release the cap object 
    cam.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    return redirect(url_for('faceapp'))

@app.route('/video-demo')
def videodemo():
    cam = cv2.VideoCapture(r'C:\project\16B-PROJECT\Data\video.mp4') 
    while(True): 
        rt, frame = cam.read()
        frame=util.pipeline_model(frame,'bgr')

        cv2.imshow('videoframe',frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
  
        
    cam.release() 
    cv2.destroyAllWindows() 
    return redirect(url_for('faceapp'))

        



@app.route('/gender',methods=['POST','GET'])
def gender():
    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(UPLOAD_FLODER,filename)
        f.save(path)
        w = getwidth(path)
        # prediction (pass to pipeline model)
        img=cv2.imread(path)
        util.pipeline_model(img,'bgr',filename)


        return render_template('gender.html',fileupload=True,img_name=filename, w=w)


    return render_template('gender.html',fileupload=False,img_name="freeai.png")

if __name__ == '__main__':
    app.run(debug=True)

