import cv2
import time
from flask import Flask, render_template, Response
'''
#Initialize Flask app
app = Flask(__name__)
'''
fitToEllipse = False
cap = cv2.VideoCapture(0)
time.sleep(3)
countFall = 0
k = 0
foregroundBackground = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    
    try:     #Converts every frame to the colour of gray scale and then the background is subtracted

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        foregroundMask = foregroundBackground.apply(gray)
        
        #Find contours
        contours, _ = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
        
            # list which stores all areas
            areas = []

            for contour in contours:

                ar = cv2.contourArea(contour)
                areas.append(ar)
            
            maximum_Area = max(areas, default = 0)

            maximum_Area_index = areas.index(maximum_Area)

            cntr = contours[maximum_Area_index]

            M = cv2.moments(cntr)
            
            x, y, w, h = cv2.boundingRect(cntr)

            cv2.drawContours(foregroundMask, [cntr], 0, (255,255,255), 3, maxLevel = 0)
            
            if h < w:

                k += 1
                
            if k > 10:

                print("FALL" + countFall + 1)

                cv2.putText(foregroundMask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            if w < h:
               
                k = 0 
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                
            cv2.imshow('video', frame)
        
            if cv2.waitKey(33) == 27:
             break
    except Exception as e:
        break
'''
def gen_frames():  

    while True:

        success, frame = cap.read()  # read the cap frame
        
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.kpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/kpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

cv2.destroyAllWindows()'''
##Def: app route for default page of webapp
#route refers to url pattern of an app
'''
@app.route('/')
def index():
    return render_template('index.html')
#Def app route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return countFall

#The ‘/video_feed’ route returns the streaming response.
#  Because this stream returns the images that are to be displayed 
# in the web page, the URL to this route is in the “src”
#  attribute of the image tag 
if __name__ == "__main__":
    app.run(debug=True)'''