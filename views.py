from authwebapp import webStream
from django.core.files import File
#from authwebapp.webStream import generate
from django.http import request
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from flask import app
from flask.templating import render_template
from .forms import UserRegistration, UserEditForm
from django.template import RequestContext
import cv2
import time


# Create your views here.

@login_required
def dashboard(request):
    context = {
        "welcome": "Welcome to your dashboard"
    }
    return render(request, 'authwebapp/dashboard.html', context=context)


def register(request):
    if request.method == 'POST':
        form = UserRegistration(request.POST or None)
        if form.is_valid():
            new_user = form.save(commit=False)
            new_user.set_password(
                form.cleaned_data.get('password')
            )
            new_user.save()
            return render(request, 'authwebapp/register_done.html')
    else:
        form = UserRegistration()

    context = {
        "form": form
    }

    return render(request, 'authwebapp/register.html', context=context)


@login_required
def edit(request):
    if request.method == 'POST':
        user_form = UserEditForm(instance=request.user,
                                 data=request.POST)
        if user_form.is_valid():
            user_form.save()
    else:
        user_form = UserEditForm(instance=request.user)
    context = {
        'form': user_form,
    }
    return render(request, 'authwebapp/edit.html', context=context)
'''
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading

class Videoself:
    cam = request.File['webStream']
    video = cv2.VideoCapture(1)
    

    while(video.isOpened()):
        ret, frame = cam.read()
      
        if not ret:
            break

        do_something()

        k = cv2.waitKey(1)
        if k == 27:
            break

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(cam):
    while True:
        frame = cam.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def livefe(request):
    try:
        cam = Videoself.video = cv2.VideoCapture()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass
    context = {
        "Welcome"
    }

    return render_template(object, 'authwebapp/index.html', context=context)



@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
    return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
mimetype = ("multipart/x-mixed-replace; boundary=frame")

'''