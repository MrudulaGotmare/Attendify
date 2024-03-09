from PyQt5.QtCore import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date
import cv2, os, numpy
from PIL import Image

ui,_=loadUiType('face-reco.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0) #The default screen would be the login screen whose index is 0
        self.LOGINBUTTON.clicked.connect(self.login)
        self.LOGOUTBUTTON.clicked.connect(self.logout)
        self.CLOSEBUTTON.clicked.connect(self.close_window)
        self.TRAINLINK1.clicked.connect(self.training_form)
        self.ATTLINK1.clicked.connect(self.attendance_entry_form)
        self.REPORTSLINK1.clicked.connect(self.reports_form)
        self.TRAININGBACK.clicked.connect(self.show_mainform)
        self.ATTENDANCEBACK.clicked.connect(self.show_mainform)
        self.REPORTSBACK.clicked.connect(self.show_mainform)
        self.TRAININGBUTTON.clicked.connect(self.start_training)
        self.RECORD.clicked.connect(self.record_attendance)
        try:
            con = sqlite3.connect("face-reco.db")
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER, name TEXT, attendancedate TEXT)")
            con.commit()
            print("Table created successfully")
        except:
            print("Error in database")

    ### LOGIN PROCESS ###    
    def login(self):
        pw = self.PASSWORD.text()
        if(pw=="123"):
            self.PASSWORD.setText("")
            self.LOGININFO.setText("")
            self.tabWidget.setCurrentIndex(1)
        else:
            self.LOGININFO.setText("Invalid Password..")
            self.PASSWORD.setText("")

    ### LOGOUT PROCESS ###    
    def logout(self):
        self.tabWidget.setCurrentIndex(0)

    ### SHOW TRAINING FORM ###    
    def training_form(self):
        self.tabWidget.setCurrentIndex(2)

    ### SHOW ATTENDANCE ENTRY FORM ###    
    def attendance_entry_form(self):
        self.tabWidget.setCurrentIndex(3)

    ### SHOW REPORT FORM ###    
    def reports_form(self):
        self.tabWidget.setCurrentIndex(4)
        # self.REPORTS.setText("Attendance Link")
        self.REPORTS.setText("<a href='https://docs.google.com/spreadsheets/d/1pBjMwm54A0BXD0xpI5uzekrTWEbbcJAxdVt221Ypvyg/edit#gid=0'>Attendance Link</a>")
    # Open the hyperlink in a web browser when clicked
        self.REPORTS.setOpenExternalLinks(True)

    ### SHOW MAIN FORM ###    
    def show_mainform(self):
        self.tabWidget.setCurrentIndex(1)

    ### CLOSE WINDOW PROCESS ###    
    def close_window(self):
        self.close()

    ### TRAINING PROCESS ###
    def start_training(self):
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = self.traineeName.text()
        path = os.path.join(datasets,sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            print("The new directory is created")
            (width,height) = (130,100)
            face_cascade = cv2.CascadeClassifier(haar_file)
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < int(self.trainingCount.text()) +1:
                print(count)
                (_,im) = webcam.read()
                ### converting image to grayscale ###
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,4)
                ### x,y is the image starting position ###
                for (x,y,w,h) in faces:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                    face = gray[y:y+h,x:x+w]
                    face_resize = cv2.resize(face,(width,height))
                    cv2.imwrite('%s/%s.png'%(path,count),face_resize)
                count += 1
                cv2.imshow('OpenCV',im)
                key = cv2.waitKey(10)
                if key == 27:
                    break
            webcam.release()
            cv2.destroyAllWindows()
            path=""
            QMessageBox.information(self,"Attendance System","Training completed Successfully")
            self.traineeName.setText("")
            self.trainingCount.setText("100")
    
    ### RECORD ATTENDANCE ###    
    def record_attendance(self):
        self.currentprocess.setText("Process started.. waiting..")
        haar_file = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)
        datasets = 'datasets'
        (images,labels,names,id) = ([],[],{},0)
        for(subdirs,dirs,files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets,subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + "/" + filename
                    label = id
                    images.append(cv2.imread(path,0))
                    labels.append(int(label))
                id += 1
        (images,labels) = [numpy.array(lis) for lis in [images,labels]]
        print(images,labels)
        (width, height) = (130,100)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images,labels)
        webcam = cv2.VideoCapture(0)
        cnt=0
        while True:
            (_,im) = webcam.read()
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
                face = gray[y:y+h,x:x+w]
                face_resize = cv2.resize(face,(width,height))
                prediction = model.predict(face_resize)
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
                if(prediction[1]<800):
                    cv2.putText(im,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                    print(names[prediction[0]])
                    self.currentprocess.setText("Dectected face" + names[prediction[0]])
                    attendanceid = 0
                    available = False
                    try:
                        connection = sqlite3.connect("face-reco.db")
                        cursor = connection.execute("SELECT MAX(attendanceid) from attendance")
                        result = cursor.fetchall()
                        if result:
                            for maxid in result:
                                attendanceid = int(maxid[0]+1)
                    except:
                        attendanceid = 1
                    print(attendanceid)

                    try:
                        con = sqlite3.connect("face-reco.db")
                        cursor = con.execute("SELECT * FROM attendance WHERE name='" + str(names[prediction[0]]) + "' and attendancedate = '" + str(date.today()) + "'")
                        result = cursor.fetchall()
                        if result:
                            available=True
                        if(available==False):
                            con.execute("INSERT INTO attendance VALUES("+ str(attendanceid) +",'"+ str(names[prediction[0]]) +"','"+ str(date.today()) +"') ")
                            con.commit()
                    except:
                        print("ERROR IN DATABASE INSERT")
                    print("Attendance Registered successfully")
                    self.currentprocess.setText("Attendance entered for " + names[prediction[0]])
                    cnt=0
                else:
                    cnt+=1
                    cv2.putText(im,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
                    if(cnt>100):
                        print("Unknown person")
                        self.currentprocess.setText("Unknown Person")
                        cv2.imwrite('unknown.jpg',im)
                        cnt=0
            cv2.imshow("Face Recognition",im)
            key = cv2.waitKey(10)
            if key==27:
                break
        webcam.release()
        cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()