import cv2

cam = cv2.VideoCapture(0)
faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam.set(3,480)
cam.set(4,640)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list   = ["(0-5)","(6-10)","(11-15)","(15-19)","(20-25)",
            "(26-30)","(31-36)","(37-41)","(42-46)","(47-52)",
            "(53-60)","(60-65)","(66-71)","(72-80)","(80,90)"]
gender_list = ["Male","Famele"]

def load_caffe_model():
    age_net    = cv2.dnn.readNetFromCaffe("deploy_age.prototxt",
                                       "age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt",
                                          "gender_net.caffemodel")
    return age_net,gender_net

def Video(age_net,gender_net) :

    while True :
        ret,frame = cam.read()
        gray      = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = faceCasc.detectMultiScale(gray,1.1,10,cv2.CASCADE_SCALE_IMAGE,(30,30))

        for (x,y,w,h) in face :
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

            face_photo = frame[y:y+h,x:x+w].copy()
            blob       = cv2.dnn.blobFromImage(face_photo,1.0,(244,244),
                                               MODEL_MEAN_VALUES,swapRB=True)
            gender_net.setInput(blob)
            gender_predict = gender_net.forward()
            gender         = gender_list[gender_predict[0].argmax()]

            age_net.setInput(blob)
            age_predict = age_net.forward()
            age         = age_list[age_predict[0].argmax()]

            text = "%s%s" %(gender,age)
            cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255,255,0),1,cv2.LINE_AA)

            cv2.imshow("FRAME",frame)

        if cv2.waitKey(27) & 0xFF == ord("q") :
            break

def main() :
    age_net,gender_net = load_caffe_model()
    Video(age_net,gender_net)

if __name__ == "__main__" :
    main()


cam.release()
cv2.destroyAllWindows()