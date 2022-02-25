import cv2
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy import asarray
from numpy import expand_dims
from PIL import Image
from scipy.spatial.distance import cosine
import streamlit as st
import tensorflow as tf
from tensorflow import keras

detector = MTCNN(min_face_size=80)
model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def is_match(known_embedding, candidate_embedding, thresh=0.4):
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        # print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        return True
    else:
        # print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        return False


def get_embeddings(pixels, person, bbox, required_size=(224, 224)):
    # faces=[extract_face(pixels)]
    h, w, c = pixels.shape
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    if x1 < 0:
        x1 = 0
    if x2 > w:
        x2 = w
    if y1 < 0:
        y1 = 0
    if y2 > h:
        y2 = h
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    samples = asarray([face_array], 'float32')
    samples = preprocess_input(samples, version=2)
    yhat = model.predict(samples)
    return yhat


st.title("face verification")
st.markdown(
    """
    <style>    
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>    
    """,
    unsafe_allow_html=True
)

st.sidebar.title("...")


# st.sidebar.subheader('upload an image')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return resized


image_file = st.sidebar.file_uploader("upload your image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    image = np.array(Image.open(image_file))
    st.sidebar.image(image)
    name = st.text_input('TYPE DOWN YOUR NAME', '')
    if name is not None:
        pixels = pyplot.imread(image_file)
        results = detector.detect_faces(pixels)
        embedding1 = get_embeddings(pixels, results[0], results[0]['box'])
        st.sidebar.markdown('start verification process by click on use webcam.')
        use_webcam = st.button('use webcam')
        stframe = st.empty()
        if use_webcam:
            cap = cv2.VideoCapture(-1)
            while True:
                ret, frame = cap.read()
                try:
                    result = detector.detect_faces(frame)

                    if result:
                        person = result[0]
                        bounding_box = person['box']
                        keypoints = person['keypoints']

                        filenames = person

                        embedding2 = get_embeddings(frame, person, bounding_box)

                        if is_match(embedding1, embedding2):
                            frame = cv2.rectangle(frame,
                                                  (bounding_box[0], bounding_box[1]),
                                                  (
                                                      bounding_box[0] + bounding_box[2],
                                                      bounding_box[1] + bounding_box[3]),
                                                  (0, 255, 0),
                                                  2)

                            frame = cv2.circle(frame, (keypoints['left_eye']), 2, (0, 255, 0), 2)
                            frame = cv2.circle(frame, (keypoints['right_eye']), 2, (0, 255, 0), 2)
                            frame = cv2.circle(frame, (keypoints['nose']), 2, (0, 255, 0), 2)
                            frame = cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 255, 0), 2)
                            frame = cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 255, 0), 2)
                            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 100), 2)

                        else:
                            frame = cv2.rectangle(frame,
                                                  (bounding_box[0], bounding_box[1]),
                                                  (
                                                      bounding_box[0] + bounding_box[2],
                                                      bounding_box[1] + bounding_box[3]),
                                                  (0, 0, 255), 2)

                            frame = cv2.circle(frame, (keypoints['left_eye']), 2, (0, 0, 255), 2)
                            frame = cv2.circle(frame, (keypoints['right_eye']), 2, (0, 0, 255), 2)
                            frame = cv2.circle(frame, (keypoints['nose']), 2, (0, 0, 255), 2)
                            frame = cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 0, 255), 2)
                            frame = cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 0, 255), 2)

                    # display resulting frame
                    stop = st.button('stop recording')
                    stframe.image(frame, channels='BGR')
                    if stop:
                        cap.release()
                        cv2.destroyAllWindows()
                except:
                    stframe.image(frame, channels='BGR')


            # When everything's done, release capture

                # cv2.destroyAllWindows()
else:
    st.sidebar.markdown('there is no uploaded file.')
    st.sidebar.markdown('please upload photo of yourself.')
