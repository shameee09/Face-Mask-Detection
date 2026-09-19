import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Face Mask Compliance Screening",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# PROFESSIONAL UI STYLE
# ==========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1220;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        color: #e2e8f0 !important;
    }

    p {
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    [data-testid="stCameraInput"] {
        border-radius: 10px;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 12px;
        border-top: 1px solid #263244;
        padding-top: 18px;
        margin-top: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

header_left, header_right = st.columns([5, 1])

with header_left:

    st.caption(
        "COMPLIANCE SCREENING SYSTEM"
    )

    st.title(
        "Face Mask Compliance Screening"
    )

    st.write(
        "Camera-based and image-upload screening for face-mask compliance."
    )


with header_right:

    st.success(
        "SYSTEM READY"
    )


st.divider()


# ==========================================================
# SCREENING INFORMATION
# ==========================================================

st.subheader(
    "Entry Compliance Screening"
)

st.write(
    "Capture an image using your camera or upload a photo "
    "for face-mask compliance assessment."
)

st.info(
    "The selected image is processed for this assessment "
    "and is not stored by this application."
)


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_detection_models():

    prototxt_path = (
        "face_detector/deploy.prototxt"
    )

    weights_path = (
        "face_detector/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    model_path = "mask_detector.h5"

    face_net = cv2.dnn.readNet(
        prototxt_path,
        weights_path
    )

    mask_net = load_model(
        model_path
    )

    return face_net, mask_net


try:

    face_net, mask_net = load_detection_models()

except Exception as error:

    st.error(
        "The screening service could not be initialized."
    )

    with st.expander(
        "Technical details"
    ):

        st.code(
            str(error)
        )

    st.stop()


# ==========================================================
# IMAGE INPUT
# ==========================================================

st.subheader(
    "Select Image Source"
)

st.caption(
    "Choose either camera capture or upload a photo."
)


# Two columns for both options

camera_col, upload_col = st.columns(2)


# ==========================================================
# CAMERA OPTION
# ==========================================================

with camera_col:

    st.markdown(
        "### 📷 Camera"
    )

    camera_image = st.camera_input(
        "Capture screening image"
    )


# ==========================================================
# UPLOAD OPTION
# ==========================================================

with upload_col:

    st.markdown(
        "### 📁 Upload Photo"
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        help="Upload a JPG, JPEG or PNG image."
    )


# ==========================================================
# DETERMINE SELECTED IMAGE
# ==========================================================

selected_image = None

if camera_image is not None:

    selected_image = camera_image

elif uploaded_image is not None:

    selected_image = uploaded_image


# ==========================================================
# PROCESS IMAGE
# ==========================================================

if selected_image is not None:

    # ------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------

    image_bytes = selected_image.getvalue()

    image_array = np.asarray(
        bytearray(image_bytes),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # ------------------------------------------------------
    # CHECK IMAGE
    # ------------------------------------------------------

    if image is None:

        st.error(
            "The selected image could not be processed."
        )

        st.stop()


    # ------------------------------------------------------
    # IMAGE DIMENSIONS
    # ------------------------------------------------------

    height, width = image.shape[:2]


    # ======================================================
    # FACE DETECTION
    # ======================================================

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(
        blob
    )

    detections = face_net.forward()


    # ======================================================
    # RESULT COUNTERS
    # ======================================================

    face_count = 0

    compliant_count = 0

    non_compliant_count = 0

    results = []


    # ======================================================
    # PROCESS EACH FACE
    # ======================================================

    for i in range(
        detections.shape[2]
    ):

        # --------------------------------------------------
        # FACE CONFIDENCE
        # --------------------------------------------------

        detection_confidence = float(
            detections[
                0,
                0,
                i,
                2
            ]
        )


        if detection_confidence < 0.5:

            continue


        face_count += 1


        # --------------------------------------------------
        # BOUNDING BOX
        # --------------------------------------------------

        box = (
            detections[
                0,
                0,
                i,
                3:7
            ]
            *
            np.array(
                [
                    width,
                    height,
                    width,
                    height
                ]
            )
        )


        start_x, start_y, end_x, end_y = (
            box.astype("int")
        )


        # --------------------------------------------------
        # KEEP COORDINATES INSIDE IMAGE
        # --------------------------------------------------

        start_x = max(
            0,
            start_x
        )

        start_y = max(
            0,
            start_y
        )

        end_x = min(
            width,
            end_x
        )

        end_y = min(
            height,
            end_y
        )


        # --------------------------------------------------
        # EXTRACT FACE
        # --------------------------------------------------

        face = image[
            start_y:end_y,
            start_x:end_x
        ]


        if face.size == 0:

            continue


        # --------------------------------------------------
        # PREPROCESS FACE
        # --------------------------------------------------

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = cv2.resize(
            face,
            (224, 224)
        )

        face = (
            face.astype("float32")
            / 255.0
        )

        face = np.expand_dims(
            face,
            axis=0
        )


        # ==================================================
        # MASK PREDICTION
        # ==================================================

        prediction = mask_net.predict(
            face,
            verbose=0
        )[0]


        mask_probability = float(
            prediction[0]
        )

        no_mask_probability = float(
            prediction[1]
        )


        # ==================================================
        # CLASSIFICATION
        # ==================================================

        if mask_probability > no_mask_probability:

            status = "COMPLIANT"

            label = "MASK"

            confidence_score = (
                mask_probability * 100
            )

            draw_color = (
                0,
                200,
                80
            )

            compliant_count += 1

        else:

            status = "NON-COMPLIANT"

            label = "NO MASK"

            confidence_score = (
                no_mask_probability * 100
            )

            draw_color = (
                0,
                0,
                255
            )

            non_compliant_count += 1


        # --------------------------------------------------
        # STORE RESULT
        # --------------------------------------------------

        results.append(
            {
                "status": status,
                "confidence": confidence_score
            }
        )


        # ==================================================
        # DRAW FACE BOX
        # ==================================================

        display_text = (
            f"{label} | "
            f"{confidence_score:.1f}%"
        )


        cv2.rectangle(
            image,
            (
                start_x,
                start_y
            ),
            (
                end_x,
                end_y
            ),
            draw_color,
            3
        )


        cv2.putText(
            image,
            display_text,
            (
                start_x,
                max(
                    start_y - 10,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            draw_color,
            2
        )


    # ======================================================
    # SCREENING RESULT
    # ======================================================

    st.divider()

    st.subheader(
        "Screening Result"
    )


    # ------------------------------------------------------
    # CONVERT IMAGE
    # ------------------------------------------------------

    result_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    st.image(
        result_image,
        caption="Face-mask compliance screening result",
        use_container_width=True
    )


    # ======================================================
    # ASSESSMENT SUMMARY
    # ======================================================

    st.subheader(
        "Assessment Summary"
    )


    if results:

        highest_confidence = max(
            result["confidence"]
            for result in results
        )

    else:

        highest_confidence = 0


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    with metric1:

        st.metric(
            "Faces Detected",
            face_count
        )


    with metric2:

        st.metric(
            "Compliant",
            compliant_count
        )


    with metric3:

        st.metric(
            "Non-Compliant",
            non_compliant_count
        )


    with metric4:

        st.metric(
            "Confidence",
            f"{highest_confidence:.1f}%"
        )


    # ======================================================
    # SCREENING DECISION
    # ======================================================

    st.subheader(
        "Screening Decision"
    )


    # ------------------------------------------------------
    # NO FACE
    # ------------------------------------------------------

    if face_count == 0:

        st.warning(
            "NO FACE DETECTED"
        )

        st.write(
            "Please upload or capture another image "
            "where the face is clearly visible."
        )


    # ------------------------------------------------------
    # NON-COMPLIANT
    # ------------------------------------------------------

    elif non_compliant_count > 0:

        st.error(
            "REVIEW REQUIRED"
        )

        st.write(
            f"{non_compliant_count} face(s) "
            "without a detected mask were identified."
        )

        st.write(
            "The subject should follow the applicable "
            "face-mask requirement before proceeding."
        )


    # ------------------------------------------------------
    # COMPLIANT
    # ------------------------------------------------------

    else:

        st.success(
            "SCREENING PASSED"
        )

        st.write(
            f"A face mask was detected for all "
            f"{compliant_count} identified face(s)."
        )


    # ======================================================
    # SCREENING INFORMATION
    # ======================================================

    st.subheader(
        "Screening Information"
    )


    info_left, info_right = (
        st.columns(2)
    )


    with info_left:

        st.markdown(
            "**Purpose**"
        )

        st.write(
            "Face-mask compliance screening at "
            "controlled entry points and other "
            "environments where mask requirements apply."
        )


    with info_right:

        st.markdown(
            "**Assessment**"
        )

        st.write(
            "Each detected face is classified as "
            "compliant or non-compliant with an "
            "associated confidence score."
        )


# ==========================================================
# BEFORE IMAGE SELECTION
# ==========================================================

else:

    st.info(
        "Capture an image using the camera or "
        "upload a photo to begin the assessment."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        Face Mask Compliance Screening
        &nbsp;•&nbsp;
        Screening and Assessment System
    </div>
    """,
    unsafe_allow_html=True
)
