# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np


def eye_aspect_ratio(eye):
	# 计算眼睛的两组垂直关键点之间的欧式距离
	A = dist.euclidean(eye[1], eye[5])		# 1,5是一组垂直关键点
	B = dist.euclidean(eye[2], eye[4])		# 2,4是一组
	# 计算眼睛的一组水平关键点之间的欧式距离
	C = dist.euclidean(eye[0], eye[3])		# 0,3是一组水平关键点

	# 计算眼睛纵横比
	ear = (A + B) / (2.0 * C)
	# 返回眼睛纵横比
	return ear


def mouth_aspect_ratio(mouth):
	# 默认二范数：求特征值，然后求最大特征值得算术平方根
	A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59（人脸68个关键点）
	B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
	C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
	mar = (A + B) / (2.0 * C)

	return mar


def nose_jaw_distance(nose, jaw):
	# 计算鼻子上一点"27"到左右脸边界的欧式距离
	face_left1 = dist.euclidean(nose[0], jaw[0])		# 27, 0
	face_right1 = dist.euclidean(nose[0], jaw[16])		# 27, 16
	# 计算鼻子上一点"30"到左右脸边界的欧式距离
	face_left2 = dist.euclidean(nose[3], jaw[2])  		# 30, 2
	face_right2 = dist.euclidean(nose[3], jaw[14])  	# 30, 14
	# 创建元组，用以保存4个欧式距离值
	face_distance = (face_left1, face_right1, face_left2, face_right2)

	return face_distance


def eyebrow_jaw_distance(leftEyebrow, jaw):
	# 计算左眉毛上一点"24"到左右脸边界的欧式距离（镜像对称）
	eyebrow_left = dist.euclidean(leftEyebrow[2], jaw[0])		# 24, 0
	eyebrow_right = dist.euclidean(leftEyebrow[2], jaw[16])  	# 24, 16
	# 计算左右脸边界之间的欧式距离
	left_right = dist.euclidean(jaw[0], jaw[16])  			# 0, 16
	# 创建元组，用以保存3个欧式距离值
	eyebrow_distance = (eyebrow_left, eyebrow_right, left_right)

	return eyebrow_distance


# construct the argument parse and parse the arguments
# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
				help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
				help="path to input video file")
ap.add_argument("-t", "--threshold", type=float, default=0.27,
				help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type=int, default=2,
				help="the number of consecutive frames the eye must be below the threshold")


def main():

	args = vars(ap.parse_args())
	EYE_AR_THRESH = args['threshold']			# 眨眼阈值
	EYE_AR_CONSEC_FRAMES = args['frames']		# 闭眼次数阈值
	# 张嘴阈值
	MAR_THRESH = 0.5

	# 初始化眨眼帧计数器和总眨眼次数
	COUNTER_EYE = 0
	TOTAL_EYE = 0
	# 初始化张嘴帧计数器和总张嘴次数
	COUNTER_MOUTH = 0
	TOTAL_MOUTH = 0
	# 初始化摇头帧计数器和摇头次数
	distance_left = 0
	distance_right = 0
	TOTAL_FACE = 0
	# 初始化点头帧计数器和点头次数
	nod_flag = 0
	TOTAL_NOD = 0

	# 初始化dlib的人脸检测器（基于HOG），然后创建面部界标预测器
	print("[Prepare000] 加载面部界标预测器...")
	# 表示脸部位置检测器
	detector = dlib.get_frontal_face_detector()
	# 表示脸部特征位置检测器
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# 左右眼的索引
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	# 嘴唇的索引
	(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
	# 鼻子的索引
	(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
	# 下巴的索引
	(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
	# 左眉毛的索引
	(Eyebrow_Start, Eyebrow_End) = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']

	# start the video stream thread
	# 启动视频流线程
	print("[Prepare111] 启动视频流线程...")
	print("[Prompt information] 按Q键退出...")
	if args['video'] == "camera":
		vs = VideoStream(src=0).start()
		fileStream = False
	else:
		vs = FileVideoStream(args["video"]).start()
		fileStream = True

	time.sleep(1.0)

	# loop over frames from the video stream
	# 循环播放视频流中的帧
	while True:

		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		if fileStream and not vs.more():
			break

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=600)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# 在灰度框中检测人脸
		rects = detector(gray, 0)

		# 循环人脸检测
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# 提取左眼和右眼坐标，然后使用该坐标计算两只眼睛的眼睛纵横比
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			# 提取嘴唇坐标，然后使用该坐标计算嘴唇纵横比
			Mouth = shape[mStart:mEnd]
			mouthMAR = mouth_aspect_ratio(Mouth)
			# 提取鼻子和下巴的坐标，然后使用该坐标计算鼻子到左右脸边界的欧式距离
			nose = shape[nStart:nEnd]
			jaw = shape[jStart:jEnd]
			NOSE_JAW_Distance = nose_jaw_distance(nose, jaw)
			# 提取左眉毛的坐标，然后使用该坐标计算左眉毛到左右脸边界的欧式距离
			leftEyebrow = shape[Eyebrow_Start:Eyebrow_End]
			Eyebrow_JAW_Distance = eyebrow_jaw_distance(leftEyebrow, jaw)

			# 对左右两只眼睛的纵横比取平均值
			ear = (leftEAR + rightEAR) / 2.0
			# 移植嘴唇纵横比
			mar = mouthMAR
			# 移植鼻子到左右脸边界的欧式距离
			face_left1 = NOSE_JAW_Distance[0]
			face_right1 = NOSE_JAW_Distance[1]
			face_left2 = NOSE_JAW_Distance[2]
			face_right2 = NOSE_JAW_Distance[3]
			# 移植左眉毛到左右脸边界的欧式距离，及左右脸边界之间的欧式距离
			eyebrow_left = Eyebrow_JAW_Distance[0]
			eyebrow_right = Eyebrow_JAW_Distance[1]
			left_right = Eyebrow_JAW_Distance[2]

			# 可视化两只眼睛
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			# 可视化嘴唇
			mouth_hull = cv2.convexHull(Mouth)
			cv2.drawContours(frame, [mouth_hull], -1, (255, 0, 0), 1)
			# 可视化鼻子和脸边界
			noseHull = cv2.convexHull(nose)
			cv2.drawContours(frame, [noseHull], -1, (0, 0, 255), 1)
			jawHull = cv2.convexHull(jaw)
			cv2.drawContours(frame, [jawHull], -1, (0, 0, 255), 1)
			# # 可视化左眉毛
			# leftEyebrowHull = cv2.convexHull(leftEyebrow)
			# cv2.drawContours(frame, [leftEyebrowHull], -1, (0, 255, 0), 1)

			# 判断眼睛纵横比是否低于眨眼阈值，如果是，则增加眨眼帧计数器
			if ear < EYE_AR_THRESH:
				COUNTER_EYE += 1
			# 否则，眼睛的纵横比不低于眨眼阈值
			else:
				# 如果闭上眼睛的次数足够多，则增加眨眼的总次数
				if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
					TOTAL_EYE += 1
				# reset the eye frame counter
				COUNTER_EYE = 0

			# 判断嘴唇纵横比是否高于张嘴阈值，如果是，则增加张嘴帧计数器
			if mar > MAR_THRESH:
				COUNTER_MOUTH += 1
			# 否则，嘴唇的纵横比低于或等于张嘴阈值
			else:
				# 如果张嘴帧计数器不等于0，则增加张嘴的总次数
				if COUNTER_MOUTH != 0:
					TOTAL_MOUTH += 1
					COUNTER_MOUTH = 0

			# 根据鼻子到左右脸边界的欧式距离，判断是否摇头
			# 左脸大于右脸
			if face_left1 >= face_right1+2 and face_left2 >= face_right2+2:
				distance_left += 1
			# 右脸大于左脸
			if face_right1 >= face_left1+2 and face_right2 >= face_left2+2:
				distance_right += 1
			# 左脸大于右脸，并且右脸大于左脸，判定摇头
			if distance_left != 0 and distance_right != 0:
				TOTAL_FACE += 1
				distance_right = 0
				distance_left = 0

			# 两边之和是否小于或等于第三边+阈值，来判断是否点头
			# 根据左眉毛到左右脸边界的欧式距离与左右脸边界之间的欧式距离作比较，判断是否点头
			if eyebrow_left+eyebrow_right <= left_right+3:
				nod_flag += 1
			if nod_flag != 0 and eyebrow_left+eyebrow_right >= left_right+3:
				TOTAL_NOD += 1
				nod_flag = 0

			# 画出画框上眨眼的总次数以及计算出的帧的眼睛纵横比
			cv2.putText(frame, "Blinks: {}".format(TOTAL_EYE), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# 画出张嘴的总次数以及计算出的帧的嘴唇纵横比
			cv2.putText(frame, "Mouth is open: {}".format(TOTAL_MOUTH), (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# 画出摇头次数
			cv2.putText(frame, "shake one's head: {}".format(TOTAL_FACE), (10, 90),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# 画出点头次数
			cv2.putText(frame, "nod: {}".format(TOTAL_NOD), (10, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			# 活体检测
			cv2.putText(frame, "Live detection: wink(5)", (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			if TOTAL_EYE >= 5:	# 眨眼五次
				cv2.putText(frame, "open your mouth(3)", (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			if TOTAL_MOUTH >= 3:	# 张嘴三次
				cv2.putText(frame, "shake your head(2)", (300, 90),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			if TOTAL_FACE >= 2:	# 摇头两次
				cv2.putText(frame, "nod(2)", (300, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			if TOTAL_NOD >= 2:	# 点头两次
				cv2.putText(frame, "Live detection: done", (300, 150),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()


if __name__ == '__main__':
	main()
