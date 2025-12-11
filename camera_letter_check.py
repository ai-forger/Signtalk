import os
import cv2
import numpy as np
from collections import defaultdict
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')

offset = 29


def distance(x, y):
	return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def draw_skeleton_from_pts(pts, w, h):
	white = np.ones((400, 400, 3), dtype=np.uint8) * 255
	x_offset = ((400 - w) // 2) - 15
	y_offset = ((400 - h) // 2) - 15
	for t in range(0, 4):
		cv2.line(white, (pts[t][0] + x_offset, pts[t][1] + y_offset), (pts[t + 1][0] + x_offset, pts[t + 1][1] + y_offset), (0, 255, 0), 3)
	for t in range(5, 8):
		cv2.line(white, (pts[t][0] + x_offset, pts[t][1] + y_offset), (pts[t + 1][0] + x_offset, pts[t + 1][1] + y_offset), (0, 255, 0), 3)
	for t in range(9, 12):
		cv2.line(white, (pts[t][0] + x_offset, pts[t][1] + y_offset), (pts[t + 1][0] + x_offset, pts[t + 1][1] + y_offset), (0, 255, 0), 3)
	for t in range(13, 16):
		cv2.line(white, (pts[t][0] + x_offset, pts[t][1] + y_offset), (pts[t + 1][0] + x_offset, pts[t + 1][1] + y_offset), (0, 255, 0), 3)
	for t in range(17, 20):
		cv2.line(white, (pts[t][0] + x_offset, pts[t][1] + y_offset), (pts[t + 1][0] + x_offset, pts[t + 1][1] + y_offset), (0, 255, 0), 3)
	cv2.line(white, (pts[5][0] + x_offset, pts[5][1] + y_offset), (pts[9][0] + x_offset, pts[9][1] + y_offset), (0, 255, 0), 3)
	cv2.line(white, (pts[9][0] + x_offset, pts[9][1] + y_offset), (pts[13][0] + x_offset, pts[13][1] + y_offset), (0, 255, 0), 3)
	cv2.line(white, (pts[13][0] + x_offset, pts[13][1] + y_offset), (pts[17][0] + x_offset, pts[17][1] + y_offset), (0, 255, 0), 3)
	cv2.line(white, (pts[0][0] + x_offset, pts[0][1] + y_offset), (pts[5][0] + x_offset, pts[5][1] + y_offset), (0, 255, 0), 3)
	cv2.line(white, (pts[0][0] + x_offset, pts[0][1] + y_offset), (pts[17][0] + x_offset, pts[17][1] + y_offset), (0, 255, 0), 3)
	for i in range(21):
		cv2.circle(white, (pts[i][0] + x_offset, pts[i][1] + y_offset), 2, (0, 0, 255), 1)
	return white


def predict_letter(model, pts):
	xs = [p[0] for p in pts]
	ys = [p[1] for p in pts]
	w = max(xs) - min(xs) + 1
	h = max(ys) - min(ys) + 1
	white = draw_skeleton_from_pts(pts, w, h)
	inp = white.reshape(1, 400, 400, 3)
	prob = np.array(model.predict(inp, verbose=0)[0], dtype='float32')
	ch1 = int(np.argmax(prob, axis=0))
	prob[ch1] = 0
	ch2 = int(np.argmax(prob, axis=0))
	prob[ch2] = 0
	# Reduced rule mapping (same as in app)
	pl = [ch1, ch2]
	l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
	     [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
	     [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
	if pl in l:
		if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
			ch1 = 0
	l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
	pl = [ch1, ch2]
	if pl in l and (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[5][0] > pts[4][0]):
		ch1 = 2
	l = [[6, 0], [6, 6], [6, 2]]
	pl = [ch1, ch2]
	if pl in l and distance(pts[8], pts[16]) < 52:
		ch1 = 2
	l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
	pl = [ch1, ch2]
	if pl in l and (pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]):
		ch1 = 3
	l = [[6, 4], [6, 1], [6, 2]]
	pl = [ch1, ch2]
	if pl in l and distance(pts[4], pts[11]) > 55:
		ch1 = 4
	l = [[3, 6], [3, 5], [3, 4]]
	pl = [ch1, ch2]
	if pl in l and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
		ch1 = 5
	l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
	pl = [ch1, ch2]
	if pl in l and pts[5][0] > pts[16][0]:
		ch1 = 6
	l = [[7, 2]]
	pl = [ch1, ch2]
	if pl in l and (pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]):
		ch1 = 6
	# Map to letters
	if ch1 == 0:
		res = 'S'
		if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
			res = 'A'
		elif pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
			res = 'T'
		elif pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
			res = 'E'
		elif pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
			res = 'M'
		elif pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
			res = 'N'
		return res
	elif ch1 == 2:
		return 'C' if distance(pts[12], pts[4]) > 42 else 'O'
	elif ch1 == 3:
		return 'G' if distance(pts[8], pts[12]) > 72 else 'H'
	elif ch1 == 7:
		return 'Y' if distance(pts[8], pts[4]) > 42 else 'J'
	elif ch1 == 4:
		return 'L'
	elif ch1 == 6:
		return 'X'
	elif ch1 == 5:
		if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
			return 'Z' if pts[8][1] < pts[5][1] else 'Q'
		return 'P'
	elif ch1 == 1:
		if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
			return 'B'
		if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
			return 'D'
		if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
			return 'F'
		if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
			return 'I'
		if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
			return 'W'
		if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
			return 'K'
		if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
			return 'U'
		if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
			return 'V'
		if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
			return 'R'
	return '?'


def main():
	print('Starting camera letter check. Press the expected letter key to record, ESC to quit.')
	model = load_model(MODEL_PATH)
	detector = HandDetector(maxHands=1)
	cap = cv2.VideoCapture(0)
	stats_correct = defaultdict(int)
	stats_total = defaultdict(int)
	current_pred = ' '
	while True:
		ok, frame = cap.read()
		if not ok:
			continue
		frame = cv2.flip(frame, 1)
		hands = detector.findHands(frame, draw=False, flipType=True)
		if hands and hands[0]:
			hand = hands[0]
			hmap = hand[0]
			pts = hmap['lmList']
			current_pred = predict_letter(model, pts)
			cv2.putText(frame, f'Pred: {current_pred}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
		else:
			cv2.putText(frame, 'No hand', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
		cv2.imshow('Letter Check', frame)
		k = cv2.waitKey(1) & 0xFF
		if k == 27:  # ESC
			break
		if 65 <= k <= 90 or 97 <= k <= 122:  # A-Z or a-z
			expected = chr(k).upper()
			stats_total[expected] += 1
			if current_pred == expected:
				stats_correct[expected] += 1
			print(f'{expected}: recorded; current_pred={current_pred}')
	cap.release()
	cv2.destroyAllWindows()
	# Report
	letters = sorted(set(list(stats_total.keys())))
	for L in letters:
		t = stats_total[L]
		c = stats_correct[L]
		acc = (c/t*100) if t else 0
		print(f'{L}: {c}/{t} ({acc:.0f}%)')


if __name__ == '__main__':
	main()
