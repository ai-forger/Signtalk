import os
import glob
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'AtoZ_3.1')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')

offset = 29

def distance(x, y):
	return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def draw_skeleton_from_pts(pts, w, h):
	white = np.ones((400, 400, 3), dtype=np.uint8) * 255
	# Centering offsets
	x_offset = ((400 - w) // 2) - 15
	y_offset = ((400 - h) // 2) - 15
	# Chains
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
	# Derive width/height from landmarks
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
	# ch3 not used in rules beyond masking
	ch3 = int(np.argmax(prob, axis=0))

	pl = [ch1, ch2]
	# The following rules mirror final_pred.py (simplified to compute final ch1)
	# Due to length, we include critical discriminators to map groups to letters.
	# Group S/A/T/E/M/N
	l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
	     [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
	     [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
	if pl in l:
		if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
			ch1 = 0

	# c/o
	l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
	pl = [ch1, ch2]
	if pl in l:
		if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
			ch1 = 2
	l = [[6, 0], [6, 6], [6, 2]]
	pl = [ch1, ch2]
	if pl in l:
		if distance(pts[8], pts[16]) < 52:
			ch1 = 2

	# gh vs others
	l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
	pl = [ch1, ch2]
	if pl in l:
		if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
			ch1 = 3

	# l group
	l = [[6, 4], [6, 1], [6, 2]]
	pl = [ch1, ch2]
	if pl in l:
		if distance(pts[4], pts[11]) > 55:
			ch1 = 4

	# z/p/q group
	l = [[3, 6], [3, 5], [3, 4]]
	pl = [ch1, ch2]
	if pl in l and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
		ch1 = 5

	# x group
	l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
	pl = [ch1, ch2]
	if pl in l and pts[5][0] > pts[16][0]:
		ch1 = 6

	# y/j group
	l = [[7, 2]]
	pl = [ch1, ch2]
	if pl in l and (pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]):
		ch1 = 6

	# Map groups to letters based on rules
	if ch1 == 0:
		# S/A/T/E/M/N
		result = 'S'
		if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
			result = 'A'
		elif pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
			result = 'T'
		elif pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
			result = 'E'
		elif pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
			result = 'M'
		elif pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
			result = 'N'
		return result
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
		# B/D/F/I/W/K/U/V/R
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
	# Fallback: unknown
	return '?' 

def test_single_image(model, detector, img_path, expected_letter):
	"""Test a single image and return prediction results"""
	try:
		img = cv2.imread(img_path)
		if img is None:
			return {'success': False, 'error': 'Failed to load image', 'prediction': None}
		
		hands = detector.findHands(img, draw=False, flipType=True)
		if not hands or not hands[0]:
			return {'success': False, 'error': 'No hand detected', 'prediction': None}
		
		hand = hands[0]
		hmap = hand[0] if isinstance(hand, (list, tuple)) else hand
		pts = hmap['lmList']
		
		# Get bounding box
		bbox = hmap.get('bbox', None)
		if bbox is not None:
			x, y, w, h = bbox
		else:
			xs = [p[0] for p in pts]
			ys = [p[1] for p in pts]
			w = max(xs) - min(xs) + 1
			h = max(ys) - min(ys) + 1
		
		pred = predict_letter(model, pts)
		correct = pred == expected_letter
		
		return {
			'success': True,
			'prediction': pred,
			'expected': expected_letter,
			'correct': correct,
			'confidence': 1.0  # Placeholder for confidence
		}
		
	except Exception as e:
		return {'success': False, 'error': str(e), 'prediction': None}

def main(samples_per_letter=10, save_results=True):
	"""Main testing function with enhanced reporting"""
	print("Loading model and detector...")
	detector = HandDetector(maxHands=1)
	
	try:
		model = load_model(MODEL_PATH)
		print(f"Model loaded successfully from {MODEL_PATH}")
	except Exception as e:
		print(f"Error loading model: {e}")
		return
	
	# Check if dataset directory exists
	if not os.path.isdir(DATASET_DIR):
		print(f"Dataset directory not found: {DATASET_DIR}")
		print("Please ensure the AtoZ_3.1 folder is in the same directory as this script.")
		return
	
	report = {}
	start_time = time.time()
	
	print(f"\nStarting comprehensive testing of all alphabets...")
	print(f"Testing {samples_per_letter} samples per letter")
	print("=" * 60)
	
	for letter in ascii_uppercase:
		folder = os.path.join(DATASET_DIR, letter)
		if not os.path.isdir(folder):
			print(f"Letter {letter}: No folder found")
			report[letter] = {'tested': 0, 'correct': 0, 'miss': 0, 'errors': [], 'accuracy': 0.0}
			continue
		
		images = sorted(glob.glob(os.path.join(folder, '*.jpg')))[:samples_per_letter]
		if not images:
			print(f"Letter {letter}: No images found")
			report[letter] = {'tested': 0, 'correct': 0, 'miss': 0, 'errors': [], 'accuracy': 0.0}
			continue
		
		print(f"Testing letter {letter} ({len(images)} images)...")
		correct = 0
		miss = 0
		errors = []
		
		for i, img_path in enumerate(images):
			result = test_single_image(model, detector, img_path, letter)
			
			if result['success']:
				if result['correct']:
					correct += 1
				else:
					errors.append({
						'image': os.path.basename(img_path),
						'predicted': result['prediction'],
						'expected': result['expected']
					})
			else:
				miss += 1
				errors.append({
					'image': os.path.basename(img_path),
					'error': result['error']
				})
			
			# Progress indicator
			if (i + 1) % 10 == 0 or (i + 1) == len(images):
				print(f"  Progress: {i + 1}/{len(images)}")
		
		accuracy = (correct / len(images)) * 100 if len(images) > 0 else 0.0
		report[letter] = {
			'tested': len(images),
			'correct': correct,
			'miss': miss,
			'errors': errors,
			'accuracy': accuracy
		}
		
		print(f"  Letter {letter}: {correct}/{len(images)} correct ({accuracy:.1f}%)")
	
	# Calculate overall statistics
	total_letters = 0
	total_correct = 0
	total_miss = 0
	
	for letter in ascii_uppercase:
		if letter in report:
			res = report[letter]
			total_letters += res['tested']
			total_correct += res['correct']
			total_miss += res['miss']
	
	overall_accuracy = (total_correct / total_letters) * 100 if total_letters > 0 else 0.0
	
	# Print comprehensive report
	print("\n" + "=" * 60)
	print("COMPREHENSIVE TESTING REPORT")
	print("=" * 60)
	
	# Summary table
	print(f"{'Letter':<5} {'Tested':<8} {'Correct':<8} {'Miss':<6} {'Accuracy':<10}")
	print("-" * 45)
	
	for letter in ascii_uppercase:
		if letter in report:
			res = report[letter]
			if res['tested'] > 0:
				print(f"{letter:<5} {res['tested']:<8} {res['correct']:<8} {res['miss']:<6} {res['accuracy']:<10.1f}%")
			else:
				print(f"{letter:<5} {'No data':<8} {'-':<8} {'-':<6} {'-':<10}")
	
	print("-" * 45)
	print(f"{'TOTAL':<5} {total_letters:<8} {total_correct:<8} {total_miss:<6} {overall_accuracy:<10.1f}%")
	
	# Performance metrics
	elapsed_time = time.time() - start_time
	print(f"\nTesting completed in {elapsed_time:.2f} seconds")
	print(f"Overall accuracy: {overall_accuracy:.1f}%")
	
	# Identify problematic letters
	problem_letters = []
	for letter in ascii_uppercase:
		if letter in report and report[letter]['tested'] > 0:
			if report[letter]['accuracy'] < 80.0:
				problem_letters.append((letter, report[letter]['accuracy']))
	
	if problem_letters:
		print(f"\nLetters with accuracy < 80%:")
		for letter, acc in sorted(problem_letters, key=lambda x: x[1]):
			print(f"  {letter}: {acc:.1f}%")
	
	# Save detailed results if requested
	if save_results:
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		results_file = f"alphabet_test_results_{timestamp}.json"
		
		# Prepare results for saving
		save_data = {
			'timestamp': timestamp,
			'test_parameters': {
				'samples_per_letter': samples_per_letter,
				'dataset_path': DATASET_DIR,
				'model_path': MODEL_PATH
			},
			'overall_stats': {
				'total_tested': total_letters,
				'total_correct': total_correct,
				'total_miss': total_miss,
				'overall_accuracy': overall_accuracy,
				'elapsed_time': elapsed_time
			},
			'letter_results': report
		}
		
		try:
			with open(results_file, 'w') as f:
				json.dump(save_data, f, indent=2)
			print(f"\nDetailed results saved to: {results_file}")
		except Exception as e:
			print(f"Warning: Could not save results to file: {e}")
	
	return report

if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description='Test ASL alphabet recognition on dataset images')
	parser.add_argument('--samples', type=int, default=10, 
					   help='Number of samples to test per letter (default: 10)')
	parser.add_argument('--no-save', action='store_true',
					   help='Do not save detailed results to file')
	
	args = parser.parse_args()
	
	# Run the test
	main(samples_per_letter=args.samples, save_results=not args.no_save)
