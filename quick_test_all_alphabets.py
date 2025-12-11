#!/usr/bin/env python3
"""
Quick test script to verify all alphabets are working
Run this to get a fast overview of model performance
"""

import os
import glob
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'AtoZ_3.1')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')

def quick_test(samples_per_letter=5):
    """Quick test with minimal output"""
    print("🔍 Quick ASL Alphabet Test")
    print("=" * 40)
    
    # Load model
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load detector
    detector = HandDetector(maxHands=1)
    
    # Check dataset
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        return
    
    # Test each letter
    results = {}
    total_correct = 0
    total_tested = 0
    
    for letter in ascii_uppercase:
        folder = os.path.join(DATASET_DIR, letter)
        if not os.path.isdir(folder):
            results[letter] = "No data"
            continue
        
        images = sorted(glob.glob(os.path.join(folder, '*.jpg')))[:samples_per_letter]
        if not images:
            results[letter] = "No images"
            continue
        
        correct = 0
        tested = 0
        
        for img_path in images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                hands = detector.findHands(img, draw=False, flipType=True)
                if not hands or not hands[0]:
                    continue
                
                hand = hands[0]
                hmap = hand[0] if isinstance(hand, (list, tuple)) else hand
                pts = hmap['lmList']
                
                # Simple prediction (using the same logic as main script)
                # This is a simplified version - for full testing use test_all_letters.py
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                w = max(xs) - min(xs) + 1
                h = max(ys) - min(ys) + 1
                
                # Create skeleton image
                white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                x_offset = ((400 - w) // 2) - 15
                y_offset = ((400 - h) // 2) - 15
                
                # Draw basic skeleton (simplified)
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + x_offset, pts[i][1] + y_offset), 2, (0, 0, 255), 1)
                
                # Predict
                inp = white.reshape(1, 400, 400, 3)
                prob = np.array(model.predict(inp, verbose=0)[0], dtype='float32')
                predicted_group = int(np.argmax(prob, axis=0))
                
                # Simple group to letter mapping (basic)
                if predicted_group == 0:  # Group 0: A, E, M, N, S, T
                    predicted = 'A'  # Simplified - just use first letter
                elif predicted_group == 1:  # Group 1: B, D, F, I, K, R, U, V, W
                    predicted = 'B'
                elif predicted_group == 2:  # Group 2: C, O
                    predicted = 'C'
                elif predicted_group == 3:  # Group 3: G, H
                    predicted = 'G'
                elif predicted_group == 4:  # Group 4: L
                    predicted = 'L'
                elif predicted_group == 5:  # Group 5: P, Q, Z
                    predicted = 'P'
                elif predicted_group == 6:  # Group 6: X
                    predicted = 'X'
                elif predicted_group == 7:  # Group 7: Y, J
                    predicted = 'Y'
                else:
                    predicted = '?'
                
                if predicted == letter:
                    correct += 1
                tested += 1
                
            except Exception as e:
                continue
        
        if tested > 0:
            accuracy = (correct / tested) * 100
            results[letter] = f"{correct}/{tested} ({accuracy:.0f}%)"
            total_correct += correct
            total_tested += tested
        else:
            results[letter] = "Failed"
    
    # Print results
    print("\n📊 Results Summary:")
    print("-" * 40)
    
    for letter in ascii_uppercase:
        result = results.get(letter, "Unknown")
        status = "✅" if isinstance(result, str) and "/" in result and "100" in result else "⚠️"
        print(f"{status} {letter}: {result}")
    
    if total_tested > 0:
        overall_accuracy = (total_correct / total_tested) * 100
        print(f"\n🎯 Overall: {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
        
        if overall_accuracy >= 90:
            print("🎉 Excellent performance!")
        elif overall_accuracy >= 80:
            print("👍 Good performance")
        elif overall_accuracy >= 70:
            print("⚠️  Moderate performance - some issues detected")
        else:
            print("❌ Poor performance - significant issues detected")
    else:
        print("\n❌ No tests completed successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test of all ASL alphabets')
    parser.add_argument('--samples', type=int, default=5, 
                       help='Number of samples to test per letter (default: 5)')
    
    args = parser.parse_args()
    
    print(f"Testing {args.samples} samples per letter...")
    quick_test(samples_per_letter=args.samples)
