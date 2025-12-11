#!/usr/bin/env python3
"""
Test script for skeleton images - direct model testing without hand detection
"""

import os
import glob
import cv2
import numpy as np
from keras.models import load_model
from string import ascii_uppercase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'AtoZ_3.1')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')

def test_skeleton_images(samples_per_letter=5):
    """Test the model directly on skeleton images"""
    print("🔍 Skeleton Image Test")
    print("=" * 40)
    
    # Load model
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Check dataset
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        return
    
    print(f"✅ Dataset directory found: {DATASET_DIR}")
    
    # Test each letter
    results = {}
    total_correct = 0
    total_tested = 0
    
    for letter in ascii_uppercase:
        folder = os.path.join(DATASET_DIR, letter)
        if not os.path.isdir(folder):
            results[letter] = "No folder"
            continue
        
        images = sorted(glob.glob(os.path.join(folder, '*.jpg')))[:samples_per_letter]
        if not images:
            results[letter] = "No images"
            continue
        
        print(f"Testing letter {letter} ({len(images)} images)...")
        correct = 0
        tested = 0
        
        for img_path in images:
            try:
                # Load image directly (these are already skeleton images)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Resize to 400x400 if needed
                if img.shape[:2] != (400, 400):
                    img = cv2.resize(img, (400, 400))
                
                # Prepare input for model
                inp = img.reshape(1, 400, 400, 3)
                
                # Make prediction
                prob = np.array(model.predict(inp, verbose=0)[0], dtype='float32')
                predicted_group = int(np.argmax(prob, axis=0))
                
                # Map group to expected letter
                expected_groups = {
                    'A': 0, 'E': 0, 'M': 0, 'N': 0, 'S': 0, 'T': 0,
                    'B': 1, 'D': 1, 'F': 1, 'I': 1, 'K': 1, 'R': 1, 'U': 1, 'V': 1, 'W': 1,
                    'C': 2, 'O': 2,
                    'G': 3, 'H': 3,
                    'L': 4,
                    'P': 5, 'Q': 5, 'Z': 5,
                    'X': 6,
                    'Y': 7, 'J': 7
                }
                
                expected_group = expected_groups.get(letter, -1)
                
                if predicted_group == expected_group:
                    correct += 1
                    print(f"  ✅ {os.path.basename(img_path)}: Group {predicted_group} (correct)")
                else:
                    print(f"  ❌ {os.path.basename(img_path)}: Group {predicted_group} (expected {expected_group})")
                
                tested += 1
                
            except Exception as e:
                print(f"  ⚠️  Error processing {os.path.basename(img_path)}: {e}")
                continue
        
        if tested > 0:
            accuracy = (correct / tested) * 100
            results[letter] = f"{correct}/{tested} ({accuracy:.1f}%)"
            total_correct += correct
            total_tested += tested
            print(f"  📊 Letter {letter}: {correct}/{tested} correct ({accuracy:.1f}%)")
        else:
            results[letter] = "Failed"
            print(f"  ❌ Letter {letter}: No successful tests")
    
    # Print results summary
    print("\n" + "=" * 40)
    print("📊 FINAL RESULTS")
    print("=" * 40)
    
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
    
    parser = argparse.ArgumentParser(description='Test ASL model on skeleton images')
    parser.add_argument('--samples', type=int, default=5, 
                       help='Number of samples to test per letter (default: 5)')
    
    args = parser.parse_args()
    
    print(f"Testing {args.samples} samples per letter...")
    test_skeleton_images(samples_per_letter=args.samples)
