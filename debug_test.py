#!/usr/bin/env python3
"""
Debug script to test ASL alphabet recognition step by step
"""

import os
import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'AtoZ_3.1')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')

def test_single_image_debug():
    """Test a single image with detailed debugging"""
    print("🔍 Debug Test - Single Image")
    print("=" * 50)
    
    # Load model
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load detector
    detector = HandDetector(maxHands=1)
    print("✅ Hand detector loaded")
    
    # Check dataset
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        return
    
    print(f"✅ Dataset directory found: {DATASET_DIR}")
    
    # Test with a single image from letter A
    test_image_path = os.path.join(DATASET_DIR, 'A', '0.jpg')
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    print(f"✅ Test image found: {test_image_path}")
    
    # Load and process image
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print("❌ Failed to load image")
            return
        
        print(f"✅ Image loaded successfully, shape: {img.shape}")
        
        # Detect hands
        hands = detector.findHands(img, draw=False, flipType=True)
        print(f"✅ Hand detection completed, found {len(hands) if hands else 0} hands")
        
        if not hands:
            print("❌ No hands detected in image")
            return
        
        print(f"   Hands structure: {type(hands)}")
        print(f"   First hand type: {type(hands[0])}")
        
        # Handle different hand result formats
        if isinstance(hands[0], (list, tuple)):
            print(f"   First hand is list/tuple with {len(hands[0])} elements")
            hand = hands[0]
            if len(hand) > 0:
                hmap = hand[0]
                print(f"   Hand[0] type: {type(hmap)}")
                print(f"   Hand[0] keys: {hmap.keys() if hasattr(hmap, 'keys') else 'No keys'}")
            else:
                print("❌ Empty hand list")
                return
        else:
            hmap = hands[0]
            print(f"   Hand is direct object, type: {type(hmap)}")
            print(f"   Hand keys: {hmap.keys() if hasattr(hmap, 'keys') else 'No keys'}")
        
        # Check if we have landmarks
        if not hasattr(hmap, 'get') or 'lmList' not in hmap:
            print("❌ No landmarks found in hand data")
            print(f"   Available attributes: {dir(hmap)}")
            return
        
        pts = hmap['lmList']
        print(f"✅ Hand landmarks extracted, {len(pts)} points")
        print(f"   First few points: {pts[:3]}")
        
        # Get bounding box
        bbox = hmap.get('bbox', None)
        if bbox is not None:
            x, y, w, h = bbox
            print(f"✅ Bounding box: x={x}, y={y}, w={w}, h={h}")
        else:
            # Calculate from landmarks
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs) + 1
            h = max(ys) - min(ys) + 1
            print(f"✅ Calculated dimensions: w={w}, h={h}")
        
        # Create skeleton image
        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
        x_offset = ((400 - w) // 2) - 15
        y_offset = ((400 - h) // 2) - 15
        
        print(f"✅ Offsets calculated: x_offset={x_offset}, y_offset={y_offset}")
        
        # Draw basic skeleton
        for i in range(21):
            cv2.circle(white, (pts[i][0] + x_offset, pts[i][1] + y_offset), 2, (0, 0, 255), 1)
        
        print("✅ Skeleton drawn")
        
        # Save debug image
        debug_path = "debug_skeleton.jpg"
        cv2.imwrite(debug_path, white)
        print(f"✅ Debug skeleton saved to: {debug_path}")
        
        # Predict
        inp = white.reshape(1, 400, 400, 3)
        print(f"✅ Input prepared, shape: {inp.shape}")
        
        prob = np.array(model.predict(inp, verbose=0)[0], dtype='float32')
        print(f"✅ Model prediction completed, output shape: {prob.shape}")
        print(f"   Raw probabilities: {prob}")
        
        predicted_group = int(np.argmax(prob, axis=0))
        confidence = float(prob[predicted_group])
        
        print(f"✅ Prediction: Group {predicted_group} with confidence {confidence:.4f}")
        
        # Map group to letter (simplified)
        group_letters = {
            0: "A/E/M/N/S/T",
            1: "B/D/F/I/K/R/U/V/W", 
            2: "C/O",
            3: "G/H",
            4: "L",
            5: "P/Q/Z",
            6: "X",
            7: "Y/J"
        }
        
        predicted_letters = group_letters.get(predicted_group, "Unknown")
        print(f"✅ Predicted letters: {predicted_letters}")
        
        # Check if this matches expected letter A
        if predicted_group == 0:
            print("✅ SUCCESS: Letter A correctly identified in group 0")
        else:
            print(f"❌ FAILED: Letter A should be in group 0, but got group {predicted_group}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_image_debug()
