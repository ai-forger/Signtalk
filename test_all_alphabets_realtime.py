import cv2
import numpy as np
import math
import os
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import tkinter as tk
from tkinter import ttk
import threading
import time

class AlphabetTester:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load model
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5'), compile=False)
        
        # Hand detector
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        
        # Testing variables
        self.offset = 29
        self.current_letter = 'A'
        self.testing_mode = False
        self.results = {letter: {'correct': 0, 'incorrect': 0, 'total': 0} for letter in ascii_uppercase}
        self.current_prediction = ''
        self.confidence = 0.0
        
        # GUI setup
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("ASL Alphabet Tester")
        self.root.geometry("800x600")
        
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        ttk.Label(control_frame, text="Current Letter:").pack(side=tk.LEFT)
        self.letter_var = tk.StringVar(value='A')
        letter_combo = ttk.Combobox(control_frame, textvariable=self.letter_var, 
                                   values=list(ascii_uppercase), width=5)
        letter_combo.pack(side=tk.LEFT, padx=5)
        letter_combo.bind('<<ComboboxSelected>>', self.change_letter)
        
        ttk.Button(control_frame, text="Start Testing", 
                  command=self.toggle_testing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset Results", 
                  command=self.reset_results).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.Frame(self.root)
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Create treeview for results
        columns = ('Letter', 'Correct', 'Incorrect', 'Total', 'Accuracy')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=10, fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready to test. Select a letter and click 'Start Testing'")
        self.status_label.pack()
        
        self.prediction_label = ttk.Label(status_frame, text="Prediction: None", font=('Arial', 14, 'bold'))
        self.prediction_label.pack()
        
        # Update results display
        self.update_results_display()
        
    def change_letter(self, event=None):
        self.current_letter = self.letter_var.get()
        self.status_label.config(text=f"Testing letter: {self.current_letter}")
        
    def toggle_testing(self):
        self.testing_mode = not self.testing_mode
        if self.testing_mode:
            self.status_label.config(text=f"Testing mode ON - Show letter {self.current_letter}")
            self.start_video_loop()
        else:
            self.status_label.config(text="Testing mode OFF")
            
    def reset_results(self):
        self.results = {letter: {'correct': 0, 'incorrect': 0, 'total': 0} for letter in ascii_uppercase}
        self.update_results_display()
        
    def update_results_display(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add results
        for letter in ascii_uppercase:
            result = self.results[letter]
            total = result['total']
            if total > 0:
                accuracy = (result['correct'] / total) * 100
                self.tree.insert('', 'end', values=(
                    letter, 
                    result['correct'], 
                    result['incorrect'], 
                    total, 
                    f"{accuracy:.1f}%"
                ))
            else:
                self.tree.insert('', 'end', values=(letter, 0, 0, 0, "0.0%"))
                
    def predict_letter(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype='float32')
        
        # Get top 3 predictions
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        
        # Store confidence
        self.confidence = float(prob[ch1]) if ch1 < len(prob) else 0.0
        
        pl = [ch1, ch2]
        
        # Apply the same rules as in final_pred.py
        # Group 0: [A, E, M, N, S, T]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        # Group 2: [C, O]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # Group 3: [G, H]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        # Group 4: [L]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # Group 5: [P, Q, Z]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # Group 6: [X]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # Group 7: [Y, J]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # Map groups to specific letters
        if ch1 == 0:
            # Group 0: [A, E, M, N, S, T]
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            elif self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'
            else:
                ch1 = 'S'
        elif ch1 == 2:
            # Group 2: [C, O]
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'
        elif ch1 == 3:
            # Group 3: [G, H]
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'
        elif ch1 == 7:
            # Group 7: [Y, J]
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'
        elif ch1 == 4:
            ch1 = 'L'
        elif ch1 == 6:
            ch1 = 'X'
        elif ch1 == 5:
            # Group 5: [P, Q, Z]
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'
        elif ch1 == 1:
            # Group 1: [B, D, F, I, K, R, U, V, W]
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            elif (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            elif (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            elif ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            elif ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            elif (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'
            else:
                ch1 = 'B'  # Default fallback
        else:
            ch1 = '?'
            
        return ch1

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def start_video_loop(self):
        def video_loop():
            while self.testing_mode:
                try:
                    ok, frame = self.vs.read()
                    if not ok or frame is None:
                        continue
                        
                    cv2image = cv2.flip(frame, 1)
                    hands = self.hd.findHands(cv2image, draw=False, flipType=True)
                    
                    if hands and hands[0]:
                        hand = hands[0]
                        map = hand[0]
                        x, y, w, h = map['bbox']
                        image = cv2image[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                        
                        if image is not None and image.size != 0:
                            handz = self.hd2.findHands(image, draw=False, flipType=True)
                            if handz and handz[0]:
                                hand = handz[0]
                                handmap = hand[0]
                                self.pts = handmap['lmList']

                                x_offset = ((400 - w) // 2) - 15
                                y_offset = ((400 - h) // 2) - 15
                                
                                # Draw hand skeleton
                                for t in range(0, 4, 1):
                                    cv2.line(white, (self.pts[t][0] + x_offset, self.pts[t][1] + y_offset), 
                                             (self.pts[t + 1][0] + x_offset, self.pts[t + 1][1] + y_offset), (0, 255, 0), 3)
                                for t in range(5, 8, 1):
                                    cv2.line(white, (self.pts[t][0] + x_offset, self.pts[t][1] + y_offset), 
                                             (self.pts[t + 1][0] + x_offset, self.pts[t + 1][1] + y_offset), (0, 255, 0), 3)
                                for t in range(9, 12, 1):
                                    cv2.line(white, (self.pts[t][0] + x_offset, self.pts[t][1] + y_offset), 
                                             (self.pts[t + 1][0] + x_offset, self.pts[t + 1][1] + y_offset), (0, 255, 0), 3)
                                for t in range(13, 16, 1):
                                    cv2.line(white, (self.pts[t][0] + x_offset, self.pts[t][1] + y_offset), 
                                             (self.pts[t + 1][0] + x_offset, self.pts[t + 1][1] + y_offset), (0, 255, 0), 3)
                                for t in range(17, 20, 1):
                                    cv2.line(white, (self.pts[t][0] + x_offset, self.pts[t][1] + y_offset), 
                                             (self.pts[t + 1][0] + x_offset, self.pts[t + 1][1] + y_offset), (0, 255, 0), 3)
                                
                                cv2.line(white, (self.pts[5][0] + x_offset, self.pts[5][1] + y_offset), 
                                         (self.pts[9][0] + x_offset, self.pts[9][1] + y_offset), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[9][0] + x_offset, self.pts[9][1] + y_offset), 
                                         (self.pts[13][0] + x_offset, self.pts[13][1] + y_offset), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[13][0] + x_offset, self.pts[13][1] + y_offset), 
                                         (self.pts[17][0] + x_offset, self.pts[17][1] + y_offset), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[0][0] + x_offset, self.pts[0][0] + y_offset), 
                                         (self.pts[5][0] + x_offset, self.pts[5][1] + y_offset), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[0][0] + x_offset, self.pts[0][0] + y_offset), 
                                         (self.pts[17][0] + x_offset, self.pts[17][1] + y_offset), (0, 255, 0), 3)

                                for i in range(21):
                                    cv2.circle(white, (self.pts[i][0] + x_offset, self.pts[i][0] + y_offset), 2, (0, 0, 255), 1)

                                # Predict letter
                                predicted = self.predict_letter(white)
                                self.current_prediction = predicted
                                
                                # Update GUI
                                self.root.after(0, self.update_prediction_display, predicted)
                                
                                # Check if prediction matches current letter
                                if predicted == self.current_letter:
                                    # Correct prediction - wait a bit then record
                                    time.sleep(0.5)  # Small delay to avoid multiple counts
                                    self.root.after(0, self.record_result, True)
                                elif predicted != '?' and predicted != ' ':
                                    # Incorrect prediction - wait a bit then record
                                    time.sleep(0.5)
                                    self.root.after(0, self.record_result, False)

                    # Display frame
                    cv2.imshow("ASL Alphabet Tester", cv2image)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                        
                except Exception as e:
                    print(f"Error in video loop: {e}")
                    continue
                    
            self.vs.release()
            cv2.destroyAllWindows()
            
        # Start video loop in separate thread
        self.video_thread = threading.Thread(target=video_loop, daemon=True)
        self.video_thread.start()
        
    def update_prediction_display(self, prediction):
        self.prediction_label.config(text=f"Prediction: {prediction} (Confidence: {self.confidence:.2f})")
        
    def record_result(self, correct):
        if correct:
            self.results[self.current_letter]['correct'] += 1
        else:
            self.results[self.current_letter]['incorrect'] += 1
            
        self.results[self.current_letter]['total'] += 1
        self.update_results_display()
        
        # Update status
        accuracy = (self.results[self.current_letter]['correct'] / self.results[self.current_letter]['total']) * 100
        self.status_label.config(text=f"Letter {self.current_letter}: {self.results[self.current_letter]['correct']}/{self.results[self.current_letter]['total']} correct ({accuracy:.1f}%)")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    tester = AlphabetTester()
    tester.run()
