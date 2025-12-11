# Enhanced Sign Language to Text and Speech Application - Improved GUI
import numpy as np
import math
import cv2
import os
import sys
import traceback
import pyttsx3
import threading
import time
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from tkinter import ttk, messagebox
import ctypes
import platform
from PIL import Image, ImageTk

# Initialize components
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"
cv2.setUseOptimized(True)
DEBUG = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class EnhancedSignLanguageApp:
    def __init__(self):
        # Initialize video capture
        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.current_image = None
        
        # Load model
        self.model = load_model(os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5'), compile=False)
        
        # Flag for controlling the speech thread
        self.running = True
        
        # Simplified speech engine setup
        self.speak_engine = None
        self.tts_lock = threading.Lock()
        try:
            self.speak_engine = pyttsx3.init()
            self.speak_engine.setProperty("rate", 180)
            voices = self.speak_engine.getProperty("voices")
            if voices:
                preferred_voice = next((v for v in voices if 'zira' in v.name.lower() or 'english' in v.name.lower()), voices[0])
                self.speak_engine.setProperty("voice", preferred_voice.id)
                print(f"Using voice: {preferred_voice.name}")
            else:
                print("No voices found.")
        except Exception as e:
            print(f"Speech engine initialization error: {e}")
        
        # Speech control setup
        self.speech_enabled = True
        self.speech_thread = threading.Thread(target=self._run_tts_loop, daemon=True)
        self.speech_thread.start()

        # Character tracking
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self.last_checked_word = ""
        
        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Text and word variables
        self.str = ""
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        
        # Animation variables
        self.animation_frame = 0
        self.last_recognized_char = ""
        self.char_highlight_timer = 0
        
        print("Loaded model from disk")
        
        self._ensure_white_background()
        
        self.setup_modern_gui()
        self.video_loop()

    def _speak(self, text):
        """Adds text to the speech engine's queue without blocking."""
        if self.speech_enabled and self.speak_engine and text:
            with self.tts_lock:
                try:
                    self.speak_engine.stop()
                    self.speak_engine.say(text)
                except Exception as e:
                    print(f"Error in _speak: {e}")

    def _run_tts_loop(self):
        """Runs the pyttsx3 event loop in the background."""
        while self.running:
            if self.speak_engine:
                try:
                    self.speak_engine.runAndWait()
                except Exception as e:
                    print(f"Error in TTS loop: {e}")
                    try:
                        self.speak_engine = pyttsx3.init()
                    except Exception as e2:
                        print(f"Failed to re-init TTS engine: {e2}")
                        self.speak_engine = None
            time.sleep(0.1)

    def _ensure_white_background(self):
        """Ensure white.jpg exists for hand skeleton drawing"""
        white_path = os.path.join(BASE_DIR, 'white.jpg')
        if not os.path.exists(white_path):
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            cv2.imwrite(white_path, white)
            print(f"Created white background image: {white_path}")

    def setup_modern_gui(self):
        """Create a modern, attractive GUI with improved layout"""
        self.root = tk.Tk()
        self.root.title("Enhanced Sign Language Recognition System")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        # Get screen dimensions for responsive design
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate window size (80% of screen)
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 800)
        # Subtle translucency for a glassy effect (Windows supports this)
        try:
            self.root.wm_attributes('-alpha', 0.98)
        except Exception:
            pass
        # Apply a soft gradient background
        self._apply_gradient_background(window_width, window_height)
        # Try enabling Windows Acrylic blur for a glass effect
        self._enable_windows_acrylic(accent_color='#0b132b')
        # Default theme
        self.theme_mode = 'dark'
        
        # Configure main grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Header Section
        self.create_header()
        
        # Main Content Area with proper grid layout
        self.create_main_content()
        
        # Status Bar
        self.create_status_bar()

    def create_header(self):
        """Create the header section with title and subtitle"""
        self.header_frame = tk.Frame(self.root, bg='#0f172a', height=100)
        self.header_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
        self.header_frame.grid_propagate(False)
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=0)
        
        # Main title
        self.title_label = tk.Label(
            self.header_frame, 
            text="🎯 Enhanced Sign Language Recognition",
            font=('Segoe UI', 28, 'bold'),
            fg='#e2e8f0',
            bg='#0f172a'
        )
        self.title_label.grid(row=0, column=0, pady=(15, 5), sticky='w', padx=20)
        
        # Subtitle
        self.subtitle_label = tk.Label(
            self.header_frame,
            text="Real-time ASL to Text & Speech Conversion with Advanced AI Recognition",
            font=('Segoe UI', 12),
            fg='#94a3b8',
            bg='#0f172a'
        )
        self.subtitle_label.grid(row=1, column=0, pady=(0, 15), sticky='w', padx=20)
        # Theme toggle button
        self.theme_btn = tk.Button(
            self.header_frame,
            text='🌗 Toggle Theme',
            command=self._toggle_theme,
            font=('Segoe UI', 10, 'bold'),
            bg='#111827', fg='#e2e8f0',
            activebackground='#1f2937', activeforeground='#e2e8f0',
            bd=2, relief='ridge', cursor='hand2'
        )
        self.theme_btn.grid(row=0, column=1, rowspan=2, sticky='e', padx=20)

    def create_main_content(self):
        """Create the main content area with proper grid layout"""
        self.main_container = tk.Frame(self.root, bg='#0b132b')
        self.main_container.grid(row=1, column=0, sticky='nsew', padx=15, pady=15)
        
        # Configure grid weights for responsive layout
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=3)  # Left panel gets 60% width
        self.main_container.grid_columnconfigure(1, weight=2)  # Right panel gets 40% width
        
        # Left Panel - Video and Skeleton
        self.create_left_panel(self.main_container)
        
        # Right Panel - Controls and Text
        self.create_right_panel(self.main_container)

    def create_left_panel(self, parent):
        """Create the left panel with video feeds"""
        self.left_panel = tk.Frame(parent, bg='#111827', relief='ridge', bd=2, highlightthickness=1, highlightbackground='#1f2937')
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Configure left panel grid
        # Stabilize proportions using uniform so rows scale proportionally
        self.left_panel.grid_rowconfigure(0, weight=3, uniform='leftpanel')  # Camera gets ~60% height
        self.left_panel.grid_rowconfigure(1, weight=2, uniform='leftpanel')  # Skeleton gets ~40% height
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Camera Feed Section
        self.camera_frame = tk.LabelFrame(
            self.left_panel,
            text="📹 Live Camera Feed",
            font=('Segoe UI', 14, 'bold'),
            fg='#38bdf8',
            bg='#111827',
            relief='groove',
            bd=3
        )
        self.camera_frame.grid(row=0, column=0, sticky='nsew', padx=15, pady=(15, 8))
        self.camera_frame.grid_rowconfigure(0, weight=1)
        self.camera_frame.grid_columnconfigure(0, weight=1)
        
        self.panel = tk.Label(
            self.camera_frame,
            bg='#0b1220',
            relief='sunken',
            bd=3,
            text="Camera Initializing...",
            font=('Segoe UI', 16),
            fg='#c7e7ff'
        )
        self.panel.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Hand Skeleton Section
        self.skeleton_frame = tk.LabelFrame(
            self.left_panel,
            text="🦴 Hand Skeleton Analysis",
            font=('Segoe UI', 14, 'bold'),
            fg='#38bdf8',
            bg='#111827',
            relief='groove',
            bd=3
        )
        self.skeleton_frame.grid(row=1, column=0, sticky='nsew', padx=15, pady=(8, 15))
        self.skeleton_frame.grid_rowconfigure(0, weight=1)
        self.skeleton_frame.grid_columnconfigure(0, weight=1)
        
        self.panel2 = tk.Label(
            self.skeleton_frame,
            bg='#0b1220',
            relief='sunken',
            bd=3,
            text="Hand skeleton will appear here",
            font=('Segoe UI', 14),
            fg='#c7e7ff'
        )
        self.panel2.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

    def create_right_panel(self, parent):
        """Create the right panel with controls and text display"""
        self.right_panel = tk.Frame(parent, bg='#111827', relief='ridge', bd=2, highlightthickness=1, highlightbackground='#1f2937')
        self.right_panel.grid(row=0, column=1, sticky='nsew')
        
        # Configure right panel grid
        self.right_panel.grid_rowconfigure(1, weight=1)  # Text area gets most space
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Current Character Display
        self.create_character_display(self.right_panel)
        
        # Text Output Area
        self.create_text_area(self.right_panel)
        
        # Control Panel
        self.create_control_panel(self.right_panel)
        
        # Word Suggestions
        self.create_suggestions_panel(self.right_panel)

    def create_character_display(self, parent):
        """Create the current character display"""
        self.char_frame = tk.LabelFrame(
            parent,
            text="🔤 Current Character",
            font=('Segoe UI', 14, 'bold'),
            fg='#38bdf8',
            bg='#111827',
            relief='groove',
            bd=3
        )
        self.char_frame.grid(row=0, column=0, sticky='ew', padx=15, pady=(15, 8))
        self.char_frame.grid_columnconfigure(0, weight=1)
        
        self.panel3 = tk.Label(
            self.char_frame,
            text="Ready",
            font=('Segoe UI', 42, 'bold'),
            fg='#e2e8f0',
            bg='#0b1220',
            relief='raised',
            bd=4,
            height=2
        )
        self.panel3.grid(row=0, column=0, sticky='ew', padx=12, pady=12)

    def create_text_area(self, parent):
        """Create the text output area"""
        self.text_frame = tk.LabelFrame(
            parent,
            text="📝 Recognized Text Output",
            font=('Segoe UI', 14, 'bold'),
            fg='#38bdf8',
            bg='#111827',
            relief='groove',
            bd=3
        )
        self.text_frame.grid(row=1, column=0, sticky='nsew', padx=15, pady=8)
        self.text_frame.grid_rowconfigure(0, weight=1)
        self.text_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrollable text area
        self.text_container = tk.Frame(self.text_frame, bg='#0b1220')
        self.text_container.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        self.text_container.grid_rowconfigure(0, weight=1)
        self.text_container.grid_columnconfigure(0, weight=1)
        
        # Scrollable text widget
        self.text_widget = tk.Text(
            self.text_container,
            font=('Segoe UI', 16),
            fg='#e5f2ff',
            bg='#0b1220',
            relief='sunken',
            bd=3,
            wrap=tk.WORD,
            state='disabled',
            cursor='arrow'
        )
        self.text_widget.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbar for text widget
        scrollbar = tk.Scrollbar(self.text_container, orient='vertical', command=self.text_widget.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.text_widget.config(yscrollcommand=scrollbar.set)
        
        # Keep the old panel5 for backward compatibility
        self.panel5 = self.text_widget

    def create_control_panel(self, parent):
        """Create the control panel with buttons"""
        self.control_frame = tk.LabelFrame(
            parent,
            text="🎮 Control Panel",
            font=('Segoe UI', 14, 'bold'),
            fg='#38bdf8',
            bg='#111827',
            relief='groove',
            bd=3
        )
        self.control_frame.grid(row=2, column=0, sticky='ew', padx=15, pady=8)
        self.control_frame.grid_columnconfigure(0, weight=1)
        
        # Button container with grid layout
        self.button_container = tk.Frame(self.control_frame, bg='#111827')
        self.button_container.grid(row=0, column=0, sticky='ew', padx=12, pady=12)
        
        # Configure button grid
        for i in range(2):
            self.button_container.grid_rowconfigure(i, weight=1)
        for i in range(3): # Changed from 2 to 3 columns for the new button
            self.button_container.grid_columnconfigure(i, weight=1)
        
        # First row buttons
        self.space_btn = tk.Button(
            self.button_container,
            text="SPACE",
            command=self.add_space,
            font=('Segoe UI', 14, 'bold'),
            bg='#ef4444',
            fg='white',
            relief='flat', # Changed to flat for modern look
            bd=2, # Reduced border width
            activebackground='#b91c1c',
            activeforeground='white',
            cursor='hand2',
            height=2,
            highlightthickness=1, # Added highlight for definition
            highlightbackground='#dc2626' # Matching highlight color
        )
        self.space_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5), pady=(0, 5))
        
        self.clear_btn = tk.Button(
            self.button_container,
            text="🗑️ CLEAR",
            command=self.clear_fun,
            font=('Segoe UI', 14, 'bold'),
            bg='#f59e0b',
            fg='white',
            relief='flat', # Changed to flat for modern look
            bd=2, # Reduced border width
            activebackground='#b45309',
            activeforeground='white',
            cursor='hand2',
            height=2,
            highlightthickness=1, # Added highlight for definition
            highlightbackground='#d97706' # Matching highlight color
        )
        self.clear_btn.grid(row=0, column=1, sticky='ew', padx=(5, 0), pady=(0, 5))
        
        # Backspace button (new)
        self.backspace_btn = tk.Button(
            self.button_container,
            text="⬅️ BACKSPACE",
            command=self.backspace_fun,
            font=('Segoe UI', 14, 'bold'),
            bg='#8b5cf6', # Using a purple tone for consistency with suggestions
            fg='white',
            relief='flat', # Changed to flat for modern look
            bd=2, # Reduced border width
            activebackground='#6d28d9',
            activeforeground='white',
            cursor='hand2',
            height=2,
            highlightthickness=1, # Added highlight for definition
            highlightbackground='#6b21a8' # Matching highlight color
        )
        self.backspace_btn.grid(row=0, column=2, sticky='ew', padx=(5, 0), pady=(0, 5))

        # Second row buttons
        self.speech_btn = tk.Button(
            self.button_container,
            text="🔊 SPEECH ON",
            command=self.toggle_speech,
            font=('Segoe UI', 14, 'bold'),
            bg='#22c55e',
            fg='white',
            relief='flat', # Changed to flat for modern look
            bd=2, # Reduced border width
            activebackground='#15803d',
            activeforeground='white',
            cursor='hand2',
            height=2,
            highlightthickness=1, # Added highlight for definition
            highlightbackground='#16a34a' # Matching highlight color
        )
        self.speech_btn.grid(row=1, column=0, sticky='ew', padx=(0, 5), pady=(5, 0))
        
        self.speak_btn = tk.Button(
            self.button_container,
            text="🎤 SPEAK NOW",
            command=self.speak_fun,
            font=('Segoe UI', 14, 'bold'),
            bg='#3b82f6',
            fg='white',
            relief='flat', # Changed to flat for modern look
            bd=2, # Reduced border width
            activebackground='#1d4ed8',
            activeforeground='white',
            cursor='hand2',
            height=2,
            highlightthickness=1, # Added highlight for definition
            highlightbackground='#2563eb' # Matching highlight color
        )
        self.speak_btn.grid(row=1, column=1, sticky='ew', padx=(5, 0), pady=(5, 0))

    def create_suggestions_panel(self, parent):
        """Create the word suggestions panel"""
        self.suggestions_frame = tk.LabelFrame(
            parent,
            text="💡 Word Suggestions",
            font=('Segoe UI', 14, 'bold'),
            fg='#38bdf8',
            bg='#111827',
            relief='groove',
            bd=3
        )
        self.suggestions_frame.grid(row=3, column=0, sticky='ew', padx=15, pady=(8, 15))
        self.suggestions_frame.grid_columnconfigure(0, weight=1)
        
        # Suggestions button container
        self.suggestions_container = tk.Frame(self.suggestions_frame, bg='#111827')
        self.suggestions_container.grid(row=0, column=0, sticky='ew', padx=12, pady=12)
        
        # Configure suggestions grid
        for i in range(4):
            self.suggestions_container.grid_columnconfigure(i, weight=1)
        
        # Create suggestion buttons
        button_style = {
            'font': ('Segoe UI', 12, 'bold'),
            'bg': '#8b5cf6',
            'fg': 'white',
            'relief': 'flat', # Changed to flat for modern look
            'bd': 2, # Reduced border width
            'activebackground': '#6d28d9',
            'activeforeground': 'white',
            'cursor': 'hand2',
            'height': 2,
            'highlightthickness': 1, # Added highlight for definition
            'highlightbackground': '#6d28d9' # Matching highlight color
        }
        
        self.b1 = tk.Button(self.suggestions_container, text="", command=self.action1, **button_style)
        self.b1.grid(row=0, column=0, sticky='ew', padx=(0, 3))
        
        self.b2 = tk.Button(self.suggestions_container, text="", command=self.action2, **button_style)
        self.b2.grid(row=0, column=1, sticky='ew', padx=3)
        
        self.b3 = tk.Button(self.suggestions_container, text="", command=self.action3, **button_style)
        self.b3.grid(row=0, column=2, sticky='ew', padx=3)
        
        self.b4 = tk.Button(self.suggestions_container, text="", command=self.action4, **button_style)
        self.b4.grid(row=0, column=3, sticky='ew', padx=(3, 0))

    def create_status_bar(self):
        """Create the status bar at the bottom"""
        self.status_frame = tk.Frame(self.root, bg='#0f172a', height=30)
        self.status_frame.grid(row=2, column=0, sticky='ew', padx=0, pady=0)
        self.status_frame.grid_propagate(False)
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="System Ready - Start signing to begin recognition",
            font=('Segoe UI', 10),
            fg='#94a3b8',
            bg='#0f172a'
        )
        self.status_label.grid(row=0, column=0, padx=15, pady=5, sticky='w')

    def _toggle_theme(self):
        self.theme_mode = 'light' if self.theme_mode == 'dark' else 'dark'
        self._apply_theme(self.theme_mode)

    def _apply_theme(self, mode: str) -> None:
        # Define palettes
        dark = {
            'root_bg1': '#0b132b', 'root_bg2': '#172554',
            'panel_bg': '#111827', 'panel_fg': '#38bdf8',
            'card_bg': '#0b1220', 'card_fg': '#e2e8f0', 'muted': '#94a3b8'
        }
        light = {
            'root_bg1': '#e8eef8', 'root_bg2': '#cfe0ff',
            'panel_bg': '#ffffff', 'panel_fg': '#0f172a',
            'card_bg': '#f7fafc', 'card_fg': '#0f172a', 'muted': '#334155'
        }
        pal = dark if mode == 'dark' else light
        # Background gradient: only update colors, keep single bg label and keep it behind
        try:
            w = max(self.root.winfo_width(), 1)
            h = max(self.root.winfo_height(), 1)
            self._apply_gradient_background(w, h, top=pal['root_bg1'], bottom=pal['root_bg2'])
        except Exception:
            pass
        # Header
        self.header_frame.config(bg=pal['panel_bg'])
        self.title_label.config(bg=pal['panel_bg'], fg=pal['card_fg'])
        self.subtitle_label.config(bg=pal['panel_bg'], fg=pal['muted'])
        self.theme_btn.config(bg=pal['panel_bg'], fg=pal['card_fg'], activebackground=pal['panel_bg'])
        # Main containers
        self.main_container.config(bg=pal['root_bg1'])
        for f in [self.left_panel, self.right_panel, self.camera_frame, self.skeleton_frame,
                  self.char_frame, self.text_frame, self.control_frame, self.suggestions_frame,
                  self.button_container, self.suggestions_container, self.text_container, self.status_frame]:
            if f:
                f.config(bg=pal['panel_bg'])
        # Labels and text areas
        for lab in [self.panel, self.panel2, self.panel3, self.text_widget]:
            lab.config(bg=pal['card_bg'], fg=pal['card_fg'])
        # Buttons
        def style_button(btn, bg, abg):
            btn.config(bg=bg, activebackground=abg, fg='white', activeforeground='white')
        if mode == 'dark':
            style_button(self.space_btn, '#ef4444', '#b91c1c')
            style_button(self.clear_btn, '#f59e0b', '#b45309')
            style_button(self.speech_btn, '#22c55e', '#15803d')
            style_button(self.speak_btn, '#3b82f6', '#1d4ed8')
            style_button(self.backspace_btn, '#8b5cf6', '#6d28d9') # Apply dark theme style to backspace
            for b in [self.b1, self.b2, self.b3, self.b4]:
                style_button(b, '#8b5cf6', '#6d28d9')
        else:
            style_button(self.space_btn, '#ef4444', '#dc2626')
            style_button(self.clear_btn, '#f59e0b', '#d97706')
            style_button(self.speech_btn, '#16a34a', '#15803d')
            style_button(self.speak_btn, '#2563eb', '#1d4ed8')
            style_button(self.backspace_btn, '#7c3aed', '#6d28d9') # Apply light theme style to backspace
            for b in [self.b1, self.b2, self.b3, self.b4]:
                style_button(b, '#7c3aed', '#6d28d9')
        # Status bar
        self.status_frame.config(bg=pal['panel_bg'])
        self.status_label.config(bg=pal['panel_bg'], fg=pal['muted'])

    def _on_button_hover(self, button, is_hovered):
        """Apply hover effects to buttons, simulating a glow."""
        glow_thickness = 3 if is_hovered else 1 # Thicker highlight on hover
        
        if self.theme_mode == 'dark':
            if button == self.space_btn:
                button.config(bg='#ff6b6b' if is_hovered else '#ef4444', 
                              highlightbackground='#ff6b6b' if is_hovered else '#dc2626',
                              highlightthickness=glow_thickness)
            elif button == self.clear_btn:
                button.config(bg='#ffc107' if is_hovered else '#f59e0b', 
                              highlightbackground='#ffc107' if is_hovered else '#d97706',
                              highlightthickness=glow_thickness)
            elif button == self.speech_btn:
                button.config(bg='#4ade80' if is_hovered else '#22c55e', 
                              highlightbackground='#4ade80' if is_hovered else '#16a34a',
                              highlightthickness=glow_thickness)
            elif button == self.speak_btn:
                button.config(bg='#60a5fa' if is_hovered else '#3b82f6', 
                              highlightbackground='#60a5fa' if is_hovered else '#2563eb',
                              highlightthickness=glow_thickness)
            elif button == self.backspace_btn: # New backspace button
                button.config(bg='#a78bfa' if is_hovered else '#8b5cf6', 
                              highlightbackground='#a78bfa' if is_hovered else '#6b21a8',
                              highlightthickness=glow_thickness)
            elif button in [self.b1, self.b2, self.b3, self.b4]:
                button.config(bg='#a78bfa' if is_hovered else '#8b5cf6', 
                              highlightbackground='#a78bfa' if is_hovered else '#6d28d9',
                              highlightthickness=glow_thickness)
        else: # Light mode
            if button == self.space_btn:
                button.config(bg='#ff6b6b' if is_hovered else '#ef4444', 
                              highlightbackground='#ff6b6b' if is_hovered else '#dc2626',
                              highlightthickness=glow_thickness)
            elif button == self.clear_btn:
                button.config(bg='#ffc107' if is_hovered else '#f59e0b', 
                              highlightbackground='#ffc107' if is_hovered else '#d97706',
                              highlightthickness=glow_thickness)
            elif button == self.speech_btn:
                button.config(bg='#4ade80' if is_hovered else '#16a34a', 
                              highlightbackground='#4ade80' if is_hovered else '#15803d',
                              highlightthickness=glow_thickness)
            elif button == self.speak_btn:
                button.config(bg='#60a5fa' if is_hovered else '#2563eb', 
                              highlightbackground='#60a5fa' if is_hovered else '#1d4ed8',
                              highlightthickness=glow_thickness)
            elif button == self.backspace_btn: # New backspace button
                button.config(bg='#a78bfa' if is_hovered else '#7c3aed', 
                              highlightbackground='#a78bfa' if is_hovered else '#5b21b6',
                              highlightthickness=glow_thickness)
            elif button in [self.b1, self.b2, self.b3, self.b4]:
                button.config(bg='#a78bfa' if is_hovered else '#7c3aed', 
                              highlightbackground='#a78bfa' if is_hovered else '#6d28d9',
                              highlightthickness=glow_thickness)

    def _enable_windows_acrylic(self, accent_color: str = '#0b132b') -> None:
        try:
            if platform.system() != 'Windows':
                return
            hwnd = self.root.winfo_id()
            user32 = ctypes.windll.user32
            SetWindowCompositionAttribute = getattr(user32, 'SetWindowCompositionAttribute')
            class ACCENTPOLICY(ctypes.Structure):
                _fields_ = [('AccentState', ctypes.c_int),
                            ('AccentFlags', ctypes.c_int),
                            ('GradientColor', ctypes.c_int),
                            ('AnimationId', ctypes.c_int)]
            class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
                _fields_ = [('Attribute', ctypes.c_int),
                            ('Data', ctypes.c_void_p),
                            ('SizeOfData', ctypes.c_size_t)]
            # Convert #RRGGBB to ABGR expected by Windows
            c = accent_color.lstrip('#')
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            a = 0x80  # 50% opacity
            gradient_color = (a << 24) | (b << 16) | (g << 8) | r
            accent = ACCENTPOLICY()
            ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
            accent.AccentState = ACCENT_ENABLE_ACRYLICBLURBEHIND
            accent.AccentFlags = 0x20 | 0x40  # blur + gradient
            accent.GradientColor = gradient_color
            accent.AnimationId = 0
            data = WINDOWCOMPOSITIONATTRIBDATA()
            data.Attribute = 19  # WCA_ACCENT_POLICY
            data.Data = ctypes.addressof(accent)
            data.SizeOfData = ctypes.sizeof(accent)
            SetWindowCompositionAttribute(ctypes.c_void_p(hwnd), ctypes.byref(data))
        except Exception:
            pass

    def _apply_gradient_background(self, width: int, height: int, top: str = '#0b132b', bottom: str = '#172554') -> None:
        """Create or update a single gradient background and keep it behind content."""
        try:
            # Convert hex colors to RGB tuples
            def hex_to_rgb(h):
                h = h.lstrip('#')
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            top_color = hex_to_rgb(top)
            bottom_color = hex_to_rgb(bottom)
            width = max(int(width), 1)
            height = max(int(height), 1)
            bg_img = Image.new('RGB', (width, height), top_color)
            # Efficient gradient draw per row
            for y in range(height):
                blend = y / max(height - 1, 1)
                r = int(top_color[0] * (1 - blend) + bottom_color[0] * blend)
                g = int(top_color[1] * (1 - blend) + bottom_color[1] * blend)
                b = int(top_color[2] * (1 - blend) + bottom_color[2] * blend)
                bg_img.paste((r, g, b), (0, y, width, y + 1))
            self._bg_tk = ImageTk.PhotoImage(bg_img)
            if hasattr(self, '_bg_label') and self._bg_label is not None:
                self._bg_label.configure(image=self._bg_tk)
            else:
                self._bg_label = tk.Label(self.root, image=self._bg_tk, borderwidth=0, highlightthickness=0)
                self._bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            # Ensure background is behind all other widgets
            self._bg_label.lower()
        except Exception:
            # Fallback solid background
            self.root.configure(bg=top)

    def add_space(self):
        """Adds a single space and speaks the last completed word."""
        if self.str and self.str[-1] != ' ':
            last_word = self.str.split(' ')[-1]
            self.str += " "
            self._update_display()
            if last_word:
                self._speak(last_word)

    def toggle_speech(self):
        """Toggle speech synthesis on/off"""
        self.speech_enabled = not self.speech_enabled
        if self.speech_enabled:
            self.speech_btn.config(text="🔊 SPEECH ON", bg='#4caf50', activebackground='#388e3c')
        else:
            self.speech_btn.config(text="🔇 SPEECH OFF", bg='#f44336', activebackground='#d32f2f')
            if self.speak_engine:
                self.speak_engine.stop()

    def _update_display(self):
        """Update the display with current text"""
        # Update the text widget
        self.text_widget.config(state='normal')
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, self.str or "Start signing to see text here...")
        self.text_widget.config(state='disabled')
        self.text_widget.see(tk.END)  # Auto-scroll to end
        
        if self.current_symbol != self.last_recognized_char:
            self._highlight_character()
            self.last_recognized_char = self.current_symbol

    def _highlight_character(self):
        """Add visual highlight animation for recognized character"""
        if self.current_symbol and self.current_symbol != " ":
            original_fg = self.panel3.cget('fg')
            original_bg = self.panel3.cget('bg')
            
            # Animation sequence
            self.panel3.config(fg='#ffeb3b', bg='#ff6f00')
            self.root.after(100, lambda: self.panel3.config(fg='#4caf50', bg='#2e7d32'))
            self.root.after(200, lambda: self.panel3.config(fg='#2196f3', bg='#1565c0'))
            self.root.after(300, lambda: self.panel3.config(fg=original_fg, bg=original_bg))

    def video_loop(self):
        """Main video processing loop"""
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                self.root.after(1, self.video_loop)
                return
                
            cv2image = cv2.flip(frame, 1)
            if cv2image is not None and cv2image.size != 0:
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy = np.array(cv2image)
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image)
                
                # Resize image to fit panel
                panel_width = self.panel.winfo_width()
                panel_height = self.panel.winfo_height()
                if panel_width > 1 and panel_height > 1:
                    resized_img = self.current_image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=resized_img)
                    self.panel.imgtk = imgtk
                    self.panel.config(image=imgtk)

                if hands and hands[0]:
                    hand = hands[0]
                    map = hand[0]
                    x, y, w, h = map['bbox']
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                    white = cv2.imread(os.path.join(BASE_DIR, 'white.jpg'))
                    if white is None:
                        white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                    
                    if image is not None and image.size != 0:
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz and handz[0]:
                            hand = handz[0]
                            handmap = hand[0]
                            self.pts = handmap['lmList']

                            x_offset = ((400 - w) // 2) - 15
                            y_offset = ((400 - h) // 2) - 15
                            
                            self._draw_hand_skeleton(white, x_offset, y_offset)
                            
                            res = white
                            self.predict(res)

                            # Update skeleton display with proper sizing
                            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                            skeleton_img = Image.fromarray(res_rgb)
                            
                            # Get panel dimensions and resize accordingly
                            panel_width = self.panel2.winfo_width()
                            panel_height = self.panel2.winfo_height()
                            if panel_width > 1 and panel_height > 1:
                                # Maintain aspect ratio
                                img_ratio = skeleton_img.width / skeleton_img.height
                                panel_ratio = panel_width / panel_height
                                
                                if img_ratio > panel_ratio:
                                    new_width = panel_width
                                    new_height = int(panel_width / img_ratio)
                                else:
                                    new_height = panel_height
                                    new_width = int(panel_height * img_ratio)
                                
                                resized_skeleton = skeleton_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                imgtk_skeleton = ImageTk.PhotoImage(image=resized_skeleton)
                                self.panel2.imgtk = imgtk_skeleton
                                self.panel2.config(image=imgtk_skeleton)
                            
                            self.panel3.config(text=self.current_symbol, font=('Segoe UI', 42, 'bold'))
                            self._update_display()

                            # Update suggestion buttons
                            self.b1.config(text=self.word1 if self.word1.strip() else "")
                            self.b2.config(text=self.word2 if self.word2.strip() else "")
                            self.b3.config(text=self.word3 if self.word3.strip() else "")
                            self.b4.config(text=self.word4 if self.word4.strip() else "")

        except Exception as e:
            if DEBUG:
                print(f"Video loop error: {e}")
                traceback.print_exc()
        finally:
            self.root.after(10, self.video_loop)

    def _draw_hand_skeleton(self, white, x_offset, y_offset):
        """Draw hand skeleton with enhanced visualization"""
        try:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
                (5, 6), (6, 7), (7, 8),           # Index
                (9, 10), (10, 11), (11, 12),       # Middle
                (13, 14), (14, 15), (15, 16),     # Ring
                (17, 18), (18, 19), (19, 20),     # Pinky
                (5, 9), (9, 13), (13, 17),        # Palm connections
                (0, 5), (0, 17)                   # Wrist connections
            ]
            
            for start, end in connections:
                if (0 <= start < len(self.pts) and 0 <= end < len(self.pts) and
                    self.pts[start] and self.pts[end]):
                    cv2.line(white, 
                            (int(self.pts[start][0] + x_offset), int(self.pts[start][1] + y_offset)),
                            (int(self.pts[end][0] + x_offset), int(self.pts[end][1] + y_offset)),
                            (0, 255, 0), 4)
            
            for i in range(min(21, len(self.pts))):
                if self.pts[i]:
                    cv2.circle(white, 
                              (int(self.pts[i][0] + x_offset), int(self.pts[i][1] + y_offset)), 
                              5, (0, 0, 255), -1)
                    cv2.circle(white, 
                              (int(self.pts[i][0] + x_offset), int(self.pts[i][1] + y_offset)), 
                              6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Skeleton drawing error: {e}")

    def distance(self, x, y):
        """Calculate distance between two points"""
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self): self._replace_word(self.word1)
    def action2(self): self._replace_word(self.word2)
    def action3(self): self._replace_word(self.word3)
    def action4(self): self._replace_word(self.word4)

    def _replace_word(self, new_word):
        """Replace the current word with a new word"""
        if new_word.strip():
            if " " in self.str:
                last_space_index = self.str.rfind(" ")
                self.str = self.str[:last_space_index + 1] + new_word.upper()
            else:
                self.str = new_word.upper()
            
            self._update_display()
            self._speak(new_word)

    def speak_fun(self):
        """Speak the current text."""
        text_to_speak = self.str.strip()
        print(f"Speak button clicked. Speaking: '{text_to_speak}'")
        self.status_label.config(text="Speaking...", fg='#ffeb3b')
        self.root.update()
        
        self._speak(text_to_speak)
        self.root.after(1500, lambda: self.status_label.config(text="System Ready", fg='#4fc3f7'))

    def clear_fun(self):
        """Clear all text"""
        self.str = ""
        self.word1, self.word2, self.word3, self.word4 = " ", " ", " ", " "
        self.word = " "
        self.last_checked_word = ""
        self._update_display()
        self.b1.config(text="")
        self.b2.config(text="")
        self.b3.config(text="")
        self.b4.config(text="")
        self.status_label.config(text="Text cleared - Ready for new input", fg='#4fc3f7')

    def backspace_fun(self):
        """Removes the last character from the text output."""
        if self.str:
            self.str = self.str[:-1]
            self._update_display()
            self.status_label.config(text="Backspace - Last character removed", fg='#ffeb3b')
        else:
            self.status_label.config(text="Nothing to backspace", fg='#ef4444')
        self.root.after(1000, lambda: self.status_label.config(text="System Ready", fg='#4fc3f7'))

    def predict(self, test_image):
        """Enhanced prediction with the same logic as original"""
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        if ch1 in ['Next', 'B', 'C', 'H', 'F', 'X']:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'
        
        # Handle character processing
        if ch1 == "next" and self.prev_char != "next":
            char_to_add = self.ten_prev_char[(self.count-1)%10] if len(self.ten_prev_char) > 0 else ''
            
            if char_to_add and char_to_add not in ["next", " "]:
                if char_to_add == "Backspace":
                    self.str = self.str[:-1]
                else:
                    self.str += char_to_add
                    self._speak(char_to_add)

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count%10] = ch1

        # Word suggestions
        if len(self.str.strip()) > 0:
            st = self.str.rfind(" ")
            word = self.str[st+1:].strip()
            self.word = word
            if len(word) > 0 and word != self.last_checked_word:
                self.last_checked_word = word
                suggestions = ddd.suggest(word)
                self.word1 = suggestions[0] if len(suggestions) > 0 else ""
                self.word2 = suggestions[1] if len(suggestions) > 1 else ""
                self.word3 = suggestions[2] if len(suggestions) > 2 else ""
                self.word4 = suggestions[3] if len(suggestions) > 3 else ""
            elif len(word) == 0:
                self.word1, self.word2, self.word3, self.word4 = "", "", "", ""
        
        # Update status based on recognition
        if ch1 and ch1 != " ":
            self.status_label.config(text=f"Recognized: {ch1}", fg='#4caf50')
        else:
            self.status_label.config(text="System Ready - Start signing to begin recognition", fg='#4fc3f7')

    def destructor(self):
        """Clean up resources"""
        print("Shutting down application...")
        self.running = False
        if self.speak_engine:
            try:
                self.speak_engine.stop()
            except:
                pass
        time.sleep(0.2)
        
        try:
            self.root.destroy()
        except:
            pass
        
        try:
            self.vs.release()
        except:
            pass
        
        cv2.destroyAllWindows()
        print("Application shutdown complete.")

if __name__ == "__main__":
    print("Starting Enhanced Sign Language Application...")
    try:
        app = EnhancedSignLanguageApp()
        app.root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()
    finally:
        print("Application terminated.")
