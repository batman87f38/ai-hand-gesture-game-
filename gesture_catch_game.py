"""
Gesture-Controlled Catch Game
-----------------------------
Controls (haath webcam ke saamne rakho):
  - Index finger ki x-position  -> basket left/right move
  - Fist (mutthi)               -> pause
  - Open palm (game over pe)    -> restart
  - Keyboard: R = restart, ESC = quit

Install:
  pip install opencv-python mediapipe pygame  (internet chahiye pehli baar model download ke liye)
"""

import os
import random
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- Settings ----------
W, H = 800, 600
FPS = 60
BASKET_W, BASKET_H = 120, 20
SMOOTHING = 0.35          # 0 se 1; kam = zyada smooth, zyada lag
HAND_MARGIN = 0.15        # camera ke kinaare ignore (poori range easily reach ho)
START_LIVES = 3
CAM_PREVIEW = (200, 150)

# Hand model (pehli baar run par apne aap download hoga, ~8 MB)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

WHITE = (240, 240, 240)
GREEN = (60, 200, 90)
RED = (220, 60, 60)
BG = (20, 24, 38)
BASKET_COLOR = (80, 160, 255)


# ---------- Hand tracking ----------
class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Webcam nahi khuli. Camera check karo.")
            sys.exit(1)
        if not os.path.exists(MODEL_PATH):
            print("Hand model download ho raha hai (sirf pehli baar)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time()
        self.x = 0.5          # smoothed x (0..1)
        self.is_fist = False
        self.is_palm = False
        self.found = False
        self.preview = None

    @staticmethod
    def fingers_up(lm):
        # index, middle, ring, pinky: tip ka y, pip joint ke y se kam => ungli upar
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        return sum(lm[t].y < lm[p].y for t, p in zip(tips, pips))

    def update(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)  # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        self.found = False
        self.is_fist = False
        self.is_palm = False

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            self.found = True

            raw_x = lm[8].x  # index fingertip
            # margin hata ke 0..1 mein remap
            raw_x = (raw_x - HAND_MARGIN) / (1 - 2 * HAND_MARGIN)
            raw_x = max(0.0, min(1.0, raw_x))
            self.x += (raw_x - self.x) * SMOOTHING  # exponential moving average

            count = self.fingers_up(lm)
            self.is_fist = count == 0
            self.is_palm = count >= 4

            h, w, _ = rgb.shape
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(rgb, pts[a], pts[b], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(rgb, pt, 4, (255, 0, 0), -1)

        small = cv2.resize(rgb, CAM_PREVIEW)
        self.preview = pygame.surfarray.make_surface(small.swapaxes(0, 1))

    def release(self):
        self.cap.release()
        self.landmarker.close()


# ---------- Game objects ----------
class Item:
    def __init__(self, speed):
        self.is_bomb = random.random() < 0.25
        self.r = 16
        self.x = random.randint(self.r, W - self.r)
        self.y = -self.r
        self.speed = speed + random.uniform(-0.5, 1.0)

    def update(self):
        self.y += self.speed

    def draw(self, surf):
        color = RED if self.is_bomb else GREEN
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), self.r)
        if self.is_bomb:
            pygame.draw.circle(surf, (0, 0, 0), (int(self.x), int(self.y)), self.r, 3)

    def hits(self, basket):
        return basket.collidepoint(self.x, self.y + self.r)


class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = START_LIVES
        self.items = []
        self.spawn_timer = 0
        self.over = False

    def difficulty(self):
        return 3 + self.score * 0.15  # score badhne par speed badhti hai

    def update(self, basket):
        self.spawn_timer += 1
        spawn_gap = max(20, 60 - self.score)
        if self.spawn_timer >= spawn_gap:
            self.spawn_timer = 0
            self.items.append(Item(self.difficulty()))

        for item in self.items[:]:
            item.update()
            if item.hits(basket):
                if item.is_bomb:
                    self.lives -= 1
                else:
                    self.score += 1
                self.items.remove(item)
            elif item.y - item.r > H:
                if not item.is_bomb:   # green miss = life gayi
                    self.lives -= 1
                self.items.remove(item)

        if self.lives <= 0:
            self.over = True


# ---------- Main ----------
def draw_text(surf, font, text, pos, color=WHITE, center=False):
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surf.blit(img, rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Gesture Catch Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 26)
    big = pygame.font.SysFont("arial", 56, bold=True)

    tracker = HandTracker()
    game = Game()
    best = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()

        tracker.update()

        basket_x = tracker.x * (W - BASKET_W)
        basket = pygame.Rect(basket_x, H - 50, BASKET_W, BASKET_H)

        paused = tracker.found and tracker.is_fist

        if game.over:
            best = max(best, game.score)
            if tracker.found and tracker.is_palm:
                game.reset()
        elif not paused:
            game.update(basket)

        # ---- draw ----
        screen.fill(BG)
        for item in game.items:
            item.draw(screen)
        pygame.draw.rect(screen, BASKET_COLOR, basket, border_radius=8)

        draw_text(screen, font, f"Score: {game.score}", (15, 10))
        draw_text(screen, font, f"Lives: {'<3 ' * game.lives}", (15, 45), RED)
        draw_text(screen, font, f"Best: {best}", (15, 80))

        if tracker.preview:
            screen.blit(tracker.preview, (W - CAM_PREVIEW[0] - 10, 10))

        if not tracker.found and not game.over:
            draw_text(screen, font, "Haath camera ke saamne laao", (W // 2, 140),
                      (255, 200, 80), center=True)
        if paused:
            draw_text(screen, big, "PAUSED", (W // 2, H // 2), center=True)
        if game.over:
            draw_text(screen, big, "GAME OVER", (W // 2, H // 2 - 40), RED, center=True)
            draw_text(screen, font, f"Score: {game.score}", (W // 2, H // 2 + 10), center=True)
            draw_text(screen, font, "Open palm dikhao ya R dabao restart ke liye",
                      (W // 2, H // 2 + 50), center=True)

        pygame.display.flip()
        clock.tick(FPS)

    tracker.release()
    pygame.quit()


if __name__ == "__main__":
    main()
