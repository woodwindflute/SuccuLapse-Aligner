# 🌱 SuccuLapse Aligner (多肉縮時對齊工具)

> **Auto-align & manual fine-tune tool for plant growth timelapses.** > 專為植物生長紀錄設計的縮時攝影對齊工具（支援透視校正）。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

## 📖 Introduction

**SuccuLapse Aligner** is a Python tool designed to fix the "shaky hands" problem in long-term plant growth photography. Unlike standard video stabilization software, this tool is optimized for object-centric alignment (e.g., a growing succulent rosette) over weeks or months.

It combines **Automatic Computer Vision** (SIFT + Homography) with a powerful **Manual "Onion Skin" Interface**, allowing you to correct not just rotation and scale, but also **Perspective Tilt (Keystone)** caused by changing camera angles.

**SuccuLapse Aligner** 是為了解決長期拍攝植物（如多肉植物）時，因手持拍攝導致的角度偏差問題。它結合了 **自動化電腦視覺算法** 與 **洋蔥皮手動微調** 介面，不僅能自動對齊，還能讓用戶手動修正 3D 透視變形（Perspective Warp），製作出完美穩定的生長紀錄縮時影片。

---

## ✨ Features (核心功能)

* **🤖 Auto-Alignment (自動對齊):** Uses SIFT feature matching & RANSAC to automatically calculate the best fit (Rotation, Scale, Translation) for the next frame.
* **🧅 Onion Skinning (洋蔥皮模式):** overlays the previous frame (semi-transparent) so you can see exactly how the images align.
* **📐 Perspective Correction (透視校正):** Unique feature to fix "Keystone" effects. Adjust `Perspective X/Y` to correct top-down or side-to-side camera tilts.
* **🌿 Smart Masking (植物遮罩):** Automatically filters out soil and background noise, focusing alignment only on the plant (green/pink/purple hues).
* **⌨️ HUD & Keyboard Control:** Full keyboard-driven interface with a Heads-Up Display showing real-time transformation values.

---

## 🛠 Installation (安裝教學)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SuccuLapse-Aligner.git
    cd SuccuLapse-Aligner
    ```

2.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy imageio
    ```

3.  **Prepare your photos:**
    * Place your plant photos (JPG/PNG) in the `photos` folder (or change the `folder` path in the script).
    * Files should be named with dates for correct sorting (e.g., `20251208.jpg`, `20251215.jpg`).

---

## 🚀 Usage (使用說明)

Run the script:
```bash
python main.py
