import cv2
import numpy as np
import os
import re
import imageio

# --- 設定區 ---
WINDOW_NAME = 'Auto-Align Professional'
OUTPUT_FOLDER = 'aligned_auto_pro'
TARGET_SIZE = (800, 800)
# -------------

def sort_files_by_date(files):
    def extract_date(f):
        match = re.search(r'(\d{8})', f)
        return int(match.group(1)) if match else 0
    return sorted(files, key=extract_date)

def get_plant_mask(img):
    """
    建立植物的遮罩，避免對齊到土壤或石頭。
    保留：綠色、粉色、紫紅色、黃色 (多肉常見色)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 遮罩 1: 綠色/黃色
    mask1 = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
    # 遮罩 2: 紅色/粉色/紫色 (跨越 180-0 度)
    mask2 = cv2.inRange(hsv, (130, 40, 40), (180, 255, 255))
    mask3 = cv2.inRange(hsv, (0, 40, 40), (20, 255, 255))
    
    mask = mask1 | mask2 | mask3
    # 稍微膨脹一點，確保邊緣有被算進去
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def calculate_auto_alignment(ref_img, curr_img):
    """
    核心演算法：計算從 curr_img 到 ref_img 的變換參數
    回傳: dict {'tx', 'ty', 'rot', 'scale'}
    """
    # 1. 取得遮罩 (只對齊植物)
    mask_ref = get_plant_mask(ref_img)
    mask_curr = get_plant_mask(curr_img)

    # 2. SIFT 特徵點偵測
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_img, mask_ref)
    kp2, des2 = sift.detectAndCompute(curr_img, mask_curr)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        print("特徵點不足，無法自動對齊。")
        return None

    # 3. 特徵匹配 (FLANN)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des2, des1, k=2) # 注意: query=curr, train=ref

    # 4. 篩選優良匹配 (Lowe's ratio test)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        print("匹配點過少。")
        return None

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 5. 計算變換矩陣 (限制為 Rigid: 旋轉+縮放+平移，不隨意變形)
    # 使用 estimateAffinePartial2D 比 findHomography 更穩定，因為它鎖定形狀不變
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is None:
        return None

    # 6. 【逆向工程】將矩陣 M 分解回我們的參數 (tx, ty, rot, scale)
    # M = [[a, -b, tx_cv],
    #      [b,  a, ty_cv]]
    # scale = sqrt(a^2 + b^2)
    # rot = atan2(b, a)
    
    a = M[0, 0]
    b = M[1, 0]
    
    scale = np.sqrt(a**2 + b**2)
    angle_rad = np.arctan2(b, a)
    angle_deg = np.degrees(angle_rad)

    # 計算位移：這一步最難，因為 OpenCV 的位移是基於左上角(0,0)，
    # 但我們的工具是基於「中心點旋轉」後再位移。
    # 公式推導： T_slider = (M * Center) - Center
    h, w = ref_img.shape[:2]
    center = np.array([w/2, h/2, 1.0])
    
    # 補成 3x3 矩陣方便運算
    M_3x3 = np.vstack([M, [0, 0, 1]])
    new_center = M_3x3.dot(center)
    
    tx = new_center[0] - center[0]
    ty = new_center[1] - center[1]

    # 透視 (Perspective) 比較難自動估計準確，通常設為 0 讓用戶微調
    return {
        'scale': scale,
        'rot': angle_deg,
        'tx': tx,
        'ty': ty,
        'px': 0, 
        'py': 0
    }

def get_perspective_matrix(img_w, img_h, state):
    angle = state['rot']
    scale = state['scale']
    tx = state['tx']
    ty = state['ty']
    px = state['px']
    py = state['py']

    src_pts = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])
    center_x, center_y = img_w // 2, img_h // 2
    pts = src_pts - [center_x, center_y]
    
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = np.dot(pts, rot_mat.T) * scale
    
    pts[0] += [-px, -py]
    pts[1] += [px, -py]
    pts[2] += [px, py]
    pts[3] += [-px, py]

    target_cx, target_cy = TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2
    pts[:, 0] += target_cx + tx
    pts[:, 1] += target_cy + ty
    
    return cv2.getPerspectiveTransform(src_pts, np.float32(pts))

def crop_center_square(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    return cv2.resize(cropped, TARGET_SIZE)

def draw_hud(img, state, auto_active):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (350, 300), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"Mode: {'AUTO-ALIGNED' if auto_active else 'Manual'}",
        f"------------------",
        f"[Arrows] Pos : {state['tx']:.1f}, {state['ty']:.1f}",
        f"[Q/E]    Rot : {state['rot']:.2f}",
        f"[W/S]    Scl : {state['scale']:.3f}",
        f"[A/D]    PrsX: {state['px']}",
        f"[Z/C]    PrsY: {state['py']}",
        f"------------------",
        f"[Space] Confirm | [R] Reset",
        f"[M] Re-Run Auto Align"
    ]
    
    for i, line in enumerate(lines):
        color = (100, 255, 100) if "AUTO" in line and i==0 else (200, 200, 200)
        cv2.putText(img, line, (10, 30 + i*25), font, 0.6, color, 1)
    return img

def main():
    folder = './photos'
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png')) and 'aligned' not in f]
    sorted_files = sort_files_by_date(files)
    
    if not sorted_files: return
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    # Base Image
    base_img = cv2.imread(os.path.join(folder, sorted_files[0]))
    base_img = crop_center_square(base_img)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"000_{sorted_files[0]}"), base_img)
    
    aligned_images = [cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)]
    prev_img = base_img.copy()

    for i in range(1, len(sorted_files)):
        filename = sorted_files[i]
        raw = cv2.imread(os.path.join(folder, filename))
        if raw is None: continue
        current_img = crop_center_square(raw)
        h, w = current_img.shape[:2]

        # --- 啟動時直接嘗試自動對齊 ---
        print(f"正在自動分析第 {i+1} 張: {filename} ...")
        auto_params = calculate_auto_alignment(prev_img, current_img)
        
        if auto_params:
            state = auto_params # 直接套用算出來的數值
            auto_active = True
            print(f"-> 自動對齊成功: Rot={state['rot']:.2f}, Scale={state['scale']:.2f}")
        else:
            state = {'tx':0, 'ty':0, 'rot':0.0, 'scale':1.0, 'px':0, 'py':0}
            auto_active = False
            print("-> 自動對齊失敗 (特徵不足)，切換回手動預設值。")

        while True:
            M = get_perspective_matrix(w, h, state)
            transformed = cv2.warpPerspective(current_img, M, TARGET_SIZE, borderValue=(30, 30, 30))
            display = cv2.addWeighted(prev_img, 0.5, transformed, 0.5, 0)
            display = draw_hud(display, state, auto_active)
            cv2.imshow(WINDOW_NAME, display)

            k = cv2.waitKeyEx(0)
            
            # 微調的精度 (Step)
            step_move = 1    # 位移 1 pixel
            step_rot = 0.1   # 旋轉 0.1 度
            step_scale = 0.005 # 縮放 0.5%
            step_persp = 1

            if k == 27: 
                cv2.destroyAllWindows()
                return
            elif k == 32 or k == 13: # Space/Enter
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{i:03d}_{filename}"), transformed)
                aligned_images.append(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
                prev_img = transformed.copy()
                break
            
            # [M] 手動重新觸發自動對齊
            elif k == ord('m'):
                p = calculate_auto_alignment(prev_img, current_img)
                if p: state, auto_active = p, True

            # [R] Reset
            elif k == ord('r'):
                state = {'tx':0, 'ty':0, 'rot':0.0, 'scale':1.0, 'px':0, 'py':0}
                auto_active = False

            # 鍵盤控制區
            elif k in [2490368, 65362, 0]: state['ty'] -= step_move # Up
            elif k in [2621440, 65364, 1]: state['ty'] += step_move # Down
            elif k in [2424832, 65361, 2]: state['tx'] -= step_move # Left
            elif k in [2555904, 65363, 3]: state['tx'] += step_move # Right
            elif k == ord('q'): state['rot'] -= step_rot
            elif k == ord('e'): state['rot'] += step_rot
            elif k == ord('w'): state['scale'] += step_scale
            elif k == ord('s'): state['scale'] -= step_scale
            elif k == ord('a'): state['px'] -= step_persp
            elif k == ord('d'): state['px'] += step_persp
            elif k == ord('z'): state['py'] -= step_persp
            elif k == ord('c'): state['py'] += step_persp
            
            # 只要動了按鍵，就取消 "AUTO" 標籤
            if k not in [ord('m')]: auto_active = False

    cv2.destroyAllWindows()
    if aligned_images:
        imageio.mimsave('auto_aligned_succulent.gif', aligned_images, duration=500, loop=0)

if __name__ == "__main__":
    main()