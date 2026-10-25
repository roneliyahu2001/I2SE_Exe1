import sys
import numpy as np
import nrrd
import csv
import cv2

def project_3d_to_plane(M, a, b, c):
    dab = np.linalg.norm(b - a)
    dbc = np.linalg.norm(c - b)
    
    if dab == 0 or dbc == 0:
        return np.zeros((1, 1), dtype=M.dtype)

    v_ab = (b - a) / dab
    v_ac = (c - a) / dab  
    
    E_width = int(np.ceil(dbc))
    E_height = int(np.ceil(dab))
    E = np.zeros((E_width, E_height), dtype=M.dtype)
    
    for y in range(E_height):
        a1 = a + v_ab * y
        a2 = a + v_ac * y
        
        d12 = np.linalg.norm(a2 - a1)
        if d12 == 0:
            continue
        dp = (a2 - a1) / d12
        
        p = a1.copy()
        num_steps = int(np.ceil(d12))
        
        for x1 in range(num_steps + 1):
            px, py, pz = int(round(p[0])), int(round(p[1])), int(round(p[2]))
            
            v = 0
            if (0 <= px < M.shape[0]) and (0 <= py < M.shape[1]) and (0 <= pz < M.shape[2]):
                v = M[px, py, pz]
                
            x = int(round((E_width / 2) - (d12 / 2) + x1))
            if 0 <= x < E_width and 0 <= y < E_height:
                E[x, y] = v
                
            p = p + dp
            
    return E

def load_points_from_csv(csv_path):
    planes = []
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            coords = [float(val) for val in row]
            if len(coords) >= 9:
                a = np.array(coords[0:3])
                b = np.array(coords[3:6])
                c = np.array(coords[6:9])
                planes.append((a, b, c))
    return planes

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python main.py <file.nrrd> <file.csv> <file.mp4>")
        sys.exit(1)
        
    nrrd_path = sys.argv[1]
    csv_path = sys.argv[2]
    mp4_path = sys.argv[3]
    
    print(f"Loading NRRD file: {nrrd_path}...")
    data, header = nrrd.read(nrrd_path)
    
    print(f"Loading CSV points: {csv_path}...")
    planes = load_points_from_csv(csv_path)
    
    if not planes:
        print("No valid planes found in CSV file.")
        sys.exit(1)
        
    print(f"Processing {len(planes)} frames for video...")
    
    first_frame = project_3d_to_plane(data, planes[0][0], planes[0][1], planes[0][2])
    h, w = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (w, h), isColor=False)
    
    for idx, (a, b, c) in enumerate(planes):
        frame = project_3d_to_plane(data, a, b, c)
        
        if frame.shape != (h, w):
            frame = cv2.resize(frame, (w, h))
            
        normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        video_writer.write(normalized_frame)
        
    video_writer.release()
    print(f"Video successfully saved to {mp4_path}")