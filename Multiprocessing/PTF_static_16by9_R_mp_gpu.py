import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import multiprocessing as mp

num_processes = mp.cpu_count() - 8  # multiprocessing에 사용할 cpu 수 
# 코어 수 이하 값 넣는 걸 추천
np.seterr(over='ignore')  # overflow warning 무시하기
max_task = 1000 # 한번에 시킬 task 수 n번 이후 cpu재시작

#parameters - plot영역설정관련
# (x0,y0) : plot영역 중심좌표
x0 = 0.0061
y0 = 3.2219
eps = 5e0 #x0 좌우로 eps만큼 plot함
eps_y = eps * (16/9)  # 16:9 비율에 맞추기 위해 y축 eps 계산
#화소수조절을 위한 parameter (3840:4K, 1920:Full HD) 
n = 3840 
nx, ny = n, int(n*(16/9)) #nx, ny : x,y축 화소수

#parameters - tetration계산 관련
max_iter = 500 #최대 몇층까지 계산할 것인지를 정함. max_iter층 만큼 계산했는데 복소수 크기가 escape_radius를 벗어나지 않으면 수렴한것으로 처리.
escape_radius = 1e+10 #복소수크기가 escape_radius를 벗어나면 발산한 것으로 처리함.

x = np.linspace(x0 - eps, x0 + eps, nx)
y = np.linspace(y0 - eps_y, y0 + eps_y, ny)
c = x[:, np.newaxis] + 1j * y[np.newaxis, :] # complex coordinates 

# 좌표들 
ijs = []
for i in range(nx):
    for j in range(ny):
        ijs.append((i,j))

def tetration(ij):
    global c, max_iter, escape_radius
    i , j = ij
    c_val = c[i,j]
    z = c_val
    
    for k in range(max_iter):
        z = c_val ** z
        if np.abs(z) > escape_radius:
            return True
        
    return False

if __name__ == '__main__':
    #divergence 계산
    pool = mp.Pool(num_processes, maxtasksperchild=max_task)
    result = pool.map(tetration, ijs)
    pool.close()
    pool.join()
    divergence_map = np.array(result).reshape(c.shape)
    
    # 이미지 데이터 회전
    rotated_map = np.rot90(divergence_map, k=-1)
    # plot
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"]) # 커스텀 컬러맵 생성: 발산은 흰색, 수렴은 검은색
    plt.imshow(rotated_map.T, extent=[y0 - eps_y, y0 + eps_y, x0 - eps, x0 + eps], origin='lower', cmap=cmap)
    plt.axis('off')  # 축 라벨과 타이틀 제거
    filename = f"mytetration_x_{x0}_y_{y0}_eps_{eps}_rotated.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
