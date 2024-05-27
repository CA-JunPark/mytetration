"""
Author: CA-JunPark
email: cskoko5786@gmail.com
GitHub: https://github.com/CA-JunPark/mytetration 

Numba 설치 설명 은 깃허브 페이지(README.md)에 추가 되어있습니다. 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import multiprocessing as mp
from numba import jit, cuda

# 추가된 parameters
num_processes = mp.cpu_count()//2   # multiprocessing에 사용할 cpu 수, 높을수록 빨라지나
                                    # 너무 높은 수를 적으면 CPU와 RAM이 감당하지 못해 프로그램이 꺼질 수 있음
max_task = 1000                     # 한번에 시킬 task 수 n번 이후 사용된 cpu재시작
                                    # 너무 높은 수를 적으면 CPU와 RAM이 감당하지 못해 프로그램이 꺼질 수 있음
ratio = 4/5                         # 비율 ex) 1, 4/5 (instagram), 9/16, 16/9(Youtube or PPT)
rotate = True                       # 이미지 데이터 회전 True = 회전o False = 회전x
np.seterr(over='ignore')            # overflow warning 무시하기

#parameters - plot영역설정관련
# (x0,y0) : plot영역 중심좌표
x0 = -0.712
y0 = 0
eps = 0.25               #x0 좌우로 eps만큼 plot함
eps_y = eps * ratio      # 비율에 맞추기 위해 y축 eps 계산
n = 3840                 # 화소수조절을 위한 parameter (3840:4K, 1920:Full HD)
nx, ny = n, int(n*ratio) #nx, ny : x,y축 화소수

#parameters - tetration계산 관련
max_iter = 500 #최대 몇층까지 계산할 것인지를 정함. max_iter층 만큼 계산했는데 복소수 크기가 escape_radius를 벗어나지 않으면 수렴한것으로 처리.
escape_radius = 1e+10 #복소수크기가 escape_radius를 벗어나면 발산한 것으로 처리함.

x = np.linspace(x0 - eps, x0 + eps, nx)      # x 좌표
y = np.linspace(y0 - eps_y, y0 + eps_y, ny)  # y 좌표
c = x[:, np.newaxis] + 1j * y[np.newaxis, :] # 복소 좌표

# 좌표들 
ijs = []
for i in range(nx):
    for j in range(ny):
        ijs.append((i,j))

@jit(target_backend='cuda')
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
    if rotate:
        rotated_map = np.rot90(divergence_map, k=-1)
    # plot
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["black", "white"]) # 커스텀 컬러맵 생성: 발산은 흰색, 수렴은 검은색
    plt.imshow(rotated_map.T, extent=[y0 - eps_y, y0 + eps_y, x0 - eps, x0 + eps], origin='lower', cmap=cmap)
    plt.axis('off')  # 축 라벨과 타이틀 제거
    filename = f"mytetration_x_{x0}_y_{y0}_eps_{eps}_rotated.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
