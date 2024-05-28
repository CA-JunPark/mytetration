## Numba & CUDA

GPU를 사용해서 더 빠르게 계산할 수 있습니다.
https://www.geeksforgeeks.org/running-python-script-on-gpu/ 참고

**Numba 설치**

##### 주의 1: NVIDIA GPU만 가능 사용 가능합니다.
##### 주의 2: RAM 사용량에 주의 (n값이 높을 수록 RAM 사용량이 치솟습니다.)

사용할 수 있는 GPU는 [이곳](https://developer.nvidia.com/cuda-gpus)에서 확인할 수 있습니다.
명시된 GPU를 사용 중이라면 

- [NVIDIA 공식 사이트](https://developer.nvidia.com/cuda-downloads)에서 CUDA Toolkit (Linux 또는 Windows) 을 다운 받고 설치합니다.

- 다음 명령어를 터미널에 입력하여 Numba를 설치합니다 : `pip3 install numba`

## PTF_static_mp_gpu.py (속도 향상된 코드)
AMD Ryzen 7 7840HS CPU와 NVIDIA GeForce RTX 4060 Laptop GPU, 16GB RAM으로 PTF_static_mp_gpu.py를 
'num_processes=8, max_task=1000, n=4000, x0 = -1.012, y0 = 0, eps = 0.03125 '으로 설정 후, 실행 시, 이미지가 약 1분 30초 만에 출력되었습니다. (약 30배)
(원본 코드 `PTF_static_4by5_R.py`는 약 45분 걸렸습니다)
RAM을 차지하는 다른 프로그램 (크롬, 디스코드 등)을 끌수록 약간의 속도 향상이 체감되었습니다.

- 같은 (x0, y0)로 ratio=16/9 사용 시 약 5~6분까지 소요 시간이 늘었습니다. 출력될 화면에 존재하는 발산 값인 포인트가 많을수록 필요한 계산 횟수가 줄어드는 게 원인이지 않을까 추측됩니다. 

- 명시된 GPU를 사용 중이 아니면 `tetration()`함수 바로 위에 적힌 `@jit(target_backend='cuda')`를 지우면 GPU를 사용하지 않습니다. 
(그래도 병렬 연산은 하니 CPU에 따라 기존 코드보단 어느정도 빨라지긴 합니다)

## 추가된 파라미터들 
`num_processes, max_task, ratio, rotate`의 설명은 코드 내의 주석 참고

## tqdm (진행 표시줄) 사용하지 않은 이유
n이 높을수록 성능저하가 심하게 느껴졌습니다. (1분30초걸리던 게 5분 이상)