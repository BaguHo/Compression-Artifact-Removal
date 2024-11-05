import matplotlib.pyplot as plt

# 데이터 설정
QF = [20, 40, 60, 80]
ARCNN = [80, 82, 84, 87]
SwinIR = [78, 84, 86, 87]
BlockCNN = [76, 78, 81, 84]
DMCNN = [72, 74, 75, 80]
FBCNN = [70, 74, 75, 75]
PxT = [82, 83, 86, 90]

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(QF, ARCNN, label="ARCNN", marker='o')
plt.plot(QF, SwinIR, label="SwinIR", marker='o')
plt.plot(QF, BlockCNN, label="BlockCNN", marker='o')
plt.plot(QF, DMCNN, label="DMCNN", marker='o')
plt.plot(QF, FBCNN, label="FBCNN", marker='o')
plt.plot(QF, PxT, label="PxT", marker='o')

# 그래프 설정
plt.title('모델별 QF에 따른 Accuracy 변화')
plt.xlabel('QF (Quality Factor)')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# 그래프 표시
plt.show()