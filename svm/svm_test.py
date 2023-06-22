import numpy as np

def race(dataFaceR_whiteExtra, dataFaceR_hispanicExtra, dataFaceR_asianExtra, dataFaceR_blackExtra, dataFaceR_otherExtra, dataFaceS_whiteExtra, dataFaceS_hispanicExtra, dataFaceS_asianExtra, dataFaceS_blackExtra, dataFaceS_otherExtra):
    dataFaceR_race = np.vstack((dataFaceR_whiteExtra, dataFaceR_hispanicExtra, dataFaceR_asianExtra, dataFaceR_blackExtra, dataFaceR_otherExtra))
    dataFaceR_raceKnn = dataFaceR_race.T
    dataFaceR_race = dataFaceR_race.real
    RraceRow, RraceCol = dataFaceR_race.shape

    dataFaceS_race = np.vstack((dataFaceS_whiteExtra, dataFaceS_hispanicExtra, dataFaceS_asianExtra, dataFaceS_blackExtra, dataFaceS_otherExtra))
    dataFaceS_raceKnn = dataFaceS_race.T
    dataFaceS_race = dataFaceS_race.real
    SraceRow, SraceCol = dataFaceS_race.shape

    whiteRow, whiteCol = dataFaceR_whiteExtra.shape
    hispanicRow, hispanicCol = dataFaceR_hispanicExtra.shape
    asianRow, asianCol = dataFaceR_asianExtra.shape
    blackRow, blackCol = dataFaceR_blackExtra.shape
    otherRow, otherCol = dataFaceR_otherExtra.shape

    whiteRowS, whiteColS = dataFaceS_whiteExtra.shape
    hispanicRowS, hispanicColS = dataFaceS_hispanicExtra.shape
    asianRowS, asianColS = dataFaceS_asianExtra.shape
    blackRowS, blackColS = dataFaceS_blackExtra.shape
    otherRowS, otherColS = dataFaceS_otherExtra.shape

    lable_whiteR = np.zeros((RraceRow, 1))
    lable_hispanicR = np.zeros((RraceRow, 1))
    lable_asianR = np.zeros((RraceRow, 1))
    lable_blackR = np.zeros((RraceRow, 1))
    lable_otherR = np.zeros((RraceRow, 1))
    lable_raceR = np.zeros((RraceRow, 1))

    lable_whiteS = np.zeros((SraceRow, 1))
    lable_hispanicS = np.zeros((SraceRow, 1))
    lable_asianS = np.zeros((SraceRow, 1))
    lable_blackS = np.zeros((SraceRow, 1))
    lable_otherS = np.zeros((SraceRow, 1))
    lable_raceS = np.zeros((SraceRow, 1))

    for i in range(1, RraceRow+1):
        if i <= whiteRow:
            lable_whiteR[i-1, 0] = 1
        elif i <= RraceRow:
            lable_whiteR[i-1, 0] = 6

    for i in range(1, RraceRow+1):
        if i <= whiteRow:
            lable_hispanicR[i-1, 0] = 6
        elif i <= hispanicRow + whiteRow:
            lable_hispanicR[i-1, 0] = 2
        elif i <= RraceRow:
            lable_hispanicR[i-1, 0]
    for i in range(1, RraceRow + 1):
        if i <= whiteRow + hispanicRow:
            lable_asianR[i - 1, 0] = 6
        elif i <= whiteRow + hispanicRow + asianRow:
            lable_asianR[i - 1, 0] = 3
        elif i <= RraceRow:
            lable_asianR[i - 1, 0] = 6

    for i in range(1, RraceRow + 1):
        if i <= whiteRow + hispanicRow + asianRow:
            lable_blackR[i - 1, 0] = 6
        elif i <= whiteRow + hispanicRow + asianRow + blackRow:
            lable_blackR[i - 1, 0] = 4
        elif i <= RraceRow:
            lable_blackR[i - 1, 0] = 6

    for i in range(1, RraceRow + 1):
        if i <= whiteRow + hispanicRow + asianRow + blackRow:
            lable_otherR[i - 1, 0] = 6
        elif i <= RraceRow:
            lable_otherR[i - 1, 0] = 5

    for i in range(1, RraceRow + 1):
        if i <= whiteRow:
            lable_raceR[i - 1, 0] = 1
        elif i <= hispanicRow + whiteRow:
            lable_raceR[i - 1, 0] = 2
        elif i <= whiteRow + hispanicRow + asianRow:
            lable_raceR[i - 1, 0] = 3
        elif i <= whiteRow + hispanicRow + asianRow + blackRow:
            lable_raceR[i - 1, 0] = 4
        elif i <= RraceRow:
            lable_raceR[i - 1, 0] = 5

    label_whiteRKnn = lable_whiteR.T
    label_hispanicRKnn = lable_hispanicR.T
    label_asianRKnn = lable_asianR.T
    label_blackRKnn = lable_blackR.T
    label_otherRKnn = lable_otherR.T

    for i in range(1, SraceRow + 1):
        if i <= whiteRowS:
            lable_whiteS[i - 1, 0] = 1
        elif i <= SraceRow:
            lable_whiteS[i - 1, 0] = 6

    for i in range(1, SraceRow + 1):
        if i <= whiteRowS:
            lable_hispanicS[i - 1, 0] = 6
        elif i <= hispanicRowS + whiteRowS:
            lable_hispanicS[i - 1, 0] = 2
        elif i <= SraceRow:
            lable_hispanicS[i - 1, 0] = 6

    for i in range(1, SraceRow + 1):
        if i <= whiteRowS + hispanicRowS:
            lable_asianS[i - 1, 0] = 6
        elif i <= whiteRowS + hispanicRowS + asianRowS:
            lable_asianS[i - 1, 0] = 3
        elif i <= SraceRow:
            lable_asianS[i - 1, 0] = 6

    for i in range(1, SraceRow + 1):
        if i <= whiteRowS + hispanicRowS + asianRowS:
            lable_blackS[i - 1, 0] = 6
        elif i <= whiteRowS + hispanicRowS + asianRowS + blackRowS:
            lable_blackS[i - 1, 0] = 4
        elif i <= SraceRow:
            lable_blackS[i - 1, 0] = 6

        for i in range(1, SraceRow + 1):
            if i <= whiteRowS + hispanicRowS + asianRowS + blackRowS:
                lable_otherS[i - 1, 0] = 6
            elif i <= SraceRow:
                lable_otherS[i - 1, 0] = 5

        for i in range(1, SraceRow + 1):
            if i <= whiteRowS:
                lable_raceS[i - 1, 0] = 1
            elif i <= hispanicRowS + whiteRowS:
                lable_raceS[i - 1, 0] = 2
            elif i <= whiteRowS + hispanicRowS + asianRowS:
                lable_raceS[i - 1, 0] = 3
            elif i <= whiteRowS + hispanicRowS + asianRowS + blackRowS:
                lable_raceS[i - 1, 0] = 4
            elif i <= SraceRow:
                lable_raceS[i - 1, 0] = 5

        label_whiteSKnn = lable_whiteS.T
        label_hispanicSKnn = lable_hispanicS.T
        label_asianSKnn = lable_asianS.T
        label_blackSKnn = lable_blackS.T
        label_otherSKnn = lable_otherS.T

        return dataFaceR_raceKnn, label_whiteRKnn, label_hispanicRKnn, label_asianRKnn, label_blackRKnn, label_otherRKnn, dataFaceS_raceKnn, label_whiteSKnn, label_hispanicSKnn, label_asianSKnn, label_blackSKnn, label_otherSKnn


lambda = 0.1 # 权重衰减参数Weight decay parameter
alpha = 0.5 # 学习速率
MAX_ITR = 3 # 最大迭代次数
X_test = dataFaceS_race
X = dataFaceR_race
label = lable_raceR

t1 = time.process_time()
theta, test_pre, rate = mysoftmax_gd(X_test, X, label, lambda, alpha, MAX_ITR)
t2 = time.process_time()
t = t2 - t1
index_t = np.where(lable_raceS == test_pre) # 找出预测正确的样本的位置
rate_test = len(index_t[0]) / len(lable_raceS) # 计算预测精度

print("t:", t)
print("rate_test:", rate_test)


