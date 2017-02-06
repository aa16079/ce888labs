from sopel import module
from emo.wdemotions import EmotionDetector

X = EmotionDetector()

Y = [0, 0, 0, 0, 0, 0]
A = [0,0,0,0,0,0]
C = 0

@module.rule('')
def hi(bot, trigger):
    global Y, C, A
    print(trigger, trigger.nick)
    R = X.detect_emotion_in_raw_np(trigger)
    C += 1
    for i in range(len(R)):
        Y[i] += R[i]
        A[i] = Y[i] / C
    print('anger: ' + str(A[0]))
    print('disgust: ' + str(A[1]))
    print('fear: ' + str(A[2]))
    print('joy: ' + str(A[3]))
    print('sadness: ' + str(A[4]))
    print('surprise: ' + str(A[5]))






