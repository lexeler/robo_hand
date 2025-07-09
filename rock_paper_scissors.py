import cv2
import mediapipe as mp
import time
import math
import serial
import random
import keyboard

HAND_WORKS=True
try:
    ser = serial.Serial(port='/dev/ttyACM0',  baudrate=9600,timeout=1)
except:
    HAND_WORKS=False

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

# Функция для вычисления угла между тремя точками (в градусах)
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    mag_ba = math.sqrt(sum(coord**2 for coord in ba))
    mag_bc = math.sqrt(sum(coord**2 for coord in bc))
    if mag_ba * mag_bc == 0:
        return 0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))

def get_diff(a,b):
    c=[]
    for i,x in enumerate(a):
        c.append(abs(x-b[i])/180)
    return sum(c)

rock1=[60,180,180,180,180]
rock2=[60,140,140,140,140]
paper=[180,0,0,0,0]
scissors=[60,0,0,180,180]
scissors2=[60,0,0,90,90]
signs=[rock1,paper,scissors,scissors2,rock2]
addit_dict={'rock':rock1,'paper':paper,'scissors':scissors}

def get_hand_gesture(angles):
    diffs=[get_diff(i, angles) for i in signs] 
    index=diffs.index(min(diffs))
    player,robot,diff = ['rock','paper','scissors','scissors2','rock2'][index], [paper,scissors,rock1,rock1,paper][index], min(diffs)
    if (player=='scissors' or player=='scissors2') and angles[0]>90:
        return 'fikciki',[-1,-1,-1,-1,-1], angles[0]
    return player,robot,diff

def is_hand_like(angles):
    like = [60,0,0,0,0]
    diff = get_diff(like, angles)
    return diff

# Словарь индексов суставов для четырёх пальцев
fingers = {
    'Big': (1,2,3,4),
    'Index':  (5, 6, 7, 8),
    'Middle': (9, 10, 11, 12),
    'Ring':   (13, 14, 15, 16),
    'Pinky':  (17, 18, 19, 20)
}

last_time=time.time()
timer=0
phase=0
is_random=1
won_already=False

def toggle(_):
    global is_random
    is_random = not is_random

keyboard.on_release_key('tab', toggle)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось получить кадр с камеры.")
            break
        
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        frame.flags.writeable = True
        height, width, _ = frame.shape

        if not is_random:
            cv2.putText(
                frame, "Z", (630, 1),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2
            )
        if (phase==1 or (phase==0 and not won_already)) and HAND_WORKS and ser.in_waiting>0:
            ser.readline()
            gesture=[60,0,0,0,0]
            ser.write('\n'.join(map(str,gesture)).encode('utf-8'))

        if phase==1:
                print("PHASE 1")
                timer+=time.time()-last_time
                last_time=time.time()
                cv2.putText(
                        frame, f"STARTS IN {math.ceil(3-timer)}", (220,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2
                    )
                print("PHASE 1", timer)
                print()
                if timer>3:
                    phase=2
                    timer=0
        
        if results.multi_hand_landmarks:
            extensions = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Рисуем скелет руки
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

                # Для каждого пальца вычисляем степень разгибания и рисуем текст
                for name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():
                    mcp = landmarks[mcp_i]
                    pip = landmarks[pip_i]
                    dip = landmarks[dip_i]
                    tip = landmarks[tip_i]

                    # Вычисляем угол в суставе PIP
                    angle = calculate_angle(
                        (mcp.x, mcp.y, mcp.z),
                        (pip.x, pip.y, pip.z),
                        (dip.x, dip.y, dip.z)
                    )
                    # Нормализация: 60° = 0, 180° = 1
                    min_angle, max_angle = 60.0, 180.0
                    angle = max(min_angle, min(max_angle, angle))
                    extension = (angle - min_angle) / (max_angle - min_angle)

                    # Позиционируем текст над кончиком пальца
                    x_px = int(tip.x * width)
                    y_px = int(tip.y * height)
                    if name=='Big':
                        extensions.append(round(int( (extension-0.5) *2 * 180)/15)*15)
                        if extensions[-1] < 60:
                            extensions[-1] = 60
                    else:
                        extensions.append(int((1-extension)*180))
                    text = f"{extension:.2f}"
                    cv2.putText(
                        frame, text, (x_px - 10, y_px - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2
                    )
            if is_hand_like(extensions) < 0.5 and phase==0:
                timer=0
                last_time=time.time()
                phase=1
                #is_random=not bool(random.randint(0,1))

            if HAND_WORKS and ser.in_waiting > 0 and phase==2:
                player,robot,diff=get_hand_gesture(extensions)
                diff=round(diff,2)
                if not is_random:
                    line = ser.readline().decode('utf-8').rstrip()
                    print(line)
                    print(extensions)
                    data='\n'.join(map(str,robot))
                    print(player,robot,diff)
                    print("=="*20)
                    if diff<1 and player!='fikciki':
                        ser.write(data.encode('utf-8'))
                    else:
                        ser.write('\n'.join(map(str,line)).encode('utf-8'))
                timer+=time.time()-last_time
                last_time=time.time()
                if timer>0.5:
                    phase=0
                    won_already=True
                    timer=0
                    if is_random:
                        num=random.randint(0,2)
                        random_sign=signs[num]
                        ser.readline()
                        print("SELECTED",['rock','paper','scissors'][num])
                        ser.write('\n'.join(map(str,random_sign)).encode('utf-8'))
        if won_already and phase==0:
            if diff<1:
                if not is_random or (random_sign is None):
                    cv2.putText(
                        frame, "ROBO-HAND WON!", (150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
                    )
                else:
                    if random_sign==robot:
                        cv2.putText(
                            frame, "ROBO-HAND WON!", (150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
                        )
                    elif random_sign==addit_dict[player]:
                        cv2.putText(
                            frame, "TIE!", (290, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
                        )
                    else:
                        cv2.putText(
                            frame, "YOU WON!", (220, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
                        )
                cv2.putText(
                    frame, "SHOW GESTURE TO PLAY AGAIN", (80, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                )
            else:
                cv2.putText(
                    frame, "WTF YOU SHOWED?", (140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2
                )
        if True:
            frame = cv2.resize(frame, (1440, 1080), interpolation=cv2.INTER_LINEAR)
            cv2.namedWindow('Fingers Extension', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Fingers Extension', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Fingers Extension', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
