#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(60);
  Serial.println("Ready to get!");
  
  servo1.attach(9);
  servo1.write(180);
  servo2.attach(5);
  servo2.write(0);
  servo3.attach(6);
  servo3.write(0);
  servo4.attach(7);
  servo4.write(0);
  servo5.attach(8);
  servo5.write(0);
  
  /*delay(5000000000);
  
  servo5.write(180);
  servo2.write(180);
  servo3.write(180);
  servo4.write(180);
  servo1.write(180);
  
  delay(1000000000000000);*/ 
}

void loop() {
  //servo1.write(0);
  if (Serial.available()){
    
    int(angle1)=Serial.parseInt();
    int(angle2)=Serial.parseInt();
    int(angle3)=Serial.parseInt();
    int(angle4)=Serial.parseInt();
    int(angle5)=Serial.parseInt();

    if (angle1<60) { angle1 = 60; }
    servo1.write(angle1);
    servo2.write(angle2);
    servo3.write(angle3);
    servo4.write(angle4);
    servo5.write(angle5);

    Serial.println("["+String(angle1)+","+String(angle2)+","+String(angle3)+","+String(angle4)+","+String(angle5)+"]");
  }
}
