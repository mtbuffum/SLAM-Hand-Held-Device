
//----------------NOTE THIS SHOULD BE RUN IN ARDUINO IDE---------------------


#include <Wire.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

#define SDA_PIN 21
#define SCL_PIN 19

//Create the object for our IMU library
Adafruit_ICM20948 icm;

void setup() {
  Serial.begin(115200);
  delay(1500);
  //Defines the I2C wires that we will use
  Wire.begin(SDA_PIN, SCL_PIN);
  //Sets the I2C clock that we will use
  Wire.setClock(100000);

  Serial.println("Starting ICM-20948...");

  if (!icm.begin_I2C(0x68, &Wire)) {   // address 0x68 confirmed
    Serial.println("Failed to find ICM20948");
    while (1) delay(1000);
  }

  Serial.println("ICM20948 found!");

  // Optional: set ranges (defaults are fine to start)
  icm.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
  icm.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
  icm.setAccelRateDivisor(0); // fastest
  icm.setGyroRateDivisor(0);  // fastest
}

void loop() {
  sensors_event_t accel, gyro, temp, mag;
  icm.getEvent(&accel, &gyro, &temp, &mag);

  // Adafruit gives:
  // accel in m/s^2, gyro in rad/s already 
  unsigned long t_us = micros();

  //serial prints to our COM3 so that we can read back down on the python script
  Serial.print("{\"t_us\":"); Serial.print(t_us);
  Serial.print(",\"ax\":"); Serial.print(accel.acceleration.x, 6);
  Serial.print(",\"ay\":"); Serial.print(accel.acceleration.y, 6);
  Serial.print(",\"az\":"); Serial.print(accel.acceleration.z, 6);
  Serial.print(",\"gx\":"); Serial.print(gyro.gyro.x, 6);
  Serial.print(",\"gy\":"); Serial.print(gyro.gyro.y, 6);
  Serial.print(",\"gz\":"); Serial.print(gyro.gyro.z, 6);
  Serial.println("}");

  delay(5); 
}
