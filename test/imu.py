from donkeycar.parts.imu import Mpu6050

imu1 = Mpu6050(0x68)
imu2 = Mpu6050(0x69)

while(True):
    ax1, ay1, az1, gx1, gy1, gy1, temp1 = imu1.run()
    ax2, ay2, az2, gx2, gy2, gy2, temp2 = imu2.run()

    print("%2f %2f %2f %2f %2f %2f %2f %2f %2f %2f %2f %2f"%(ax1, ay1, az1, gx1, gy1, gy1, ax2, ay2, az2, gx2, gy2, gy2))
