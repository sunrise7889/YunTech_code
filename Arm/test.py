from xarm.wrapper import XArmAPI


arm = XArmAPI('192.168.1.160')

# 清除錯誤與警告
arm.connect()
arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)
arm.set_mode(0)     # Position control
arm.set_state(0)    # Ready

# 控制移動（六軸角度控制）
arm.set_servo_angle(angle=[30, 0, 0, 60, 0, 90], speed=20, wait=True)

# 斷線
arm.disconnect()
