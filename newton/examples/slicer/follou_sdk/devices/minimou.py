from enum import IntEnum
import math
from ..base import DeviceController

class Cmd(IntEnum):
    """Command enumeration for MiniMou device controller."""
    RESET = 0
    GET_DEVICE_TYPE = 1
    GET_ANGLES = 2
    GET_POSITION = 3
    GET_ORIENTATION = 4
    GET_VELOCITY = 5
    GET_TOOLPOS_AND_VOLTAGE = 6
    SET_FORCE = 7
    GET_CURRENT_DELTA_T = 8
    GET_STATUS = 9
    SET_MOTOR_TORQUES = 10
    SET_LED_INTENSITY = 11
    SET_COLLISION_OBJECT_TYPE = 12
    SET_COLLISION_OBJECT_ACTIVE = 13
    SET_COLLISION_OBJECT_P0 = 14
    SET_COLLISION_OBJECT_V0 = 15
    SET_COLLISION_OBJECT_N = 16
    SET_COLLISION_OBJECT_Q = 17
    SET_COLLISION_OBJECT_R = 18
    SET_COLLISION_OBJECT_S = 19
    SET_COLLISION_OBJECT_T = 20
    SET_COLLISION_OBJECT_STIFFNESS = 21
    SET_COLLISION_OBJECT_DAMPING = 22
    SET_COLLISION_OBJECT_FRICTION = 23
    GET_LAST_COLLISION_FORCE = 24
    GET_LAST_PWM = 25
    GET_LAST_COLLISION_DATA = 26
    SET_TOOL_JAW_OPENING_ANGLE = 27
    GET_TOOL_JAW_TORQUE = 28
    SET_TOOL_DATA = 29
    GET_RAW_ENCODER_VALUES = 30
    GET_ENCODER_SCALING_VALUES = 31
    GET_MOTOR_SCALING_VALUES = 32
    SET_MANUAL_PWM = 33
    GET_BOARD_TEMP = 34
    GET_BATTERY_VOLTAGE = 35
    GET_CALIBRATION_STATUS = 36
    GET_AMPLIFIERS_STATUS = 37
    SET_POWER_ON_MANUAL = 38
    SET_FAN_ON_MANUAL = 39
    SET_FF_ENABLE = 40
    GET_SERIAL_NUM = 41
    GET_BUILD_DATE = 42
    SET_CHARGE_ENABLE = 43
    GET_PART_TEMPERATURES = 44
    SET_MAX_USB_CHARGE_CURRENT = 45
    GET_USB_CHARGING_CURRENT = 46
    SET_DEADBAND_PWM_WIDTH = 47
    GET_MOTOR_ANG_VEL = 48
    SET_INSTR_PART_DATA = 49
    SET_INSTR_JOINT_ANGLE = 50
    SET_COMM_ANG0 = 51
    GET_COMM_ANG0 = 52
    GET_COMM_CAL_STATE = 53
    SET_COMM_CAL_START = 54
    GET_HANDLE_OPENING_VALUE = 55
    SET_HANDLE_FORCE = 56
    SET_HANDLE_LOOP_GAIN = 57
    GET_HANDLE_OPTO_FORCE = 58
    SET_HANDLE_ZERO_FORCE = 59
    GET_HANDLE_POS_VOLTAGE = 60
    SET_HANDLE_LED = 61
    GET_HANDLE_OPTO_VOLTAGE = 62
    GET_HANDLE_CONNECTION_STATE = 63
    GET_HANDLE_ACTIVITY = 64
    SET_HANDLE_FORCE_OFFSET = 65
    GET_HANDLE_LAST_PWM = 66
    SET_HANDLE_TO_CALIBRATE = 67
    GET_HANDLE_CALIBRATION_STATUS = 68
    SET_HANDLE_MANUAL_PWM = 69
    SET_GRAVITY_COMP_WEIGHT = 70
    GET_PRED_COMPLIANCE = 71
    SET_CURR_CONT_FACTOR = 72
    SET_STABILIZATION_FACTOR = 73
    SET_TUNNEL = 74

class Dof(IntEnum):
    """Degrees of freedom enumeration."""
    M0 = 0
    M1 = 1
    M2 = 2
    YAW = 3
    PITCH = 4
    ROT = 5

class MiniMou(DeviceController):
    """Python wrapper for the MiniMou haptic device controller."""
    
    _MM_DIV = 100000.0  # Scaling factor for millimeter values
    
    def __init__(self, device_id: int):
        super().__init__(device_id, cmd_enum_size=len(Cmd))
        self.get_dt_cmd = Cmd.GET_CURRENT_DELTA_T
        self.device_scale_factor = 1.0  # Can be adjusted as needed
        self._init_matrix()
        self.transfer_matrix()
        
        # Initialize device settings
        self.set_ff_enable(True)
        self.set_dead_band_width(10, 10, 10)
        self.set_gravity_comp_weight(23.0)
        self.set_stabilization_factor(1.0)
        
        self.inserted_instrument_name = ""

    def _init_matrix(self):
        """Initialize command matrix parameters."""
        # Default initialization
        self.divider = [1.0] * len(Cmd)
        self.num_rets = [0] * len(Cmd)
        self.num_sends = [0] * len(Cmd)
        self.update_interval = [0] * len(Cmd)
        
        # Subscription rates
        self.update_interval[Cmd.GET_ANGLES] = 1
        self.update_interval[Cmd.GET_POSITION] = 1
        self.update_interval[Cmd.GET_ORIENTATION] = 1
        self.update_interval[Cmd.GET_VELOCITY] = 1
        self.update_interval[Cmd.GET_TOOLPOS_AND_VOLTAGE] = 1
        self.update_interval[Cmd.GET_CURRENT_DELTA_T] = 61
        self.update_interval[Cmd.GET_STATUS] = 103
        self.update_interval[Cmd.GET_LAST_PWM] = 31
        self.update_interval[Cmd.GET_BOARD_TEMP] = 10001
        self.update_interval[Cmd.GET_BATTERY_VOLTAGE] = 10003
        self.update_interval[Cmd.GET_CALIBRATION_STATUS] = 101
        self.update_interval[Cmd.GET_PRED_COMPLIANCE] = 11
        self.update_interval[Cmd.GET_HANDLE_OPENING_VALUE] = 1
        self.update_interval[Cmd.GET_HANDLE_OPTO_FORCE] = 1
        self.update_interval[Cmd.GET_HANDLE_POS_VOLTAGE] = 1
        self.update_interval[Cmd.GET_HANDLE_OPTO_VOLTAGE] = 1
        self.update_interval[Cmd.GET_HANDLE_CONNECTION_STATE] = 101
        self.update_interval[Cmd.GET_HANDLE_ACTIVITY] = 1
        
        # Return value counts
        self.num_rets[Cmd.RESET] = 1
        self.num_rets[Cmd.GET_DEVICE_TYPE] = 1
        self.num_rets[Cmd.GET_ANGLES] = 6
        self.num_rets[Cmd.GET_POSITION] = 3
        self.num_rets[Cmd.GET_ORIENTATION] = 4
        self.num_rets[Cmd.GET_VELOCITY] = 3
        self.num_rets[Cmd.GET_TOOLPOS_AND_VOLTAGE] = 2
        self.num_rets[Cmd.GET_CURRENT_DELTA_T] = 1
        self.num_rets[Cmd.GET_STATUS] = 1
        self.num_rets[Cmd.GET_LAST_COLLISION_FORCE] = 3
        self.num_rets[Cmd.GET_LAST_PWM] = 3
        self.num_rets[Cmd.GET_LAST_COLLISION_DATA] = 1
        self.num_rets[Cmd.GET_TOOL_JAW_TORQUE] = 1
        self.num_rets[Cmd.GET_RAW_ENCODER_VALUES] = 3
        self.num_rets[Cmd.GET_ENCODER_SCALING_VALUES] = 3
        self.num_rets[Cmd.GET_MOTOR_SCALING_VALUES] = 3
        self.num_rets[Cmd.GET_BOARD_TEMP] = 1
        self.num_rets[Cmd.GET_BATTERY_VOLTAGE] = 1
        self.num_rets[Cmd.GET_CALIBRATION_STATUS] = 3
        self.num_rets[Cmd.GET_AMPLIFIERS_STATUS] = 1
        self.num_rets[Cmd.GET_MOTOR_ANG_VEL] = 3
        self.num_rets[Cmd.GET_SERIAL_NUM] = 1
        self.num_rets[Cmd.GET_BUILD_DATE] = 1
        self.num_rets[Cmd.GET_COMM_ANG0] = 1
        self.num_rets[Cmd.GET_COMM_CAL_STATE] = 1
        self.num_rets[Cmd.GET_HANDLE_OPENING_VALUE] = 1
        self.num_rets[Cmd.GET_HANDLE_OPTO_FORCE] = 1
        self.num_rets[Cmd.GET_HANDLE_POS_VOLTAGE] = 1
        self.num_rets[Cmd.GET_HANDLE_OPTO_VOLTAGE] = 1
        self.num_rets[Cmd.GET_HANDLE_CONNECTION_STATE] = 1
        self.num_rets[Cmd.GET_HANDLE_ACTIVITY] = 1
        self.num_rets[Cmd.GET_HANDLE_LAST_PWM] = 1
        self.num_rets[Cmd.GET_HANDLE_CALIBRATION_STATUS] = 1
        self.num_rets[Cmd.GET_PRED_COMPLIANCE] = 1
        
        # Send value counts
        self.num_sends[Cmd.RESET] = 1
        self.num_sends[Cmd.SET_FORCE] = 3
        self.num_sends[Cmd.SET_MOTOR_TORQUES] = 3
        self.num_sends[Cmd.SET_LED_INTENSITY] = 3
        self.num_sends[Cmd.SET_COLLISION_OBJECT_TYPE] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_ACTIVE] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_P0] = 4
        self.num_sends[Cmd.SET_COLLISION_OBJECT_V0] = 4
        self.num_sends[Cmd.SET_COLLISION_OBJECT_N] = 4
        self.num_sends[Cmd.SET_COLLISION_OBJECT_Q] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_R] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_S] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_T] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_STIFFNESS] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_DAMPING] = 2
        self.num_sends[Cmd.SET_COLLISION_OBJECT_FRICTION] = 2
        self.num_sends[Cmd.SET_TOOL_JAW_OPENING_ANGLE] = 1
        self.num_sends[Cmd.SET_TOOL_DATA] = 4
        self.num_sends[Cmd.SET_MANUAL_PWM] = 3
        self.num_sends[Cmd.SET_POWER_ON_MANUAL] = 1
        self.num_sends[Cmd.SET_FAN_ON_MANUAL] = 1
        self.num_sends[Cmd.SET_FF_ENABLE] = 1
        self.num_sends[Cmd.SET_CHARGE_ENABLE] = 1
        self.num_sends[Cmd.SET_MAX_USB_CHARGE_CURRENT] = 1
        self.num_sends[Cmd.SET_DEADBAND_PWM_WIDTH] = 3
        self.num_sends[Cmd.SET_INSTR_PART_DATA] = 8
        self.num_sends[Cmd.SET_GRAVITY_COMP_WEIGHT] = 1
        self.num_sends[Cmd.SET_CURR_CONT_FACTOR] = 3
        self.num_sends[Cmd.SET_STABILIZATION_FACTOR] = 1
        self.num_sends[Cmd.SET_TUNNEL] = 8
        
        # Scaling factors
        mm_div = self._MM_DIV
        self.divider[Cmd.GET_ANGLES] = mm_div
        self.divider[Cmd.GET_VELOCITY] = mm_div / self.device_scale_factor
        self.divider[Cmd.GET_POSITION] = mm_div / self.device_scale_factor
        self.divider[Cmd.GET_ORIENTATION] = mm_div
        self.divider[Cmd.GET_TOOLPOS_AND_VOLTAGE] = mm_div
        self.divider[Cmd.SET_FORCE] = mm_div
        self.divider[Cmd.GET_CURRENT_DELTA_T] = mm_div
        self.divider[Cmd.SET_MOTOR_TORQUES] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_P0] = mm_div / self.device_scale_factor
        self.divider[Cmd.SET_COLLISION_OBJECT_V0] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_N] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_Q] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_R] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_S] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_T] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_STIFFNESS] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_DAMPING] = mm_div
        self.divider[Cmd.SET_COLLISION_OBJECT_FRICTION] = mm_div
        self.divider[Cmd.GET_LAST_COLLISION_FORCE] = mm_div
        self.divider[Cmd.SET_TOOL_JAW_OPENING_ANGLE] = mm_div
        self.divider[Cmd.GET_TOOL_JAW_TORQUE] = mm_div
        self.divider[Cmd.GET_ENCODER_SCALING_VALUES] = mm_div
        self.divider[Cmd.GET_MOTOR_SCALING_VALUES] = mm_div
        self.divider[Cmd.GET_BOARD_TEMP] = mm_div
        self.divider[Cmd.GET_BATTERY_VOLTAGE] = mm_div
        self.divider[Cmd.GET_PART_TEMPERATURES] = mm_div
        self.divider[Cmd.SET_MAX_USB_CHARGE_CURRENT] = mm_div
        self.divider[Cmd.GET_USB_CHARGING_CURRENT] = mm_div
        self.divider[Cmd.GET_MOTOR_ANG_VEL] = mm_div
        self.divider[Cmd.SET_INSTR_PART_DATA] = mm_div
        self.divider[Cmd.SET_COMM_ANG0] = mm_div
        self.divider[Cmd.GET_COMM_ANG0] = mm_div
        self.divider[Cmd.GET_HANDLE_OPENING_VALUE] = mm_div
        self.divider[Cmd.SET_HANDLE_FORCE] = mm_div
        self.divider[Cmd.SET_HANDLE_LOOP_GAIN] = mm_div
        self.divider[Cmd.GET_HANDLE_OPTO_FORCE] = mm_div
        self.divider[Cmd.GET_HANDLE_POS_VOLTAGE] = mm_div
        self.divider[Cmd.GET_HANDLE_OPTO_VOLTAGE] = mm_div
        self.divider[Cmd.SET_HANDLE_FORCE_OFFSET] = mm_div
        self.divider[Cmd.GET_PRED_COMPLIANCE] = mm_div
        self.divider[Cmd.SET_TUNNEL] = mm_div
        
        # Special dividers
        self.divider[Cmd.GET_CURRENT_DELTA_T] = 100.0  # dt * 100

    # Device properties
    def get_instrument_name(self) -> str:
        return self.inserted_instrument_name
    
    def set_instrument_name(self, name: str):
        self.inserted_instrument_name = name

    # Core device operations
    def reset(self, mode: int = 0):
        self.set_send_val(Cmd.RESET, 0, mode)

    def get_orientation(self) -> tuple:
        """Get orientation as a standard quaternion `(x, y, z, w)`."""
        qx = self.get_return(Cmd.GET_ORIENTATION, 0)
        qy = self.get_return(Cmd.GET_ORIENTATION, 1)
        qz = -self.get_return(Cmd.GET_ORIENTATION, 2)
        qw = -self.get_return(Cmd.GET_ORIENTATION, 3)
        return (qx, qy, qz, qw)
        

    def get_calibration_status(self) -> int:
        return (int(self.get_return(Cmd.GET_CALIBRATION_STATUS, 0)) * 100 + 
                int(self.get_return(Cmd.GET_CALIBRATION_STATUS, 1)) * 10 + 
                int(self.get_return(Cmd.GET_CALIBRATION_STATUS, 2)))

    def get_battery_voltage(self) -> float:
        return self.get_return(Cmd.GET_BATTERY_VOLTAGE, 0)

    def get_position(self) -> tuple:
        """Get position as (x, y, z, 1) in left-handed coordinate system."""
        return (self.get_return(Cmd.GET_POSITION, 0),
                self.get_return(Cmd.GET_POSITION, 2),
                self.get_return(Cmd.GET_POSITION, 1),
                1)

    def get_velocity(self) -> tuple:
        """Get velocity as (x, y, z) in mm/s."""
        return (self.get_return(Cmd.GET_VELOCITY, 0),
                self.get_return(Cmd.GET_VELOCITY, 2),
                self.get_return(Cmd.GET_VELOCITY, 1))

    def get_board_temp(self) -> float:
        return self.get_return(Cmd.GET_BOARD_TEMP, 0)

    def get_last_pwm(self, dof: Dof) -> int:
        return int(self.get_return(Cmd.GET_LAST_PWM, int(dof)))

    def get_power_on_status(self) -> bool:
        status = int(self.get_return(Cmd.GET_STATUS, 0))
        return bool(status & 0x40000)

    # Angle getters
    def get_angle(self, dof: Dof) -> float:
        """Get angle in degrees for specified degree of freedom."""
        return self.get_return(Cmd.GET_ANGLES, int(dof)) * 180 / math.pi

    def get_rot_angle(self) -> float:
        return self.get_angle(Dof.ROT)

    def get_pitch_angle(self) -> float:
        return self.get_angle(Dof.PITCH)

    def get_yaw_angle(self) -> float:
        return self.get_angle(Dof.YAW)

    def get_m0_angle(self) -> float:
        return self.get_angle(Dof.M0)

    def get_m1_angle(self) -> float:
        return self.get_angle(Dof.M1)

    def get_m2_angle(self) -> float:
        return self.get_angle(Dof.M2)

    def get_tool_pos(self) -> float:
        return self.get_return(Cmd.GET_TOOLPOS_AND_VOLTAGE, 0)

    def get_tool_pos_voltage(self) -> float:
        return self.get_return(Cmd.GET_TOOLPOS_AND_VOLTAGE, 1)

    def get_handle_opening_value(self) -> float:
        return self.get_return(Cmd.GET_HANDLE_OPENING_VALUE, 0)

    def get_handle_activity(self) -> int:
        return int(self.get_return(Cmd.GET_HANDLE_ACTIVITY, 0))

    def get_handle_pos_voltage(self) -> float:
        return self.get_return(Cmd.GET_HANDLE_POS_VOLTAGE, 0)

    def get_handle_opto_voltage(self) -> float:
        return self.get_return(Cmd.GET_HANDLE_OPTO_VOLTAGE, 0)

    def get_handle_connection_state(self) -> int:
        return int(self.get_return(Cmd.GET_HANDLE_CONNECTION_STATE, 0))

    def get_calibration_status_dof(self, dof: int) -> int:
        return int(self.get_return(Cmd.GET_CALIBRATION_STATUS, dof))

    def get_amplifier_status(self) -> int:
        return int(self.get_return(Cmd.GET_AMPLIFIERS_STATUS, 0))

    # Device control commands
    def set_torques(self, m0: float, m1: float, m2: float):
        self.set_send_float(Cmd.SET_MOTOR_TORQUES, int(Dof.M0), m0)
        self.set_send_float(Cmd.SET_MOTOR_TORQUES, int(Dof.M1), m1)
        self.set_send_float(Cmd.SET_MOTOR_TORQUES, int(Dof.M2), m2)

    def set_force(self, fx: float, fy: float, fz: float):
        self.set_send_float(Cmd.SET_FORCE, 0, fx)
        self.set_send_float(Cmd.SET_FORCE, 2, fy)  # Note axis swap
        self.set_send_float(Cmd.SET_FORCE, 1, fz)  # Note axis swap

    def set_opening_ang(self, ang: float):
        self.set_send_float(Cmd.SET_TOOL_JAW_OPENING_ANGLE, 0, ang * math.pi / 180.0)

    def set_dead_band_width(self, m0: int, m1: int, m2: int):
        self.set_send_val(Cmd.SET_DEADBAND_PWM_WIDTH, int(Dof.M0), m0)
        self.set_send_val(Cmd.SET_DEADBAND_PWM_WIDTH, int(Dof.M1), m1)
        self.set_send_val(Cmd.SET_DEADBAND_PWM_WIDTH, int(Dof.M2), m2)

    def set_collision_object_p0(self, obj_index: int, p0: tuple):
        self.set_send_val(Cmd.SET_COLLISION_OBJECT_P0, 0, obj_index)
        self.set_send_float(Cmd.SET_COLLISION_OBJECT_P0, 1, p0[0])
        self.set_send_float(Cmd.SET_COLLISION_OBJECT_P0, 2, p0[2])  # Axis swap
        self.set_send_float(Cmd.SET_COLLISION_OBJECT_P0, 3, p0[1])  # Axis swap

    # Similar implementations for other collision object methods...
    # (set_collision_object_v0, set_collision_object_n, etc.)

    def set_pwm(self, m0: int, m1: int, m2: int):
        self.set_send_val(Cmd.SET_MANUAL_PWM, int(Dof.M0), m0)
        self.set_send_val(Cmd.SET_MANUAL_PWM, int(Dof.M1), m1)
        self.set_send_val(Cmd.SET_MANUAL_PWM, int(Dof.M2), m2)

    def set_power(self, on: bool):
        self.set_send_val(Cmd.SET_POWER_ON_MANUAL, 0, int(on))

    def set_ff_enable(self, on: bool):
        self.set_send_val(Cmd.SET_FF_ENABLE, 0, int(on))

    def set_charge_enable(self, on: bool):
        self.set_send_val(Cmd.SET_CHARGE_ENABLE, 0, int(on))

    def set_gravity_comp_weight(self, weight: float):
        self.set_send_float(Cmd.SET_GRAVITY_COMP_WEIGHT, 0, weight)

    def set_stabilization_factor(self, factor: float):
        self.set_send_float(Cmd.SET_STABILIZATION_FACTOR, 0, factor)

    def set_current_control_factor(self, ccf0: float, ccf1: float, ccf2: float):
        self.set_send_float(Cmd.SET_CURR_CONT_FACTOR, 0, ccf0)
        self.set_send_float(Cmd.SET_CURR_CONT_FACTOR, 1, ccf1)
        self.set_send_float(Cmd.SET_CURR_CONT_FACTOR, 2, ccf2)

    def get_pred_compliance(self) -> float:
        return self.get_return(Cmd.GET_PRED_COMPLIANCE, 0)

    def is_in_cal_position(self) -> bool:
        status = int(self.get_return(Cmd.GET_STATUS, 0))
        return bool(status & 0x10)

    def set_instr_part(self, part_ix: int, p0: tuple, p1: tuple, radius: float):
        self.set_send_val(Cmd.SET_INSTR_PART_DATA, 0, part_ix)
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 1, p0[0] / self.device_scale_factor)
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 2, p0[2] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 3, p0[1] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 4, p1[0] / self.device_scale_factor)
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 5, p1[2] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 6, p1[1] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_INSTR_PART_DATA, 7, radius / self.device_scale_factor)

    def set_tunnel(self, p0: tuple, v0: tuple, stiffness: float, active: bool):
        self.set_send_float(Cmd.SET_TUNNEL, 0, p0[0] / self.device_scale_factor)
        self.set_send_float(Cmd.SET_TUNNEL, 1, p0[2] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_TUNNEL, 2, p0[1] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_TUNNEL, 3, v0[0] / self.device_scale_factor)
        self.set_send_float(Cmd.SET_TUNNEL, 4, v0[2] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_TUNNEL, 5, v0[1] / self.device_scale_factor)  # Axis swap
        self.set_send_float(Cmd.SET_TUNNEL, 6, stiffness / self.device_scale_factor)
        self.set_send_val(Cmd.SET_TUNNEL, 7, int(active))
