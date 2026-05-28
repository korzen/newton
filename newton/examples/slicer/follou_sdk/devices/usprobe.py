from enum import IntEnum
import math
from ..base import DeviceController


class Cmd(IntEnum):
    RESET = 0
    GET_DEVICE_TYPE = 1
    GET_POS_ROT = 2            # needle pos (mm) + rotation (rad)
    SET_FORCE = 3
    SET_TIP_FORCE = 4
    SET_TIP_LENGTH = 5
    SET_SHAFT_FORCE = 6
    SET_FORCE_INTERVAL = 7
    GET_CURRENT_DELTA_T = 8
    GET_STATUS = 9
    GET_LAST_PWM = 10
    GET_SERIAL_NUM = 11
    GET_BUILD_DATE = 12
    GET_INCLINO_ANGLES = 13
    GET_INCLINO_ACC = 14
    GET_ROLL_ANGLE = 15
    GET_PITCH_ANGLE = 16


class USProbe(DeviceController):
    """Python wrapper for the Follou US‑Probe haptic device."""

    _MM_DIV = 100000.0

    def __init__(self, device_id: int):
        super().__init__(device_id, cmd_enum_size=len(Cmd))
        self.get_dt_cmd = Cmd.GET_CURRENT_DELTA_T
        self._init_matrix()
        self.transfer_matrix()

    # ------------------------------------------------------------------
    def _init_matrix(self):
        # subscription intervals
        self.update_interval[Cmd.GET_POS_ROT]        = 1
        self.update_interval[Cmd.GET_CURRENT_DELTA_T] = 11
        self.update_interval[Cmd.GET_LAST_PWM]       = 1
        self.update_interval[Cmd.GET_INCLINO_ANGLES] = 1
        self.update_interval[Cmd.GET_INCLINO_ACC]    = 1
        self.update_interval[Cmd.GET_ROLL_ANGLE]     = 1
        self.update_interval[Cmd.GET_PITCH_ANGLE]    = 1

        # return sizes
        self.num_rets[Cmd.GET_DEVICE_TYPE]    = 1
        self.num_rets[Cmd.GET_POS_ROT]        = 2
        self.num_rets[Cmd.GET_CURRENT_DELTA_T]= 1
        self.num_rets[Cmd.GET_STATUS]         = 1
        self.num_rets[Cmd.GET_LAST_PWM]       = 1
        self.num_rets[Cmd.GET_SERIAL_NUM]     = 1
        self.num_rets[Cmd.GET_BUILD_DATE]     = 1
        self.num_rets[Cmd.GET_INCLINO_ANGLES] = 3
        self.num_rets[Cmd.GET_INCLINO_ACC]    = 3
        self.num_rets[Cmd.GET_ROLL_ANGLE]     = 1
        self.num_rets[Cmd.GET_PITCH_ANGLE]    = 1

        # send sizes
        self.num_sends[Cmd.RESET]              = 1
        self.num_sends[Cmd.SET_FORCE]          = 1
        self.num_sends[Cmd.SET_TIP_FORCE]      = 1
        self.num_sends[Cmd.SET_TIP_LENGTH]     = 1
        self.num_sends[Cmd.SET_SHAFT_FORCE]    = 1
        self.num_sends[Cmd.SET_FORCE_INTERVAL] = 3

        # dividers
        self.divider[Cmd.GET_POS_ROT]          = self._MM_DIV
        self.divider[Cmd.GET_INCLINO_ANGLES]   = self._MM_DIV
        self.divider[Cmd.GET_INCLINO_ACC]      = self._MM_DIV
        self.divider[Cmd.GET_ROLL_ANGLE]       = self._MM_DIV
        self.divider[Cmd.GET_PITCH_ANGLE]      = self._MM_DIV
        self.divider[Cmd.SET_FORCE]            = self._MM_DIV
        self.divider[Cmd.SET_TIP_FORCE]        = self._MM_DIV
        self.divider[Cmd.SET_TIP_LENGTH]       = self._MM_DIV
        self.divider[Cmd.SET_SHAFT_FORCE]      = self._MM_DIV
        self.divider[Cmd.SET_FORCE_INTERVAL]   = self._MM_DIV
        self.divider[Cmd.GET_CURRENT_DELTA_T]  = 100.0  # dt * 100

    # ------------------------------------------------------------------
    # getters -----------------------------------------------------------
    def get_needle_position(self):
        # In millimeters
        return self.get_return(Cmd.GET_POS_ROT, 0)

    def get_needle_rotation(self):
        # In radians
        return self.get_return(Cmd.GET_POS_ROT, 1)
    
    def get_needle_pulse(self):
        return self.get_return(Cmd.GET_LAST_PWM, 0)

    def get_inclino_angle(self, axis: int):
        # In radians
        return self.get_return(Cmd.GET_INCLINO_ANGLES, axis)

    def get_inclino_acc(self):
        # In radians
        return (
            self.get_return(Cmd.GET_INCLINO_ACC, 0),
            self.get_return(Cmd.GET_INCLINO_ACC, 2),
            self.get_return(Cmd.GET_INCLINO_ACC, 1),
        )

    def get_roll_angle_deg(self):
        return self.get_return(Cmd.GET_ROLL_ANGLE, 0)

    def get_pitch_angle_deg(self):
        return self.get_return(Cmd.GET_PITCH_ANGLE, 0)

    # setters -----------------------------------------------------------
    def reset(self, mode: int = 0):
        self.set_send_float(Cmd.RESET, 0, mode)

    def set_force(self, force_n: float):
        self.set_send_float(Cmd.SET_FORCE, 0, force_n)

    def set_tip_force(self, force_n: float):
        self.set_send_float(Cmd.SET_TIP_FORCE, 0, force_n)

    def set_tip_length_mm(self, length_mm: float):
        self.set_send_float(Cmd.SET_TIP_LENGTH, 0, length_mm)

    def set_shaft_force(self, force_n: float):
        self.set_send_float(Cmd.SET_SHAFT_FORCE, 0, force_n)

    def set_force_interval(self, force_n: float, start_mm: float, end_mm: float):
        self.set_send_float(Cmd.SET_FORCE_INTERVAL, 0, force_n)
        self.set_send_float(Cmd.SET_FORCE_INTERVAL, 1, start_mm)
        self.set_send_float(Cmd.SET_FORCE_INTERVAL, 2, end_mm)
