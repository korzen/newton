from enum import IntEnum
import math
from ..base import DeviceController


class Cmd(IntEnum):
    """Subset of the firmware command table for the Tracker‑4D device.

    Add the rest of the enum entries when you need them; the ones below are
    enough to stream pose + insertion length at ~1 kHz.
    """

    RESET = 0
    GET_DEVICE_TYPE = 1
    GET_ANGLES_AND_LENGTH = 2   # yaw, pitch, roll, insertion L
    GET_POSITION = 3            # 3‑tuple (mm) – optional if you only need angles
    GET_ORIENTATION = 4         # (x, y, z, angle)
    GET_VELOCITY = 5
    GET_CURRENT_DELTA_T = 6     # loop Δt in seconds (×100)
    GET_STATUS = 7
    SET_ROLL_ANGLE = 8          # commanded roll – ε.g. from probe
    SET_LED_INTENSITY = 9
    GET_SERIAL_NUM = 10
    GET_BUILD_DATE = 11
    SET_IMPLANT_MODE = 12


class Tracker4D(DeviceController):
    """Python wrapper for the Follou Tracker‑4D hand‑held tracker."""

    _MM_DIV = 100000.0   # firmware sends ints – divider converts back to float

    def __init__(self, device_id: int):
        # tell the base class how many commands exist **before** ctor runs
        super().__init__(device_id, cmd_enum_size=len(Cmd))
        self.get_dt_cmd = Cmd.GET_CURRENT_DELTA_T
        self._init_matrix()
        self.transfer_matrix()

    # ------------------------------------------------------------------
    #   Command‑matrix setup (subscriptions, scaling, etc.)
    # ------------------------------------------------------------------
    def _init_matrix(self):
        # ---------------------------------------------------------------
        # 1) subscription rates (prime numbers spread the bus load)
        # ---------------------------------------------------------------
        self.update_interval[Cmd.GET_ANGLES_AND_LENGTH] = 1
        self.update_interval[Cmd.GET_POSITION]          = 1
        self.update_interval[Cmd.GET_ORIENTATION]       = 1
        self.update_interval[Cmd.GET_VELOCITY]          = 1
        self.update_interval[Cmd.GET_CURRENT_DELTA_T]   = 11
        self.update_interval[Cmd.GET_STATUS]            = 13

        # ---------------------------------------------------------------
        # 2) return-value counts (GETs)
        # ---------------------------------------------------------------
        self.num_rets[Cmd.RESET]                 = 0
        self.num_rets[Cmd.GET_DEVICE_TYPE]       = 1
        self.num_rets[Cmd.GET_ANGLES_AND_LENGTH] = 4
        self.num_rets[Cmd.GET_POSITION]          = 3
        self.num_rets[Cmd.GET_ORIENTATION]       = 4
        self.num_rets[Cmd.GET_VELOCITY]          = 3
        self.num_rets[Cmd.GET_CURRENT_DELTA_T]   = 1
        self.num_rets[Cmd.GET_STATUS]            = 1
        self.num_rets[Cmd.GET_SERIAL_NUM]        = 1
        self.num_rets[Cmd.GET_BUILD_DATE]        = 1

        # ---------------------------------------------------------------
        # 3) send-value counts (SETs)
        # ---------------------------------------------------------------
        self.num_sends[Cmd.RESET]            = 1
        self.num_sends[Cmd.SET_ROLL_ANGLE]   = 1
        self.num_sends[Cmd.SET_LED_INTENSITY]= 3
        self.num_sends[Cmd.SET_IMPLANT_MODE] = 1

        # ---------------------------------------------------------------
        # 4) scaling / dividers
        # ---------------------------------------------------------------
        mm_div = self._MM_DIV                        # 100 000.0

        self.divider[Cmd.GET_ANGLES_AND_LENGTH] = mm_div
        self.divider[Cmd.GET_VELOCITY]          = mm_div 
        self.divider[Cmd.GET_POSITION]          = mm_div 
        self.divider[Cmd.GET_ORIENTATION]       = mm_div
        self.divider[Cmd.GET_CURRENT_DELTA_T]   = 100.0      # Δt sent as sec×100
        self.divider[Cmd.SET_ROLL_ANGLE]        = mm_div


    # ------------------------------------------------------------------
    #   Convenience high‑level getters / setters
    # ------------------------------------------------------------------
    def get_angle_rad(self, ix: int) -> float:
        return self.get_return(Cmd.GET_ANGLES_AND_LENGTH, ix)

    def get_angle_deg(self, ix: int) -> float:
        return math.degrees(self.get_angle_rad(ix))

    # Public helpers ----------------------------------------------------
    def get_yaw_deg(self):
        return self.get_angle_deg(0)

    def get_pitch_deg(self):
        return self.get_angle_deg(1)

    def get_roll_deg(self):
        return self.get_angle_deg(2)

    def get_insertion_length_mm(self):
        return self.get_return(Cmd.GET_ANGLES_AND_LENGTH, 3)

    def set_roll_angle(self, angle: float):
        """Tell the tracker what *absolute* roll angle you want (deg)."""
        self.set_send_float(Cmd.SET_ROLL_ANGLE, 0, angle)

    # status helpers ----------------------------------------------------
    def is_yaw_calibrated(self):
        return bool(int(self.get_return(Cmd.GET_STATUS, 0)) & 0x01)

    def is_pitch_calibrated(self):
        return bool(int(self.get_return(Cmd.GET_STATUS, 0)) & 0x02)

    def left_pedal_pressed(self):
        return bool(int(self.get_return(Cmd.GET_STATUS, 0)) & 0x04)

    def right_pedal_pressed(self):
        return bool(int(self.get_return(Cmd.GET_STATUS, 0)) & 0x08)
    
    def set_implant_mode(self, on: bool):
        int_on = 1 if on else 0
        self.set_send_val(Cmd.SET_IMPLANT_MODE, int_on)

    # ------------------------------------------------------------------
    #   Orientation / position – optional (needs more dividers & sizes)
    # ------------------------------------------------------------------
    # Implement when you need them:
    #   def orientation_quat(self): ...
    #   def position_vec(self): ...
