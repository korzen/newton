from enum import IntEnum
from typing import Optional

from ..base import DeviceController


class Cmd(IntEnum):
    """Command table for the Instrument‑Box (IB) controller.

    The numeric values match the original Unity/C# enum **exactly** so that
    the command matrix can be copied 1‑to‑1 to the firmware.
    """

    RESET = 0
    GET_DEVICE_TYPE = 1
    GET_OPENING_VALUES = 2
    GET_HANDLE_IDS = 3
    GET_PEDAL_STATES = 4
    SET_ALL_FORCES = 5
    SET_CHAN_FORCE = 6
    GET_STATUS = 7
    GET_CALIBRATION_STATUS = 8
    GET_MOTOR_BOARD_STATUS = 9
    GET_BATTERY_VOLTAGE = 10
    GET_BOARD_TEMP = 11
    SET_MANUAL_PWM = 12
    SET_POWER_ON_MANUAL = 13
    SET_FAN_ON_MANUAL = 14
    GET_LAST_PWM = 15
    SET_FF_ENABLE = 16
    GET_CURRENT_DELTA_T = 17
    SET_LOOP_GAIN = 18
    GET_OPTO_FORCE = 19
    SET_ZERO_FORCE = 20
    GET_POS_VOLTAGE = 21
    GET_HANDLE_IDS_REAL = 22
    GET_BUILD_DATE = 23
    GET_SERIAL_NUM = 24
    SET_CHARGE_ENABLE = 25
    SET_HANDLE_LED = 26
    GET_OPTO_VOLTAGES = 27
    SET_TO_CALIBRATE = 28
    GET_PART_TEMPERATURES = 29
    SET_MAX_USB_CHARGE_CURRENT = 30
    GET_USB_CHARGING_CURRENT = 31
    GET_CONNECTION_STATES = 32
    GET_HANDLES_ACTIVITY = 33
    SET_FORCE_OFFSET = 34


class IBController(DeviceController):
    """Python wrapper for the Follou Instrument‑Box controller.

    This is a faithful port of the original Unity/C# *IBController*.
    """

    _FLOAT_DIV = 10_000.0

    # ------------------------------------------------------------------
    def __init__(self, device_id: int, num_instrument_channels: int = 6):
        # Inspector value in Unity → constructor argument in Python.
        self.num_instrument_channels: int = num_instrument_channels

        super().__init__(device_id, cmd_enum_size=len(Cmd))
        self.get_dt_cmd = Cmd.GET_CURRENT_DELTA_T

        # Instrument meta‑data
        self._first_tool_id: int = 3

        self._init_matrix()
        self.transfer_matrix()

        # Bring device into default operating mode (mimic Unity Start())
        self.set_ff_enable(True)
        # Set loop gains for channel 2 and 0 as done in Unity Start()
        self.set_loop_gain(2, 4.0, 0.0)
        self.set_loop_gain(0, 4.0, 0.0)

    # ------------------------------------------------------------------
    #   Command‑matrix setup
    # ------------------------------------------------------------------
    def _init_matrix(self) -> None:
        ui = self.update_interval
        nr = self.num_rets
        ns = self.num_sends
        dv = self.divider
        n_ch = self.num_instrument_channels
        fd = self._FLOAT_DIV

        # Subscription intervals (prime numbers spread CAN load)
        ui[Cmd.GET_OPENING_VALUES] = 1
        ui[Cmd.GET_CURRENT_DELTA_T] = 63
        ui[Cmd.GET_STATUS] = 89
        ui[Cmd.GET_LAST_PWM] = 41
        ui[Cmd.GET_BOARD_TEMP] = 929
        ui[Cmd.GET_BATTERY_VOLTAGE] = 857
        ui[Cmd.GET_CALIBRATION_STATUS] = 881
        ui[Cmd.GET_OPTO_FORCE] = 3
        ui[Cmd.GET_OPTO_VOLTAGES] = 3
        ui[Cmd.GET_PEDAL_STATES] = 11

        # Return sizes (GETs)
        nr[Cmd.RESET] = 1
        nr[Cmd.GET_DEVICE_TYPE] = 1
        nr[Cmd.GET_OPENING_VALUES] = n_ch
        nr[Cmd.GET_HANDLE_IDS] = n_ch
        nr[Cmd.GET_PEDAL_STATES] = 2
        nr[Cmd.GET_STATUS] = 1
        nr[Cmd.GET_CALIBRATION_STATUS] = n_ch
        nr[Cmd.GET_MOTOR_BOARD_STATUS] = n_ch
        nr[Cmd.GET_BATTERY_VOLTAGE] = 1
        nr[Cmd.GET_BOARD_TEMP] = 1
        nr[Cmd.GET_LAST_PWM] = n_ch
        nr[Cmd.GET_CURRENT_DELTA_T] = 1
        nr[Cmd.GET_OPTO_FORCE] = n_ch
        nr[Cmd.GET_POS_VOLTAGE] = n_ch
        nr[Cmd.GET_HANDLE_IDS_REAL] = n_ch
        nr[Cmd.GET_BUILD_DATE] = 1
        nr[Cmd.GET_SERIAL_NUM] = 1
        nr[Cmd.GET_OPTO_VOLTAGES] = n_ch

        # Send sizes (SETs)
        ns[Cmd.SET_ALL_FORCES] = n_ch
        ns[Cmd.SET_CHAN_FORCE] = 2  # chan + force
        ns[Cmd.SET_MANUAL_PWM] = n_ch
        ns[Cmd.SET_POWER_ON_MANUAL] = 1
        ns[Cmd.SET_FAN_ON_MANUAL] = 1
        ns[Cmd.SET_FF_ENABLE] = 1
        ns[Cmd.SET_LOOP_GAIN] = 3  # chan + kp + kd
        ns[Cmd.SET_ZERO_FORCE] = 1  # chan
        ns[Cmd.SET_CHARGE_ENABLE] = 1
        ns[Cmd.SET_HANDLE_LED] = 2  # chan + power (1‑100)
        ns[Cmd.SET_TO_CALIBRATE] = 1  # chan
        ns[Cmd.SET_MAX_USB_CHARGE_CURRENT] = 1
        ns[Cmd.SET_FORCE_OFFSET] = 2  # chan + offsetPWM

        # Scaling / dividers
        dv[Cmd.GET_OPENING_VALUES] = fd
        dv[Cmd.SET_ALL_FORCES] = fd
        dv[Cmd.SET_CHAN_FORCE] = fd
        dv[Cmd.GET_BATTERY_VOLTAGE] = fd
        dv[Cmd.GET_BOARD_TEMP] = fd
        dv[Cmd.GET_CURRENT_DELTA_T] = fd  # overwritten below
        dv[Cmd.SET_LOOP_GAIN] = fd
        dv[Cmd.GET_OPTO_FORCE] = fd
        dv[Cmd.GET_POS_VOLTAGE] = fd
        dv[Cmd.GET_HANDLE_IDS_REAL] = fd
        dv[Cmd.GET_OPTO_VOLTAGES] = fd

        # Δt sent in sec×100 → divide by 100
        dv[Cmd.GET_CURRENT_DELTA_T] = 100.0

    # ------------------------------------------------------------------
    #   High‑level getters
    # ------------------------------------------------------------------
    def _chan_from_id(self, tool_id: int) -> int:
        return tool_id - self._first_tool_id

    def get_opening_prc_from_channel(self, chan: int) -> float:
        if 0 <= chan < self.num_instrument_channels:
            return self.get_return(Cmd.GET_OPENING_VALUES, chan) * 100.0
        return 0.0

    def get_opening_prc_from_id(self, tool_id: int) -> float:
        return self.get_opening_prc_from_channel(self._chan_from_id(tool_id))

    def get_force_from_channel(self, chan: int) -> float:
        if 0 <= chan < self.num_instrument_channels:
            return self.get_return(Cmd.GET_OPTO_FORCE, chan)
        return 0.0

    def get_force_from_id(self, tool_id: int) -> float:
        return self.get_force_from_channel(self._chan_from_id(tool_id))

    def get_last_pwm_from_channel(self, chan: int) -> int:
        if 0 <= chan < self.num_instrument_channels:
            return int(self.get_return(Cmd.GET_LAST_PWM, chan))
        return 0

    def get_last_pwm_from_id(self, tool_id: int) -> int:
        return self.get_last_pwm_from_channel(self._chan_from_id(tool_id))

    def get_battery_voltage(self) -> float:
        return self.get_return(Cmd.GET_BATTERY_VOLTAGE, 0)

    def get_board_temp(self) -> float:
        return self.get_return(Cmd.GET_BOARD_TEMP, 0)

    def get_amp_status(self) -> int:
        return (
            int(self.get_return(Cmd.GET_MOTOR_BOARD_STATUS, 0)) * 1000
            + int(self.get_return(Cmd.GET_MOTOR_BOARD_STATUS, 1)) * 100
            + int(self.get_return(Cmd.GET_MOTOR_BOARD_STATUS, 2)) * 10
            + int(self.get_return(Cmd.GET_MOTOR_BOARD_STATUS, 3))
        )

    def get_pedal_state(self, pedal: int) -> int:
        return int(self.get_return(Cmd.GET_PEDAL_STATES, pedal))

    # ------------------------------------------------------------------
    #   Setters / configuration
    # ------------------------------------------------------------------
    def set_ff_enable(self, on: bool) -> None:
        self.set_send_val(Cmd.SET_FF_ENABLE, 0, 1 if on else 0)

    def set_handle_force_to_channel(self, chan: int, force_n: float) -> None:
        if 0 <= chan < self.num_instrument_channels:
            self.set_send_float(Cmd.SET_ALL_FORCES, chan, force_n)

    def set_handle_force_to_id(self, tool_id: int, force_n: float) -> None:
        self.set_handle_force_to_channel(self._chan_from_id(tool_id), force_n)

    def set_loop_gain(self, chan: int, kp: float, kd: float) -> None:
        self.set_send_val(Cmd.SET_LOOP_GAIN, 0, chan)
        self.set_send_float(Cmd.SET_LOOP_GAIN, 1, kp)
        self.set_send_float(Cmd.SET_LOOP_GAIN, 2, kd)


__all__ = ["Cmd", "IBController"]
