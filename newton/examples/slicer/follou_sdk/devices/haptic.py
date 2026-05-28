from enum import IntEnum
import math
from typing import Optional, Tuple

from ..base import DeviceController


class Cmd(IntEnum):
    """Subset of the firmware command table for the Haptic‑Device controller.

    The enumeration values match the C# implementation exactly so that the
    command matrix can be transferred 1‑to‑1 to the device firmware.
    """

    RESET = 0
    GET_DEVICE_TYPE = 1
    GET_ANGLES_AND_LENGTH = 2
    GET_TOOL_ID = 3
    GET_CURRENT_DELTA_T = 4
    GET_STATUS = 5
    SET_MOTOR_FORCE_AND_TORQUES = 6
    SET_TIP_FORCE_AND_ROT_TORQUE = 7
    SET_YAW_PITCH_ZERO_ANG = 8
    SET_LED_BLINK_MODE = 9
    SET_COLLISION_OBJECT = 10
    SET_COLLISION_OBJECT_ACTIVE = 11
    SET_COLLISION_OBJECT_P0 = 12
    SET_COLLISION_OBJECT_V0 = 13
    SET_COLLISION_OBJECT_N = 14
    SET_COLLISION_OBJECT_Q = 15
    SET_COLLISION_OBJECT_R = 16
    SET_COLLISION_OBJECT_S = 17
    SET_COLLISION_OBJECT_T = 18
    SET_COLLISION_OBJECT_STIFFNESS = 19
    SET_COLLISION_OBJECT_DAMPING = 20
    SET_COLLISION_OBJECT_FRICTION = 21
    GET_LAST_COLLISION_FORCE = 22
    GET_LAST_PWM = 23
    GET_LAST_COLLISION_DATA = 24
    SET_TOOL_JAW_OPENING_ANGLE = 25
    GET_TOOL_JAW_TORQUE = 26
    SET_TOOL_DATA = 27
    GET_TOOL_INSERTED = 28
    GET_TOOL_TIP_VELOCITY = 29
    GET_TOOL_TIP_POSITION = 30
    GET_TOOL_DIRECTION = 31
    GET_RAW_ENCODER_VALUES = 32
    GET_ENCODER_SCALING_VALUES = 33
    GET_MOTOR_SCALING_VALUES = 34
    SET_MANUAL_PWM = 35
    GET_BOARD_TEMP = 36
    GET_BATTERY_VOLTAGE = 37
    GET_CALIBRATION_STATUS = 38
    GET_AMPLIFIERS_STATUS = 39
    GET_HALL_STATES = 40
    SET_POWER_ON_MANUAL = 41
    SET_FAN_ON_MANUAL = 42
    SET_FF_ENABLE = 43
    GET_SERIAL_NUM = 44
    GET_BUILD_DATE = 45
    SET_CHARGE_ENABLE = 46
    GET_TIP_LENGTH = 47
    GET_PART_TEMPERATURES = 48
    SET_MAX_USB_CHARGE_CURRENT = 49
    GET_USB_CHARGING_CURRENT = 50
    SET_DEADBAND_PWM_WIDTH = 51


class Dof(IntEnum):
    """Degrees‑of‑freedom indices used throughout the API."""

    ROT = 0
    PITCH = 1
    Z = 2
    YAW = 3


class HapticDevice(DeviceController):
    """Python wrapper for the Follou Haptic‑Device controller.

    The implementation is a line‑for‑line port of the original Unity/C# class,
    adapted to the Python style already used for :class:`Tracker4D` and
    :class:`USProbe`.
    """

    _FLOAT_DIV = 10_000.0  # firmware sends integers – divider converts back

    # ------------------------------------------------------------------
    # Construction / initialisation
    # ------------------------------------------------------------------
    def __init__(self, device_id: int):
        # Inform :class:`DeviceController` about the command‑table size *first*.
        super().__init__(device_id, cmd_enum_size=len(Cmd))

        # Which command returns the loop Δt (needed by base‑class timing code)
        self.get_dt_cmd = Cmd.GET_CURRENT_DELTA_T

        # Instrument meta‑data (purely application‑level – not on device)
        self._inserted_instrument_name: Optional[str] = None

        # Build the command matrix and push it to the firmware.
        self._init_matrix()
        self.transfer_matrix()

    # ------------------------------------------------------------------
    #   Command‑matrix setup (subscriptions, return/send sizes, scaling…)
    # ------------------------------------------------------------------
    def _init_matrix(self) -> None:
        """Populate *update_interval*, *num_rets*, *num_sends* and *divider*.

        The values are copied 1‑for‑1 from the Unity implementation.  Where the
        original first set a divider to :pyattr:`_FLOAT_DIV` and then overwrote
        it (e.g. **Δt** → *100.0*), the final value after all assignments is
        what appears here.
        """

        # 1) subscription intervals (use unique primes to spread the CAN load)
        self.update_interval[Cmd.GET_ANGLES_AND_LENGTH] = 1
        self.update_interval[Cmd.GET_TOOL_TIP_VELOCITY] = 1
        self.update_interval[Cmd.GET_TOOL_ID] = 1
        self.update_interval[Cmd.GET_CURRENT_DELTA_T] = 61
        self.update_interval[Cmd.GET_STATUS] = 103
        self.update_interval[Cmd.GET_LAST_PWM] = 31
        self.update_interval[Cmd.GET_BOARD_TEMP] = 1001
        self.update_interval[Cmd.GET_BATTERY_VOLTAGE] = 1003
        self.update_interval[Cmd.GET_CALIBRATION_STATUS] = 101
        self.update_interval[Cmd.GET_HALL_STATES] = 997
        self.update_interval[Cmd.GET_TIP_LENGTH] = 53

        # 2) return‑value counts (GETs)
        self.num_rets[Cmd.RESET] = 1
        self.num_rets[Cmd.GET_DEVICE_TYPE] = 1
        self.num_rets[Cmd.GET_ANGLES_AND_LENGTH] = 4
        self.num_rets[Cmd.GET_TOOL_ID] = 1
        self.num_rets[Cmd.GET_CURRENT_DELTA_T] = 1
        self.num_rets[Cmd.GET_STATUS] = 1
        self.num_rets[Cmd.GET_LAST_COLLISION_FORCE] = 3
        self.num_rets[Cmd.GET_LAST_PWM] = 4
        self.num_rets[Cmd.GET_LAST_COLLISION_DATA] = 1
        self.num_rets[Cmd.GET_TOOL_JAW_TORQUE] = 1
        self.num_rets[Cmd.GET_TOOL_INSERTED] = 1
        self.num_rets[Cmd.GET_TOOL_TIP_VELOCITY] = 3
        self.num_rets[Cmd.GET_TOOL_TIP_POSITION] = 3
        self.num_rets[Cmd.GET_TOOL_DIRECTION] = 3
        self.num_rets[Cmd.GET_RAW_ENCODER_VALUES] = 4
        self.num_rets[Cmd.GET_ENCODER_SCALING_VALUES] = 4
        self.num_rets[Cmd.GET_MOTOR_SCALING_VALUES] = 4
        self.num_rets[Cmd.GET_BOARD_TEMP] = 1
        self.num_rets[Cmd.GET_BATTERY_VOLTAGE] = 1
        self.num_rets[Cmd.GET_CALIBRATION_STATUS] = 3
        self.num_rets[Cmd.GET_AMPLIFIERS_STATUS] = 4
        self.num_rets[Cmd.GET_HALL_STATES] = 2
        self.num_rets[Cmd.GET_SERIAL_NUM] = 1
        self.num_rets[Cmd.GET_BUILD_DATE] = 1
        self.num_rets[Cmd.GET_TIP_LENGTH] = 1

        # 3) send‑value counts (SETs)
        self.num_sends[Cmd.RESET] = 1
        self.num_sends[Cmd.SET_MOTOR_FORCE_AND_TORQUES] = 4
        self.num_sends[Cmd.SET_TIP_FORCE_AND_ROT_TORQUE] = 4
        self.num_sends[Cmd.SET_YAW_PITCH_ZERO_ANG] = 2
        self.num_sends[Cmd.SET_LED_BLINK_MODE] = 1
        self.num_sends[Cmd.SET_COLLISION_OBJECT] = 19
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
        self.num_sends[Cmd.SET_MANUAL_PWM] = 4
        self.num_sends[Cmd.SET_POWER_ON_MANUAL] = 1
        self.num_sends[Cmd.SET_FAN_ON_MANUAL] = 1
        self.num_sends[Cmd.SET_FF_ENABLE] = 1
        self.num_sends[Cmd.SET_CHARGE_ENABLE] = 1
        self.num_sends[Cmd.SET_MAX_USB_CHARGE_CURRENT] = 1
        self.num_sends[Cmd.SET_DEADBAND_PWM_WIDTH] = 4

        # 4) scaling / dividers
        fd = self._FLOAT_DIV
        self.divider[Cmd.GET_ANGLES_AND_LENGTH] = fd
        self.divider[Cmd.GET_CURRENT_DELTA_T] = fd  # overwritten just below
        self.divider[Cmd.SET_MOTOR_FORCE_AND_TORQUES] = fd
        self.divider[Cmd.SET_TIP_FORCE_AND_ROT_TORQUE] = fd
        self.divider[Cmd.GET_LAST_COLLISION_FORCE] = fd
        self.divider[Cmd.SET_TOOL_JAW_OPENING_ANGLE] = fd
        self.divider[Cmd.GET_TOOL_JAW_TORQUE] = fd
        self.divider[Cmd.GET_TOOL_TIP_VELOCITY] = fd
        self.divider[Cmd.GET_TOOL_TIP_POSITION] = fd
        self.divider[Cmd.GET_TOOL_DIRECTION] = fd
        self.divider[Cmd.GET_ENCODER_SCALING_VALUES] = fd
        self.divider[Cmd.GET_MOTOR_SCALING_VALUES] = fd
        self.divider[Cmd.GET_BOARD_TEMP] = fd
        self.divider[Cmd.GET_BATTERY_VOLTAGE] = fd
        self.divider[Cmd.GET_AMPLIFIERS_STATUS] = fd
        self.divider[Cmd.GET_TIP_LENGTH] = fd
        self.divider[Cmd.GET_PART_TEMPERATURES] = fd
        self.divider[Cmd.SET_MAX_USB_CHARGE_CURRENT] = fd
        self.divider[Cmd.GET_USB_CHARGING_CURRENT] = fd

        # Δt sent as seconds ×100 → divide by 100 to get seconds
        self.divider[Cmd.GET_CURRENT_DELTA_T] = 100.0

    # ------------------------------------------------------------------
    #   High‑level getters – direct 1‑to‑1 ports of the C# API
    # ------------------------------------------------------------------
    def get_instrument_name(self) -> Optional[str]:
        return self._inserted_instrument_name

    def set_instrument_name(self, name: str) -> None:
        self._inserted_instrument_name = name

    # Value helpers ----------------------------------------------------
    def get_value(self, dof: Dof) -> float:
        """Angles in **radians** for *yaw/pitch/rot*, length in **mm** for *Z*."""
        return self.get_return(Cmd.GET_ANGLES_AND_LENGTH, dof.value)

    # Tool / instrument helpers ---------------------------------------
    def get_instrument_insertion(self) -> bool:
        return self.get_return(Cmd.GET_TOOL_INSERTED, 0) > 0

    def get_instrument_direction(self) -> Tuple[float, float, float]:
        return (
            self.get_return(Cmd.GET_TOOL_DIRECTION, 0),
            self.get_return(Cmd.GET_TOOL_DIRECTION, 1),
            self.get_return(Cmd.GET_TOOL_DIRECTION, 2),
        )

    def get_tip_length(self) -> float:
        return self.get_return(Cmd.GET_TIP_LENGTH, 0)

    def get_tool_id(self) -> int:
        return int(self.get_return(Cmd.GET_TOOL_ID, 0))

    # Calibration / amplifier status ----------------------------------
    def get_calibration_status(self) -> int:
        return (
            int(self.get_return(Cmd.GET_CALIBRATION_STATUS, 0)) * 100
            + int(self.get_return(Cmd.GET_CALIBRATION_STATUS, 1)) * 10
            + int(self.get_return(Cmd.GET_CALIBRATION_STATUS, 2))
        )

    def get_calibration_status_dof(self, dof: int) -> int:
        return int(self.get_return(Cmd.GET_CALIBRATION_STATUS, dof))

    def get_amp_status(self) -> int:
        return (
            int(self.get_return(Cmd.GET_AMPLIFIERS_STATUS, 0)) * 1000
            + int(self.get_return(Cmd.GET_AMPLIFIERS_STATUS, 1)) * 100
            + int(self.get_return(Cmd.GET_AMPLIFIERS_STATUS, 2)) * 10
            + int(self.get_return(Cmd.GET_AMPLIFIERS_STATUS, 3))
        )

    # Encoder raw values ----------------------------------------------
    def get_encoder_yaw(self) -> int:
        return int(self.get_return(Cmd.GET_RAW_ENCODER_VALUES, Dof.YAW.value))

    def get_encoder_pitch(self) -> int:
        return int(self.get_return(Cmd.GET_RAW_ENCODER_VALUES, Dof.PITCH.value))

    def get_encoder_rot(self) -> int:
        return int(self.get_return(Cmd.GET_RAW_ENCODER_VALUES, Dof.ROT.value))

    def get_encoder_z(self) -> int:
        return int(self.get_return(Cmd.GET_RAW_ENCODER_VALUES, Dof.Z.value))

    # Collision forces -------------------------------------------------
    def get_collision_force(self) -> Tuple[float, float, float]:
        return (
            self.get_return(Cmd.GET_LAST_COLLISION_FORCE, 0),
            self.get_return(Cmd.GET_LAST_COLLISION_FORCE, 1),
            self.get_return(Cmd.GET_LAST_COLLISION_FORCE, 2),
        )

    def get_collision_force_axis(self, axis: int) -> float:
        return self.get_collision_force()[axis] if 0 <= axis < 3 else 0.0

    # Board / battery monitoring --------------------------------------
    def get_battery_voltage(self) -> float:
        return self.get_return(Cmd.GET_BATTERY_VOLTAGE, 0)

    def get_board_temp(self) -> float:
        return self.get_return(Cmd.GET_BOARD_TEMP, 0)

    # Tip pose ---------------------------------------------------------
    def get_tip_position(self) -> Tuple[float, float, float]:
        """Return **cm** coordinates (original code divided by /100)."""
        return (
            self.get_return(Cmd.GET_TOOL_TIP_POSITION, 0) / 100.0,
            self.get_return(Cmd.GET_TOOL_TIP_POSITION, 1) / 100.0,
            self.get_return(Cmd.GET_TOOL_TIP_POSITION, 2) / 100.0,
        )

    def get_tip_velocity(self) -> Tuple[float, float, float]:
        """Velocity in **mm/s** using a right‑handed coordinate system."""
        return (
            self.get_return(Cmd.GET_TOOL_TIP_VELOCITY, 0),
            self.get_return(Cmd.GET_TOOL_TIP_VELOCITY, 1),
            self.get_return(Cmd.GET_TOOL_TIP_VELOCITY, 2),
        )

    # Jaw / PWM --------------------------------------------------------
    def get_jaw_torque(self) -> float:
        """Jaw torque in **N·mm**."""
        return self.get_return(Cmd.GET_TOOL_JAW_TORQUE, 0)

    def get_last_pwm(self, dof: "Dof") -> int:
        return int(self.get_return(Cmd.GET_LAST_PWM, dof.value))

    # ------------------------------------------------------------------
    #   Status / warning helpers
    # ------------------------------------------------------------------
    def _status_flags(self) -> int:
        return int(self.get_return(Cmd.GET_STATUS, 0))

    def get_power_on_status(self) -> bool:
        return bool(self._status_flags() & 0x40000)

    def get_fan_status(self) -> bool:
        return bool(self._status_flags() & 0x200000)

    def _warn_or_fault(self, mask: int) -> int:
        return int(self.get_power_on_status() and (self._status_flags() & mask) > 0)

    # Over‑temperature warnings
    def get_yaw_over_temp_warning(self) -> int:  # noqa: N802 – original C# name
        return self._warn_or_fault(0x80)

    def get_pitch_over_temp_warning(self) -> int:
        return self._warn_or_fault(0x20)

    def get_rot_over_temp_warning(self) -> int:
        return self._warn_or_fault(0x10)

    def get_z_over_temp_warning(self) -> int:
        return self._warn_or_fault(0x40)

    # Faults (share the same bit masks in legacy firmware)
    def get_yaw_fault(self) -> int:
        return self._warn_or_fault(0x80)

    def get_pitch_fault(self) -> int:
        return self._warn_or_fault(0x20)

    def get_rot_fault(self) -> int:
        return self._warn_or_fault(0x10)

    def get_z_fault(self) -> int:
        return self._warn_or_fault(0x40)

    # ------------------------------------------------------------------
    #   Actuation / configuration setters
    # ------------------------------------------------------------------
    def set_force_and_torques(
        self,
        z_force_n: float,
        yaw_torque_nm: float,
        pitch_torque_nm: float,
        rot_torque_nm: float,
    ) -> None:
        """Z‑force in **N**, torques in **N·m** (yaw/pitch/rot)."""
        self.set_send_float(Cmd.SET_MOTOR_FORCE_AND_TORQUES, Dof.ROT.value, rot_torque_nm)
        self.set_send_float(Cmd.SET_MOTOR_FORCE_AND_TORQUES, Dof.PITCH.value, pitch_torque_nm)
        self.set_send_float(Cmd.SET_MOTOR_FORCE_AND_TORQUES, Dof.Z.value, z_force_n)
        self.set_send_float(Cmd.SET_MOTOR_FORCE_AND_TORQUES, Dof.YAW.value, yaw_torque_nm)

    # Hall sensors -----------------------------------------------------
    def get_yaw_hall_state(self) -> float:
        return self.get_return(Cmd.GET_HALL_STATES, 0)

    def get_pitch_hall_state(self) -> float:
        return self.get_return(Cmd.GET_HALL_STATES, 1)

    # Jaw opening / zero‑angle calibration ----------------------------
    def set_opening_angle_deg(self, angle_deg: float) -> None:
        self.set_send_float(Cmd.SET_TOOL_JAW_OPENING_ANGLE, 0, math.radians(angle_deg))

    def set_yaw_pitch_zero_angles_deg(self, zero_yaw_deg: float, zero_pitch_deg: float) -> None:
        """Deprecated – handled by on‑board calibration, preserved for API compat."""
        self.set_send_float(Cmd.SET_YAW_PITCH_ZERO_ANG, 0, math.radians(zero_yaw_deg))
        self.set_send_float(Cmd.SET_YAW_PITCH_ZERO_ANG, 1, math.radians(zero_pitch_deg))

    # Dead‑band PWM widths --------------------------------------------
    def set_dead_band_width(self, z: int, yaw: int, pitch: int, rot: int) -> None:
        for dof, width in (
            (Dof.ROT, rot),
            (Dof.PITCH, pitch),
            (Dof.Z, z),
            (Dof.YAW, yaw),
        ):
            self.set_send_val(Cmd.SET_DEADBAND_PWM_WIDTH, dof.value, width)

    # Diagnostic manual PWM -------------------------------------------
    def set_pwm(self, rot_pwm: int, pitch_pwm: int, z_pwm: int, yaw_pwm: int) -> None:
        for dof, pwm in (
            (Dof.ROT, rot_pwm),
            (Dof.PITCH, pitch_pwm),
            (Dof.Z, z_pwm),
            (Dof.YAW, yaw_pwm),
        ):
            self.set_send_val(Cmd.SET_MANUAL_PWM, dof.value, pwm)

    # Power / fan / LED / feed‑forward enable -------------------------
    def set_power(self, on: bool) -> None:
        self.set_send_val(Cmd.SET_POWER_ON_MANUAL, 0, 1 if on else 0)

    def set_fan(self, on: bool) -> None:
        self.set_send_val(Cmd.SET_FAN_ON_MANUAL, 0, 1 if on else 0)

    def set_led_blink_mode(self, mode: int) -> None:
        self.set_send_val(Cmd.SET_LED_BLINK_MODE, 0, mode)

    def set_ff_enable(self, on: bool) -> None:
        self.set_send_val(Cmd.SET_FF_ENABLE, 0, 1 if on else 0)

    # ------------------------------------------------------------------
    #   Collision‑object commands (string based) – placeholders
    # ------------------------------------------------------------------
    # The original Unity implementation pushed *string baked* commands via
    # an *AppendCmdString()* helper that is not part of the Python base‑class.
    # These wrappers therefore raise :class:`NotImplementedError` so that the
    # caller can decide how to forward the strings (e.g. via ASCII CAN frames
    # or another side‑channel).
    def _string_cmd_placeholder(self, what: str) -> None:  # noqa: D401 – helper
        raise NotImplementedError(
            f"{what}() is not implemented – firmware expects a raw string "
            "packet.  Provide an implementation if you need this feature."
        )

    def set_collision_object(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object")

    def set_collision_object_pos(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_pos")

    def set_collision_object_dir(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_dir")

    def set_collision_object_active(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_active")

    def set_collision_object_friction(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_friction")

    def set_collision_object_damping(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_damping")

    def set_collision_object_stiffness(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_stiffness")

    def set_collision_object_radius(self, cmd_string: str) -> None:
        self._string_cmd_placeholder("set_collision_object_radius")

    # ------------------------------------------------------------------
    #   Convenience angle/length helpers (deg/mm – matches Unity helpers)
    # ------------------------------------------------------------------
    def get_rot_angle_deg(self) -> float:
        return math.degrees(self.get_return(Cmd.GET_ANGLES_AND_LENGTH, Dof.ROT.value))

    def get_pitch_angle_deg(self) -> float:
        return math.degrees(self.get_return(Cmd.GET_ANGLES_AND_LENGTH, Dof.PITCH.value))

    def get_yaw_angle_deg(self) -> float:
        return math.degrees(self.get_return(Cmd.GET_ANGLES_AND_LENGTH, Dof.YAW.value))

    def get_z_length_mm(self) -> float:
        return self.get_return(Cmd.GET_ANGLES_AND_LENGTH, Dof.Z.value)


__all__ = [
    "Cmd",
    "Dof",
    "HapticDevice",
]
