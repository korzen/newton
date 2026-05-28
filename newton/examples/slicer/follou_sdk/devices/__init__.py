from .tracker4d import Tracker4D
from .usprobe import USProbe
from .haptic import HapticDevice
from .ibcontroller import IBController
from .minimou import MiniMou
from .scope import Scope


DEVICE_MAP = {
    "Tracker4D": Tracker4D,
    "USProbe":   USProbe, 
    "HapticDevice": HapticDevice,
    "InstrumentBox": IBController,
    "MiniMou": MiniMou,
    "Scope": Scope
}