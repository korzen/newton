from ctypes import c_int, c_float, c_char_p
from .loader import load_lib

lib = load_lib()          # loads & keeps a global handle

# discovery / global
InitDLL                = lib.InitDLL;               InitDLL.restype = c_int
GetNextDevice          = lib.GetNextDevice;         GetNextDevice.restype = c_int
GetDeviceType          = lib.GetDeviceType;         GetDeviceType.restype = c_char_p
GetCOMPortName         = lib.GetCOMPortName;        GetCOMPortName.restype = c_char_p
AskString              = lib.AskString;             AskString.argtypes = (c_int,); AskString.restype = c_char_p

# per–device
SetupCmd               = lib.SetupCmd;              SetupCmd.argtypes = (c_int, c_int, c_int, c_int, c_float, c_int)
GetReturnVal           = lib.GetReturnVal;          GetReturnVal.argtypes = (c_int, c_int, c_int); GetReturnVal.restype = c_float
SetSendVal             = lib.SetSendVal;            SetSendVal.argtypes = (c_int, c_int, c_int, c_int)
UpdateDevice           = lib.UpdateDevice;          UpdateDevice.argtypes = (c_int,)
CloseDevice            = lib.CloseDevice;           CloseDevice.argtypes = (c_int,)