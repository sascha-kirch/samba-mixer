from enum import Enum


class DataFormat(Enum):
    CK = "capacity_k"
    SOH = "soh"


class TrainingObjective(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ClsTokenType(Enum):
    NONE = "none"
    HEAD = "head"
    MIDDLE = "middle"
    TAIL = "tail"


class TimeSignalResamplingType(Enum):
    LINEAR = "linear"
    RANDOM = "random"
    ANCHORS = "anchors"
    ANCHORS_NEW = "anchors_new"


class OversampleType(Enum):
    NONE = "none"
    X2 = "x2"
    X3 = "x3"
    MAX = "max"
