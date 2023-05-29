from dataclasses import dataclass


@dataclass
class ImageConfig:
    start_x: int
    start_y: int
    xrowmod: int
    ycolmod: int



configs = {
    "0" : ImageConfig(93, 130, -0.35, 0),     # OK
    "1" : ImageConfig(126, 125, 0.35, 0.5),   # OK
    "2" : ImageConfig(115, 135, 0.35, 0.15),  # OK
    "3" : ImageConfig(95, 135, -0.35, -0.15), # OK
    "4" : ImageConfig(114, 135, 0.35, 0.5),   # OK
    "5" : ImageConfig(117, 123, 0.35, 0.5),   # OK
    "6" : ImageConfig(102, 123, -0.35, 0.5),  # OK
    "7" : ImageConfig(123, 116, 0.35, 0.5),   # OK
    "8" : ImageConfig(122, 120, 0.35, 0.5),   # OK
    "9" : ImageConfig(140, 120, 0.5, 0.95),   # OK
}


