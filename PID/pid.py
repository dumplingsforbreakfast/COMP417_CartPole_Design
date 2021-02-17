import numpy as np
import cv2
import sys
import time



class PIDController:
    def __init__(self, target_pos):
        self.target_pos = target_pos
        # Values for 2.7g ball
        #self.target_pos = 0.5
        self.Kp = 3130.25
        self.Ki = 3298.32
        self.Kd = 252.10

        self.bias = 0.0
        self.error_0_to_t = 0.0
        self.prev_error = 0.0
        self.error_at_t = 0.0
        return

    def reset(self):
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0
        self.bias = 0.0
        self.error_0_to_t = 0.0
        self.prev_error = 0.0
        self.error_at_t = 0.0
        return

    def get_fan_rpm(self, vertical_ball_position):
        delta_t = 1.0 / 60.0

        # calculate the error function at every frame
        prev_error = self.error_at_t
        self.error_at_t = self.target_pos - vertical_ball_position
        self.error_0_to_t += self.error_at_t * delta_t

        # calculate pid:  u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        _P = self.Kp * self.error_at_t
        _I = self.Ki * self.error_0_to_t
        _D = self.Kd * (self.error_at_t - prev_error) / delta_t
        output = _P + _I + _D

        return output


