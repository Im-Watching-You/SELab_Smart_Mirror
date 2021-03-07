"""
Date 2019. 06. 18.
Programmer: MH
Description: Code for greeting to detected user.
"""


class Greeting:

    def __init__(self):
        self.n_detected_people = 0
        self.detected_id = 0
        self.is_greet = False
        self.is_session = False

    def change_greet_state(self, b):
        self.is_greet = b

    def change_session_state(self, b):
        self.is_session = b

    def set_detected_num(self, n):
        self.n_detected_people = n

    def get_detected_num(self):
        return self.n_detected_people

    def get_is_greet(self):
        return self.is_greet

    def get_is_session(self):
        return self.is_session

    def greet(self, cur_id):
        if self.detected_id == cur_id:
            pass
        else:
            self.is_greet = False
            self.is_session = False

    def check_greet_case(self, nf):
        if self.n_detected_people == nf:    # number of prev. people == number of current people
            if nf == 1:
                print("# recognize")
                pass
            elif nf > 1:
                # choose again
                print("# choose")
        elif self.n_detected_people > nf:
            if nf == 1:
                print("# recognize")
                # recognize
                pass
            elif nf > 1:
                print("# choose")
                # choose again
        elif self.n_detected_people < nf:
            if nf == 1:  # n_d_p is 0
                print("# recognize")
                # recognize
                pass
            elif nf > 1:
                print("# choose")
                # choose
                pass



