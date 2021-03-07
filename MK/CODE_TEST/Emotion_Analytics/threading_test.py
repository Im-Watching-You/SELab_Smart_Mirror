import threading
import time


class MyThreading:
    def __init__(self):
        self.value = 0
        self.run()

    def get_value(self):
        return self.value

    def run(self):
        def th():
            while True:
                self.value += 1
                time.sleep(1)
        t = threading.Thread(target=th)
        t.start()

    def run_num(self, adding_num):
        def th(adding_num):
            self.value = adding_num + 1
            time.sleep(1)
        t = threading.Thread(target=th, args=(adding_num,))
        t.start()


class Main:
    def __init__(self):
        self.th = MyThreading()
        self.run()
        # self.run_adding()

    def run(self):
        def sub_run():
            while True:
                result = self.th.get_value()
                print(result)
                time.sleep(2)
        t = threading.Thread(target=sub_run)
        t.start()

    def run_adding(self):
        def sub_run():
            inp = [1, 2, 3, 4, 5]
            i = 0
            while True:
                self.th.run_num(inp[i%len(inp)])
                result = self.th.get_value()
                print(result)
                time.sleep(2)
                i+=1
        t = threading.Thread(target=sub_run)
        t.start()


if __name__ == '__main__':
    test = Main()
