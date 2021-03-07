import cv2


class TextView:
    def __init__(self):
        self.txt_size = 0.9
        self.txt_color = (50, 50, 50)
        self.btn_font = cv2.FONT_HERSHEY_SIMPLEX

    def put_text(self, frame, txt, loc):
        if not loc == 3:
            x = 30
            y = 100
        if len(txt) > 1:
            for t in txt:
                cv2.putText(frame, t, (x, y), self.btn_font,
                            self.txt_size, self.txt_color, 2, cv2.LINE_AA)
                size = cv2.getTextSize(t, self.btn_font, self.txt_size, 2)
                y += (size[0][1]+10)
        elif len(txt) == 1:
            cv2.putText(frame, txt[0], (x, y), self.btn_font,
                        self.txt_size, self.txt_color, 2, cv2.LINE_AA)
