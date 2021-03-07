"""
Date: 2019.05.27
Programmer: MH
Description: Classes for button, button group and button manager
"""
import cv2
from flags import Flags, ButtonFlag
from styles import Style


class Button:
    """
    Button Class
    To initialize button location and size & action for whole buttons
    """
    x = 0
    y = 0
    h = 0
    w = 0
    color_on = (255, 0, 0)
    txt = ""
    icon = None
    color_off = (169, 169, 169)

    def __init__(self,  w, h, x=None, y=None, txt=None, style=None, icon=None, base=None):
        """
        To initialize
        :param w: int, width
        :param h: int, height
        :param x: int, start x axis
        :param y: int , start y axis
        :param txt: string, button text
        :param style: tuple, button on color
        :param icon: ndarray, icon image source
        :param base: tuple, button color if the button color is same at whole action (style must be None)
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if txt is not None:
            self.txt = txt
        if style is not None:
            self.color_on = style
        if base is not None:
            self.color_on = base
            self.color_off = base
        self.txt_size = 0.9
        self.txt_color = (50, 50, 50)
        self.btn_font = cv2.FONT_HERSHEY_SIMPLEX
        self.btn_on = False
        self.btn_color = self.color_off
        self.icon = icon
        self.img_w = 0
        self.img_h = 0
        self.back = False

    def set_x_y(self, x, y):
        self.x = x
        self.y = y

    def set_txt(self, txt):
        self.txt = txt

    def get_txt(self):
        return self.txt

    def get_x_y(self):
        return self.x, self.y

    def get_h_w(self):
        return self.h, self.w

    def set_on_off(self, s):
        self.btn_on = s
        if self.btn_on:
            self.btn_color = self.color_on
        else:
            self.btn_color = self.color_off

    def set_style(self, style):
        """
        To set button style
        :param style: tuple, button on color
        :return: None
        """
        if self.color_off != self.color_on:
            self.color_on = style
            if self.txt == "Theme":
                self.btn_color = self.color_on

    def draw(self, frame, x=None, y=None, img_h=None, img_w=None, txt=None):
        """
        To draw button
        :param frame: array, current input image
        :param x:
        :param y:
        :param img_h:
        :param img_w:
        :param txt:
        :return:
        """
        if txt is not None:
            self.txt = txt
        if (self.back) or (self.icon is None):
            cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color=self.btn_color,
                        thickness=cv2.FILLED)
        if self.icon is None:   # To add text in the button

            size = cv2.getTextSize(self.txt, self.btn_font, self.txt_size, 1)
            cv2.putText(frame, self.txt, (self.x + int(self.w / 2 - size[0][0] / 2),
                                             self.y + int(self.h / 2) + 10), self.btn_font,
                        self.txt_size, self.txt_color, 2, cv2.LINE_AA)
        else:   # To add icon image in the button
            if self.back:
                self.put_icon(frame, self.icon, x=int(self.x + (self.get_w()/2) - int(self.img_w*0.0625/2)), y=self.y + 5, white=False)
            else:
                self.put_icon(frame, self.icon, x=self.x, y=self.y, white=False)

    def on_click(self, x, y):
        """
        To initialize click event
        :param x: int, selected x location
        :param y: int, selected y location
        :return: int (action index) or None
        """
        #print(x,y)
        if (self.x <= x <= self.x+self.w) and (self.y <= y <= self.y+self.h):
            # To set button state and color
            if not self.btn_on:
                self.btn_color = self.color_on
                self.btn_on = not self.btn_on
            else:
                self.btn_on = not self.btn_on
                self.btn_color = self.color_off
            # To set functions button click event
            if self.txt == "Public":
                Flags.Public = not Flags.Public
            elif self.txt == "Personal":
                Flags.Personal = not Flags.Personal
            elif self.txt == "Assessment":
                Flags.Assess = not Flags.Assess
            elif self.txt == "Suggestion":
                Flags.Suggest = not Flags.Suggest
            elif self.txt == "Evolution":
                Flags.Evolution = not Flags.Evolution
            elif self.txt == "Diary":
                Flags.Diary = not Flags.Diary
                return 201
            elif self.txt == "Read":
                ButtonFlag.diary_read_flag = not ButtonFlag.diary_read_flag
            elif self.txt == "Record":
                pass
            elif self.txt == "Setting":
                return 200
            elif self.txt == "Diary_Back" or self.txt == "Setting_Back":
                Flags.Diary = False
                ButtonFlag.diary_read_flag = False
                ButtonFlag.record_flag = False
                return 100
            elif self.txt == "Display":
                return 300
            elif self.txt == "Theme":
                return 301
            elif self.txt == "Purple" or self.txt == "Green" or self.txt == "Blue":
                return self.txt
            elif self.txt == 'Top':
                return 0
            elif self.txt == 'Right':
                return 1
            elif self.txt == "Bottom":
                return 2
            elif self.txt == "Left":
                return 3
            elif self.txt == "+":
                pass
            elif self.txt == "-":
                pass
            elif self.txt == "O":
                Flags.BTN_Original = True
            elif self.txt == "C":
                Flags.Btn_capture = True
            elif self.txt == "V":
                Flags.BTN_Voice = True

    def set_is_touched(self):
        self.is_touched = not self.is_touched

    def put_icon(self, img, icon, x, y, white=True, size=50):
        # To put icon in button
        icon_img = cv2.resize(icon, dsize=(size, size), interpolation=cv2.INTER_AREA)
        width = icon_img.shape[0]
        height = icon_img.shape[1]
        roi = img[y:y+height, x:x+width]    # To need to set icon location to middle of button
        mask = cv2.bitwise_not(icon_img)
        if white:
            frame = cv2.add(roi, mask)
        else:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img[y:y+height, x:x+width] = frame

    def get_w(self):
        return self.w

    def get_h(self):
        return self.h


class BtnGroup:
    """
    Button Group Class
    To set group of button (location of the groups, margins between buttons)
    """
    list_btn = []

    def __init__(self, list_btn, w, h):
        self.list_btn = list_btn
        self.bottom_bound = 0.93
        self.top_bound = 0.01
        self.left_bound = 0.07
        self.side_up_bound = 0.2
        self.btn_interval_x = int(w * 0.0325)
        self.btn_interval_y = int(h * 0.1 + 10)
        self.start_x = 0
        self.start_y = 0
        self.btn_w = list_btn[0].get_w()
        self.btn_h = list_btn[0].get_h()

    def draw_all(self, frame, x=None, y=None, loc=0):
        """
        To draw group of buttons following current location and x, y location
        :param frame: ndarray, input frame
        :param x: int, start x axis
        :param y: int, start y axis
        :param loc: int, location of the group
        :return:
        """
        self.start_x = x
        self.start_y = y
        cur_x = x
        cur_y = y
        for b in self.list_btn:
            b.set_x_y(cur_x, cur_y)
            b.draw(frame)
            if loc == 0:  # Top
                cur_x += (self.btn_interval_x+b.get_w())
            elif loc == 1:    # Right
                cur_y += self.btn_interval_y
            elif loc == 2: # Bottom
                cur_x += (self.btn_interval_x+b.get_w())
            elif loc == 3: # Left
                cur_y += self.btn_interval_y
        return frame

    def on_click(self, x, y):
        """
        To set event of button and receive actions
        :param x: int, selected x axis
        :param y: int, selected y axis
        :return: int or None, clicked button action
        """
        result = None
        for b in self.list_btn:
            r = b.on_click(x, y)
            if r is not None:
                result = r
        return result

    def get_start_location(self):
        return self.start_x, self.start_y

    def get_under_location(self):
        x, y = self.get_start_location()
        return x, y + self.list_btn[0].get_h()

class BtnGroupManager:
        current_key = 100
        width = 0
        height = 0
        group_curr = []

        conf = [0, 1, 2, 3]
        count = 0

        def __init__(self, width=None, height=None):
            """
            To initialize Button Group Manager
            :param width: int, input frame width
            :param height: int, input frame height
            """
            self.width = width
            self.height = height
            self.loc = 1
            self.btn_height_rate = 0.06
            self.btn_width_rate = 0.27
            self.color_list = Style.blues
            self.current_style = "Blue"
            self.group_theme = None
            self.group_display = None
            self.list_main = {'key': 100, 'btn': []}
            self.list_diary = {'key': 201, 'btn': []}
            self.list_setting = {'key': 200, 'btn': []}
            self.list_display_btn = {'key': 300, 'btn': []}
            self.list_theme = {'key': 301, 'btn': []}
            self.list_zoom = {'key': 0, 'btn': []}
            self.make_btn()

        def make_btn(self):
            """
            To make buttons and make button groups
            :return:
            """
            # Left Top button group
            self.btn_zoom_in = Button(50, 50, txt="+", base=(169, 169, 169))
            self.btn_zoom_out = Button(50, 50, txt="-", base=(169, 169, 169))
            icon_org = cv2.imread('icon/zoom_original_50.png')
            self.btn_zoom_original = Button(50, 50, txt="O", icon=icon_org, base=(169, 169, 169))
            icon_cap = cv2.imread('icon/capture_50.png')
            self.btn_capture = Button(50, 50, txt="C", icon=icon_cap, base=(169, 169, 169))
            icon_voi = cv2.imread('icon/baseline_mic_black_50dp.png')
            self.btn_voice = Button(50, 50, txt="V", icon=icon_voi, base=(169, 169, 169))
            self.list_zoom['btn'] = [self.btn_zoom_in, self.btn_zoom_out,
                                     self.btn_zoom_original, self.btn_capture, self.btn_voice]

            icon_back = cv2.imread('icon/back_75.png')

            # Main Buttons
            self.btn_public = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Public",
                                     style=self.color_list[0])
            self.btn_personal = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Personal",
                                       style=self.color_list[0])
            self.btn_assess = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Assessment",
                                     style=self.color_list[0])
            self.btn_evol = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Evolution",
                                   style=self.color_list[0])
            self.btn_diary = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Diary", base=(169, 169, 169))
            self.btn_setting = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Setting", base=(169, 169, 169))
            self.list_main['btn'] = [self.btn_public, self.btn_personal, self.btn_assess, self.btn_evol,
                                     self.btn_diary, self.btn_setting]

            # Diary Buttons
            self.btn_diary_back = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Diary_Back",
                                         icon=icon_back, base=(169, 169, 169))
            self.btn_diary_back.img_w = self.width
            self.btn_diary_back.img_h = self.height
            self.btn_diary_back.back = True
            self.btn_record = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Record",
                                     style=self.color_list[1])
            self.btn_read = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Read", style=self.color_list[1])
            self.list_diary['btn'] = [self.btn_diary_back, self.btn_record, self.btn_read]

            # Setting Buttons
            self.btn_stt_back = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Setting_Back",
                                       icon=icon_back, base=(169, 169, 169))
            self.btn_stt_back.img_w = self.width
            self.btn_stt_back.img_h = self.height
            self.btn_stt_back.back = True
            self.btn_display = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Display",
                                      style=self.color_list[1])
            self.btn_assistant = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Assistant",
                                        style=self.color_list[1])
            self.btn_register = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Register",
                                       style=self.color_list[1])
            self.btn_theme = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Theme",
                                    style=self.color_list[1])
            self.list_setting['btn'] = [self.btn_stt_back, self.btn_display, self.btn_assistant, self.btn_register,
                                        self.btn_theme]

            # Display (Location) Buttons
            self.btn_bottom = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Bottom",
                                     style=self.color_list[2])
            self.btn_right = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Right",
                                    style=self.color_list[2])
            self.btn_top = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Top", style=self.color_list[2])
            self.btn_left = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Left", style=self.color_list[2])
            self.list_display_btn['btn'] = [self.btn_bottom, self.btn_right, self.btn_top, self.btn_left]

            # Theme Buttons
            self.btn_blue = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Blue", base=(255, 178, 96))
            self.btn_purple = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Purple", base=(255, 94, 185))
            self.btn_green = Button(int(self.width * self.btn_width_rate), int(self.height * self.btn_height_rate), txt="Green", base=(124, 255, 77))
            self.list_theme['btn'] = [self.btn_blue, self.btn_purple, self.btn_green]

        def draw_btn(self, frame):
            """
            To draw buttons following group location
            :param frame: ndarray, input frame
            :return:
            """
            overlay = frame.copy()
            if self.loc == 0:   # Top
                x = int(self.width*0.2)
                y = 10
                if self.current_key >= 300:
                    x2 = int(self.list_setting['btn'][1].x)
                    y2 = int(y + (self.list_display_btn['btn'][0].get_h() + 10))
                    x3 = int(self.list_setting['btn'][4].x)
                    y3 = int(y + (self.list_theme['btn'][0].get_h() + 10))
            elif self.loc == 1:  # Right
                x = int(self.width - self.list_main['btn'][0].get_w())
                y = int(self.height * 0.2)
                if self.current_key >= 300:
                    x2 = int(x - (self.list_display_btn['btn'][0].get_w() + 10))
                    y2 = int(self.list_setting['btn'][1].y)
                    x3 = int(x - (self.list_theme['btn'][0].get_w() + 10))
                    y3 = int(self.list_setting['btn'][4].y)
            elif self.loc == 2:  # Bottom
                x = int(self.width*0.2)
                y = int(self.height - self.list_main['btn'][0].get_h() - 10)
                if self.current_key >= 300:
                    x2 = int(self.list_setting['btn'][1].x)
                    y2 = int(y - len(self.list_display_btn['btn'])*(self.list_display_btn['btn'][0].get_h() + 10)-10)
                    x3 = int(self.list_setting['btn'][4].x)
                    y3 = int(y - len(self.list_theme['btn'])*(self.list_theme['btn'][0].get_h() + 10)-10)
            elif self.loc == 3:  # Left
                x = 0
                y = int(self.height * 0.2)
                if self.current_key >= 300:
                    x2 = int(x + (self.list_display_btn['btn'][0].get_w() + 20))
                    y2 = int(self.list_setting['btn'][1].y)
                    x3 = int(x + (self.list_theme['btn'][0].get_w() + 20))
                    y3 = int(self.list_setting['btn'][4].y)
            self.width, self.height = frame.shape[1], frame.shape[0]
            self.group_zoom = BtnGroup(self.list_zoom['btn'], self.width, self.height)
            self.group_zoom.btn_interval_x = 20
            if self.loc == 0:
                self.group_zoom.draw_all(frame, x=10, y=self.height-60, loc=0)
            else:
                self.group_zoom.draw_all(frame, x=10, y=10, loc=0)
            if self.current_key == 100:
                # Main Controller
                if Flags.Unknown:
                    self.list_main['btn'] = [self.btn_public, self.btn_personal, self.btn_assess,
                                             self.btn_evol, self.btn_diary, self.btn_setting]
                else:
                    self.list_main['btn'] = [self.btn_public, self.btn_assess, self.btn_setting]
                self.group_curr = BtnGroup(self.list_main['btn'], self.width, self.height)
                self.group_curr.draw_all(frame, x=x, y=y, loc=self.loc)
            elif self.current_key == 200:
                self.group_curr = BtnGroup(self.list_setting['btn'], self.width, self.height)
                self.group_curr.draw_all(frame, x=x, y=y, loc=self.loc)
            elif self.current_key == 201:
                self.group_curr = BtnGroup(self.list_diary['btn'], self.width, self.height)
                self.group_curr.draw_all(frame, x=x, y=y, loc=self.loc)
            elif self.current_key >= 300:
                self.group_curr = BtnGroup(self.list_setting['btn'], self.width, self.height)
                self.group_curr.draw_all(frame, x=x, y=y, loc=self.loc)
                if self.current_key == 300:
                    self.group_display = BtnGroup(self.list_display_btn['btn'], self.width, self.height)
                    self.group_display.draw_all(frame, x=x2, y=y2, loc=1)
                if self.current_key == 301:
                    self.group_theme = BtnGroup(self.list_theme['btn'], self.width, self.height)
                    self.group_theme.draw_all(frame, x=x3, y=y3, loc=1)

            elif self.current_style == "Purple" or self.current_style == "Blue" or self.current_style == "Green":
                # To change theme
                self.change_style(self.current_style)
            # To add alpha at button
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            return frame

        def change_location(self, loc):
            self.loc = loc

        def change_style(self, colors):
            """
            To change button style
            :param colors: string, color type
            :return:
            """
            if colors == "Purple":
                self.color_list = Style.purples
            elif colors == "Blue":
                self.color_list = Style.blues
            elif colors == "Green":
                self.color_list = Style.greens
            for b in self.list_main['btn']:
                b.set_style(self.color_list[0])
            for b in self.list_setting['btn']:
                b.set_style(self.color_list[1])
            for b in self.list_diary['btn']:
                b.set_style(self.color_list[1])
            for b in self.list_display_btn['btn']:
                b.set_style(self.color_list[2])

        def set_size(self, w, h):
            self.width = w
            self.height = h

        def on_click(self, x, y):
            """
            To send clicked locations and receive events
            :param x: int, selected x axis
            :param y: int, selected y axis
            :return:
            """

            result = self.group_curr.on_click(x, y)
            if result==100:
                for b in self.list_diary['btn']:
                    b.set_on_off(False)
                for b in self.list_setting['btn']:
                    b.set_on_off(False)

            if result is not None:
                self.current_key = result

            if self.group_display is not None:  # To check actions from display
                loc = self.group_display.on_click(x, y)
                if loc is not None:
                    self.change_location(loc)
            if self.group_theme is not None: # To check actions from theme
                self.current_style = self.group_theme.on_click(x, y)
                if self.current_style is not None:
                    self.change_style(self.current_style)
            self.group_zoom.on_click(x, y)

