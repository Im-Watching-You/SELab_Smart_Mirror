from buttons import Style
import cv2
import os
from PIL import ImageFont, ImageDraw, Image
from threading import Thread
from db_connector import DiarySession
from pydub import AudioSegment
from pydub.playback import play


class Box:
    def __init__(self, text, width, height):
        self.img_width = width
        self.img_height = height

        self.box_text = text
        self.box_location_x = 0
        self.box_location_y = 0
        self.box_color = Style.white
        self.box_width = int(self.img_width * 0.14)
        self.box_height = int(self.img_height * 0.0625)

        self.text_size = 1
        self.text_color = Style.black
        self.box_font = Style.font[0]
        self.upper_line = False
        self.under_line = False

    def set_location_x(self, x):
        self.box_location_x = x

    def set_location_y(self, y):
        self.box_location_y = y

    def name_change(self, name):
        self.box_text = name

    def make_line(self, img, upper=False):
        start_x, start_y = self.get_start_location()
        if upper:
            cv2.line(img, (start_x, start_y), (start_x + self.box_width, start_y), Style.black, 1)
        else:
            cv2.line(img, (start_x, start_y + self.box_height), self.get_end_location(), Style.black, 1)

    def set_btn_size(self, width, height):
        self.box_width = width
        self.box_height = height

    def get_start_location(self):
        return (self.box_location_x, self.box_location_y)

    def get_end_location(self):
        return (self.box_location_x + self.box_width, self.box_location_y + self.box_height)

    def draw_box(self, img):
        cv2.rectangle(img, self.get_start_location(), self.get_end_location(), self.box_color, cv2.FILLED)
        if self.upper_line:
            self.make_line(img, upper=True)
        if self.under_line:
            self.make_line(img)

    def put_text(self, img, left=False, thick=2):
        if left:
            cv2.putText(img, self.box_text, (self.box_location_x + 20,
                                             self.box_location_y + int(self.box_height/2) + 10), self.box_font,
                        self.text_size, self.text_color, thick, cv2.LINE_AA)
        else:
            size = cv2.getTextSize(self.box_text, self.box_font, self.text_size, 1)
            cv2.putText(img, self.box_text, (self.box_location_x + int(self.box_width / 2 - size[0][0] / 2),
                                             self.box_location_y + int(self.box_height/2) + 10), self.box_font, self.text_size, self.text_color, thick, cv2.LINE_AA)

    def put_icon(self, img, icon, x, y, white=True, size=50):
        icon_img = cv2.resize(icon, dsize=(size, size), interpolation=cv2.INTER_AREA)
        width = icon_img.shape[0]
        height = icon_img.shape[1]
        roi = img[y:y+height, x:x+width]
        mask = cv2.bitwise_not(icon_img)
        if white:
            frame = cv2.add(roi, mask)
        else:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img[y:y+height, x:x+width] = frame

    def in_menu(self, x, y):
        if (x >= self.box_location_x) and (x <= self.box_location_x + self.box_width) and (y >= self.box_location_y) and (y <= self.box_location_y + self.box_height):
            return True


class ListView:
    def __init__(self):
        self.img_width = 0
        self.img_height = 0
        self.start_x = 0
        self.start_y = 0
        self.num_of_list = 5
        self.content_height = 0
        self.content_width = 0
        self.content_box_list = []
        self.content_box_color = Style.white
        self.diary_data = None
        self.content_index = None

        self.detail_flag = False
        self.detail_data = None
        self.detail_content_img = None
        self.detail_list = []

        self.back_icon = cv2.imread('icon/back_75.png')
        self.play_icon = cv2.imread('icon/play_50.png')
        self.trash_icon = cv2.imread('icon/trash_icon_75.png')
        self.icon_list = []

    def set_XY(self, x, y):
        self.start_x = x
        self.start_y = y + 10

    def set_size(self, width, height):
        self.img_width = width
        self.img_height = height
        self.content_height = int(self.img_height * 0.2 / self.num_of_list)
        self.content_width = int(self.img_width * 0.14 * 2.1)

    def set_num_of_list(self, num):
        self.num_of_list = num

    def set_data(self, data):
        self.diary_data = data

    def listview_init(self):
        self.content_box_list = []

    def make_listview(self, img):
        # self.set_size(img_width, img_height)
        self.listview_init()
        if self.detail_flag:
            self.change_view(self.detail_data)
            self.locate_content_box(self.detail_list, detail=True)
            for i in range(0, len(self.detail_list)):
                self.detail_list[i].draw_box(img)
                if i == 0:
                    self.detail_list[i].put_text(img)
                elif i == len(self.detail_list)-1:
                    self.draw_text_image(img, self.detail_list[i].box_location_x, self.detail_list[i].box_location_y)
                else:
                    self.detail_list[i].put_text(img, left=True)
            self.set_icon(img)
        else:
            self.make_content_box()
            self.locate_content_box(self.content_box_list)
            for i in range(0, len(self.content_box_list)):
                self.content_box_list[i].draw_box(img)
                self.content_box_list[i].put_text(img)

    def make_content_box(self):
        title = Box('Smart Diary', self.img_width, self.img_height)
        title.box_height = self.content_height
        title.box_width = self.content_width
        title.box_color = self.content_box_color
        self.content_box_list.append(title)
        for i in range(0, len(self.diary_data)):
            content = Box('No data', self.img_width, self.img_height)
            content.box_height = self.content_height
            content.box_width = self.content_width
            content.box_color = self.content_box_color
            content.box_text = self.diary_data[i]['session_time']
            self.content_box_list.append(content)

    def locate_content_box(self, list, detail=False):
        list[0].box_location_x = self.start_x
        list[0].box_location_y = self.start_y
        for i in range(0, len(list)-1):
            list[i+1].box_location_x = list[i].box_location_x
            if detail and i >= 4:
                list[i + 1].box_location_y = list[i].box_location_y + list[i].box_height
            else:
                list[i+1].box_location_y = list[i].box_location_y + list[i].box_height + 2

    def in_list(self, x, y):
        for i in range(0, len(self.content_box_list)):
            if self.content_box_list[i].in_menu(x, y):
                return True
            else:
                continue
        return False

    def listview_event(self, x, y):
        for i in range(1, len(self.content_box_list)):
            if self.content_box_list[i].in_menu(x, y):
                self.content_event(i-1)
            else:
                continue

    def content_event(self, index):
        if self.diary_data is not None:
            self.detail_data = self.diary_data[index]
            self.detail_flag = True

    def change_view(self, data):
        self.detail_list = []
        title = Box(data['session_time'], self.img_width, self.img_height)
        title.box_width = self.content_width
        title.box_height = self.content_height
        title.box_color = self.content_box_color
        title.under_line = True
        title.text_size = 0.7

        emotion_list = []
        emotion_list.append((data['emotion1'], data['value1']))
        if data['emotion2']: emotion_list.append((data['emotion2'], data['value2']))
        if data['emotion3']: emotion_list.append((data['emotion3'], data['value3']))
        emotion_list = sorted(emotion_list, key=lambda emotion: emotion[1], reverse=True)

        emotion_text = Box('Diary Tone', self.img_width, self.img_height)
        emotion_text.box_width = self.content_width
        emotion_text.box_height = self.content_height
        emotion_text.box_color = self.content_box_color
        emotion_text.under_line = True
        emotion_text.text_size = 0.9
        self.detail_list = [title, emotion_text]

        for i in range(0, len(emotion_list)):
            emotion_state = Box('{} ({}%)'.format(emotion_list[i][0], emotion_list[i][1]), self.img_width,
                                   self.img_height)
            emotion_state.box_width = self.content_width
            emotion_state.box_height = self.content_height
            emotion_state.box_color = self.content_box_color
            emotion_state.text_size = 0.8
            self.detail_list.append(emotion_state)

        content_text = Box('Content', self.img_width, self.img_height)
        content_text.box_width = self.content_width
        content_text.box_height = self.content_height
        content_text.box_color = self.content_box_color
        content_text.under_line = True
        content_text.upper_line = True
        content_text.text_size = 0.9
        self.detail_list.append(content_text)
        self.content_index = len(self.detail_list) - 1

        content_path = data['content']
        f = open(content_path, 'r')
        content = f.read()
        txt_image_path, txt_image_height = self.get_text_image(content)
        self.detail_content_img = cv2.imread(txt_image_path)
        content_box = Box('Content Box', self.img_width, self.img_height)
        content_box.box_width = self.content_width
        content_box.box_height = txt_image_height
        content_box.box_color = self.content_box_color
        self.detail_list.append(content_box)

    def set_icon(self, img):
        back = Box('Back', self.img_width, self.img_height)
        back.box_width = 50
        back.box_height = 50
        back.box_location_x = self.start_x
        back.box_location_y = self.start_y
        back.put_icon(img, self.back_icon, back.box_location_x, back.box_location_y, white=False)

        start_file = Box('Play', self.img_width, self.img_height)
        start_file.box_width = 50
        start_file.box_height = 50
        start_file.box_location_x = self.start_x + self.content_width - start_file.box_width
        start_file.box_location_y = self.detail_list[self.content_index].box_location_y + 10
        start_file.put_icon(img, self.play_icon, start_file.box_location_x, start_file.box_location_y, white=False,
                            size=30)

        trash = Box('trash', self.img_width, self.img_height)
        trash.box_width = 50
        trash.box_height = 50
        trash.box_location_x = self.start_x + self.content_width - trash.box_width
        trash.box_location_y = self.start_y
        trash.put_icon(img, self.trash_icon, trash.box_location_x, trash.box_location_y, white=False)

        self.icon_list = [back, start_file, trash]

    def in_event(self, x, y):
        for i in range(0, len(self.icon_list)):
            if self.icon_list[i].in_menu(x, y):
                return True
            else:
                continue
        return False

    def detail_event(self, x, y):
        for i in range(0, len(self.icon_list)):
            if self.icon_list[i].in_menu(x, y):
                if i == 0:
                    self.detail_data = None
                    self.detail_flag = False
                    self.detail_list = []
                elif i == 1:
                    self.play_recorded()
                elif i == 2:
                    self.delete_contents()
            # 파일 재생
            else:
                continue
        return False

    def play_recorded(self, f_name=".\\tmp\\temp.wav"):
        def f():
            file_name = self.detail_data['file_name']
            song = AudioSegment.from_wav(file_name)
            play(song)
        Thread(target=f).start()

    def delete_contents(self):
        if self.detail_data:
            diary_session = DiarySession()
            diary_session.delet_diary_session(self.detail_data['session_time']) #DB Row Delete
            if os.path.isfile(self.detail_data['file_name']):
                os.remove(self.detail_data['file_name'])
            if os.path.isfile(self.detail_data['content']):
                os.remove(self.detail_data['content'])
            self.detail_data = None
            self.detail_flag = False
            self.detail_list = []
        else:
            print('No data')

    def get_text_image(self, text):
        length = 30
        content_list = [text[i:i + length] for i in range(0, len(text), length)]
        height = self.content_height * len(content_list)
        width = self.content_width
        txt_img = Image.new('RGB', (width, height), Style.white)
        font = ImageFont.truetype('./font/Sunflower-Bold.ttf', size=25)
        d = ImageDraw.Draw(txt_img)
        for i in range(0, len(content_list)):
            d.text((15, i*self.content_height + 10), content_list[i], fill=Style.black, font=font)
        txt_img.save('./diary/text_img.png')
        return './diary/text_img.png', height

    def draw_text_image(self, img, y, x):
        width = self.detail_content_img.shape[0]
        height = self.detail_content_img.shape[1]
        img[x:x+width, y:y+height] = self.detail_content_img