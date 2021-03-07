class Flags:
    Personal = False
    Public = False
    Assess = False
    Suggest = False
    Evolution = False
    Diary = False
    Setting = False

    TTS_public = False
    assistant_on = False
    zoomed = False
    Double_touched = False

    Joining = False

    Unknown = True
    #######################################

    Btn_capture = False
    BTN_Original = True
    BTN_Voice = False


class InfoMode:
    public_mode = 1
    personal_mode = 1
    assessment_mode = 1
    suggestion_mode = 1
    evolution_mode = 1


class ButtonFlag:
    main_menu = 1
    first_sub = 2
    second_sub = 3

    bottom = 0
    right = 1
    top = 2
    left = 3

    public = 0
    personal = 1
    assessment = 2
    suggestion = 3
    evolution = 4
    diary = 5
    setting = 6

    back_diary = 0
    record = 1
    read = 2

    back = 0
    display = 1
    assistant = 2
    register = 3
    theme = 4

    theme_blue = 0
    theme_purple = 1
    theme_green = 2

    zoom_in = 0
    zoom_out = 1
    zoom_original = 2
    capture = 3
    record_icon =4

    button_location_first = 0

    record_flag = False
    diary_read_flag = False

    display_flag = False
    assistant_flag = False
    register_flag = False
    theme_flag = False


def unknown_flag():
    if Flags.Unknown:
        ButtonFlag.public = 0
        ButtonFlag.personal = 100
        ButtonFlag.assessment = 1
        ButtonFlag.suggestion = 2
        ButtonFlag.evolution = 100
        ButtonFlag.diary = 100
        ButtonFlag.setting = 3
    else:
        ButtonFlag.public = 0
        ButtonFlag.personal = 1
        ButtonFlag.assessment = 2
        ButtonFlag.suggestion = 3
        ButtonFlag.evolution = 4
        ButtonFlag.diary = 5
        ButtonFlag.setting = 6


class Weight:
    weight = 2.5    # To save increment rate of display pixel from (640, 480)
