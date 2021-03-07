class StringMaker:

    @staticmethod
    def list_to_string_with_comma(list):
        """
        METHOD To utilize for sql statement creation
        INPUT : LIST OF KEYS OR VALUES (on purpose)
        RETURN EXAMPLE : (key1), (key2), (key3) ...
        """
        sentence = ''
        for i in range(len(list)):
            if i is not len(list) - 1:
                sentence += list[i] + ', '
            else:
                sentence += list[i]
        return sentence

    @staticmethod
    def list_to_string_with_comma_quote(list):
        """
        METHOD To utilize for sql statement creation
        INPUT : LIST OF KEYS OR VALUES (on purpose)
        RETURN EXAMPLE : '(key1)', '(key2)', '(key3)' ...
        """
        sentence = ''
        for i in range(len(list)):
            if i is not len(list) - 1:
                sentence += '\'' + list[i] + '\'' + ', '
            else:
                sentence += '\'' + list[i] + '\''
        return sentence

    @staticmethod
    def list_to_string_with_comma_quote_num(list):
        """
        METHOD To utilize for sql statement creation
        INPUT : LIST OF KEYS OR VALUES (on purpose)
        RETURN EXAMPLE : '(key1)', '(key2)', '(key3)' ...
        """
        sentence = ''
        for i in range(len(list)):
            if i is not len(list) - 1:
                sentence += ' ' + list[i] + ' ' + ', '
            else:
                sentence += ' ' + list[i] + ' '
        return sentence

    @staticmethod
    def key_equation_value(dict):
        """
        METHOD To utilize for sql statement creation
        INPUT : DICTIONARY
        RETURN EXAMPLE : (key1) = (value1), (key2) = (value2), (key3) = (value3) ...
        """
        sentence = ""
        try:
            for i in range(len(dict)):
                if i is not len(dict) - 1:
                    sentence += list(dict.keys())[i] + " = \'" + list(dict.values())[i] + "\', "
                else:
                    sentence += list(dict.keys())[i] + " = \'" + list(dict.values())[i] + "\'"
            return sentence
        except Exception as e:
            print(e)
            print('Check your type in \'dict_info\' again.')
            print("Value should be String.")
            return ""
# -------------------- Test Code --------------------
#
# a = {'name': 'park', 'phone': '01051310025'}
# b = {'name': 'Lee'}
# c = {'name': 'Kwon', 'phone': '020', 'gender': 'male'}
#
# print("\n1. Dictionary a")
# StringMaker.list_to_string_with_comma(list(a.keys()))
# StringMaker.key_equation_value(a)
#
# print("\n2. Dictionary b")
# StringMaker.list_to_string_with_comma(list(b.keys()))
# StringMaker.key_equation_value(b)
#
# print("\n3. Dictionary c")
# StringMaker.list_to_string_with_comma(list(c.keys()))
# StringMaker.key_equation_value(c)
