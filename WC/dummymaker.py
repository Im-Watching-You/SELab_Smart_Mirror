import random
from datetime import datetime


# To make valid dummy date
# Returns a String , conforms to YYYYMMDDHH format
def dummy_date(start_year=1950):

    try:
        year = random.randrange(start_year, datetime.now().year)

    except ValueError:
        print("Start year can't exceed present date")
        return None

    # check Leap Year
    if year % 4 is 0 and year % 100 is 0 and year % 400 is 0:
        isleap = True
    elif year % 4 is 0 and year % 100 is 0:
        isleap = False

    elif year % 4 is 0:
        isleap = True
    else:
        isleap = False

    month = random.randrange(1, 12)

    if month <= 7:
        # 2
        if month is 2:
            # leap year
            if isleap is True:
                day = random.randrange(1, 29)

            # Not leap year
            elif isleap is False:
                day = random.randrange(1, 28)

        # 1, 3, 5, 7
        if month % 2 is 1:
            day = random.randrange(1,31)

        # 4, 6
        else:
            day = random.randrange(1,30)

    elif month > 7:
        # 9, 11
        if month % 2 is 1:
            day = random.randrange(1, 30)

        # 8, 10, 12
        else:
            day = random.randrange(1, 31)

    # Convert Year, Month, Day into String
    # Format : YYYYMMDD
    year = str(year)

    # MM Format
    if month < 10:
        month = '0' + str(month)
    else:
        month = str(month)

    # DD Format
    if day < 10:
        day = '0' + str(day)
    else:
        day = str(day)

    hour = random.randrange(0,24)
    if hour <10:
        hour = '0' + str(hour)
    else:
        hour = str(hour)
    return year + month + day + hour


# To make Random Name
# Returns a List, made up of [ first name, last name ]
# I referenced 'https://www.ef.com/wwen/english-resources/english-names/' to choose names, which are most common used.
def dummy_name(gender='male'):

    male_first_name = ['Oliver', 'Jack', 'Harry', 'Jacob', 'Charlie', 'Thomas', 'George', 'Oscar', 'James', 'William',
                       'Jake', 'Connor', 'Kyle', 'Joe', 'Reece', 'Rhys', 'Damian', 'Noah', 'Liam', 'Mason', 'Ethan',
                       'Michael', 'Alexander', 'Daniel', 'Robert', 'David', 'Richard', 'Joseph', 'Charles']

    female_first_name = ['Amelia', 'Olivia', 'Isla', 'Emily', 'Poppy', 'Ava', 'Isabella', 'Jessica', 'Lily', 'Sophie',
                         'Margaret', 'Samantha', 'Bethany', 'Elizabeth', 'Joanne', 'Megan', 'Victoria', 'Lauren',
                         'Michelle', 'Tracy', 'Emma', 'Sophia', 'Mia', 'Abigail', 'Madison', 'Charlotte', 'Mary',
                         'Patricia', 'Jennifer', 'Elizabeth', 'Linda', 'Barbara', 'Susan', 'Sarah']

    last_name = ['Smith', 'Murphy', 'Li', 'Jones', 'O\'Kelly', 'Johnson', 'Wilson', 'Williams', 'Lam', 'Brown', 'Walsh',
                 'Martin', 'Taylor', 'Gelbero', 'Davies', 'O\'Brien', 'Miller', 'Roy', 'Bryne', 'Davis', 'Tremblay',
                 'Morton', 'Singh', 'Evans', 'O\'Ryan', 'Garcia', 'Lee', 'White', 'Wang', 'Thomas', 'Rodriguez',
                 'Gagnon', 'Anderson']

    last_name = random.choice(last_name)

    if gender is 'male':
        first_name = random.choice(male_first_name)
        return [first_name, last_name]
    elif gender is 'female':
        first_name =random.choice(female_first_name)
        return [first_name, last_name]
    else:
        print('Type of gender should be \'male\' or \'female\'.')
        return None


# To make Random Phone Number
def dummy_phone_number():
    return '010' + str(random.randrange(1000, 9999)) + str(random.randrange(1000, 9999))


# To make Random Email
# Returns a String 'name@domain.com'
def dummy_email():
    email_domain = ['@gmail.com', '@hotmail.com', '@yahoo.com', '@hanmail.net', '@daum.net', '@naver.com']
    email_name = 'test' + str(random.randrange(1, 10000))

    return email_name + random.choice(email_domain)

if __name__ == '__main__':
    pass
    # for i in range(10):
    #     print(dummy_age())
    # print(dummy_name(2020))          # error case
    # for i in range(20):
    #     print(dummy_name('female'))
    # print(dummy_email())
