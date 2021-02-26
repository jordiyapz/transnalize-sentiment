class MyException(Exception):
    def __init__(self, name, message):
        self.name = name
        self.message = message

    def __str__(self):
        return self.message
