class ParamNotFoundException(Exception):
    def __init__(self, msg: str):
        self.msg = msg


class UnsupportedParamException(Exception):
    def __init__(self, msg: str):
        self.msg = msg
