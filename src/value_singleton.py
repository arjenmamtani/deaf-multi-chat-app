# shared.py
# A small wrapper class used to store and retrieve a shared string
class Value:
    def __init__(self):
        # holds the most recent transcription result
        self._value = ""

    def update_str(self, new_value: str):
        #Update the stored value with a new string.
        if isinstance(new_value, str):
            self._value = new_value
        else:
            raise ValueError("Input must be a string")

    def get_value(self):
        #Retrieve the stored value.
        return self._value


# Create a global instance of Value to be used in other files
shared_value = Value()